import os
import io
import shutil
import pathlib
import random
import cv2
import numpy as np
import gradio as gr
from PIL import Image

# Import remote inference helpers
from backend import (
    run_remote_inference, qwen_image_edit_remote, generate_z_image_turbo_remote,
    start_keepalive_thread, record_frontend_heartbeat, set_wan_heartbeat_state,
    set_qwen_heartbeat_state, set_z_image_heartbeat_state
)
from studio_pipeline import (
    StudioPipeline,
    ASPECT_RATIO_PRESETS,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_MODEL,
    generate_three_studio_seeds,
    restore_extracted_video_frame,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 16  # 16 + 1 = 17 frames (~1.0s)
MAX_FRAMES_MODEL = 80  # 80 + 1 = 81 frames (~5.0s, native max capacity of Wan 2.2)

MIN_DURATION = 1.0
MAX_DURATION = 5.0
DEFAULT_DURATION = 3.5

SCHEDULERS = [
    "UniPCMultistep",
    "FlowMatchEulerDiscrete",
    "SASolver",
    "DEISMultistep",
    "DPMSolverMultistepInverse",
    "DPMSolverMultistep",
    "DPMSolverSinglestep",
]

default_prompt_qwen = "camera capturing overall people in a complete full frame, 4k resolution, preserve face details"
default_prompt_i2v = "cinematic natural motion, preserve subject appearance and facial identity, smooth animation, photorealistic high resolution"
default_negative_prompt = (
    "blurry, low quality, low resolution, worst quality, jpeg artifacts, compression artifacts, "
    "pixelated, out of focus, grainy, noisy, oversaturated, undersaturated, bad anatomy, deformed, "
    "disfigured, extra limbs, missing limbs, extra fingers, missing fingers, fused fingers, mutated hands, "
    "poorly drawn hands, poorly drawn face, malformed limbs, long neck, distorted proportions, "
    "asymmetrical eyes, cross-eyed, watermark, signature, text, logo, cropped, cut off, frame, "
    "border, duplicate, cloned face, mutation, ugly, tiling, "
    "face morphing, changing face, altered facial features, changing identity, inconsistent face, "
    "shifting features, melting face, face distortion, facial transformation, mutating face, "
    "temporal flickering, pixel noise, pixel degradation, digital artifacts, warping"
)

CANVAS_MODES = [
    "Scale base image",
    "Custom size",
    "Portrait (768x1024)",
    "Landscape (1024x768)",
    "Square (1024x1024)",
]

CANVAS_FIT_MODES = ["Cover", "Contain", "Stretch", "Pad"]

# ── Helper functions ───────────────────────────────────────────────────────────

get_timestamp_js = """
function() {
    const video = document.querySelector('#generated-video video');
    if (video) {
        return video.currentTime;
    }
    return 0;
}
"""


def extract_frame(video_path, timestamp, reference_path=None, restore=True):
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_num = int(float(timestamp) * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and target_frame_num >= total_frames:
        target_frame_num = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if restore:
            ref_pil = None
            if reference_path and isinstance(reference_path, str) and os.path.exists(reference_path):
                try:
                    ref_pil = Image.open(reference_path).convert("RGB")
                except Exception:
                    ref_pil = None
            elif isinstance(reference_path, Image.Image):
                ref_pil = reference_path
            return restore_extracted_video_frame(rgb_frame, reference_pil=ref_pil)
        return rgb_frame
    return None


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    aspect_ratio = width / height
    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM
    image_to_resize = image
    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))
    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF
    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))
    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def get_num_frames(duration_seconds: float):
    """Calculates valid Wan 2.2 frame count strictly adhering to 3D VAE 4k+1 math (17 to 81 frames)."""
    raw_frames = int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
    ))
    k = max(MIN_FRAMES_MODEL // 4, min(MAX_FRAMES_MODEL // 4, round(raw_frames / 4)))
    return 1 + (k * 4)


def clear_all_outputs():
    """Safely cleans all files and subdirectories from the local outputs/ folder."""
    outputs_dir = pathlib.Path(__file__).parent / "outputs"
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)
        try:
            gr.Info("Outputs folder is already empty.")
        except Exception:
            pass
        return "Outputs folder is already empty."

    deleted_count = 0
    for item in list(outputs_dir.iterdir()):
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                deleted_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                deleted_count += 1
        except Exception as e:
            print(f"[app] Error deleting {item}: {e}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    msg = f"Cleared {deleted_count} item(s) from outputs folder."
    try:
        gr.Info(msg)
    except Exception:
        pass
    return msg


# ── Qwen Image Edit Rapid AIO ────────────────────────────────────────────────
def run_qwen_generate(
    base_image,
    ref_image_1,
    ref_image_2,
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    steps,
    true_cfg_scale,
    guidance_scale,
    num_images,
    custom_width,
    custom_height,
    progress=gr.Progress(track_tqdm=True),
):
    active_prompt = prompt.strip() if (prompt and prompt.strip()) else default_prompt_qwen
    if base_image is None and not active_prompt:
        raise gr.Error("Please upload an image or enter an editing prompt.")

    count = max(1, min(6, int(num_images)))
    progress(0.05, desc=f"Preparing request for Qwen Image Edit ({count} image(s)) on Modal...")

    ref_images = [r for r in [ref_image_1, ref_image_2] if r is not None]
    actual_seed = random.randint(0, MAX_SEED - count) if randomize_seed else int(seed)

    # If no base image, use a blank canvas (pure text-to-image mode)
    if base_image is None:
        base_image = Image.new("RGB", (1024, 1024), (255, 255, 255))

    try:
        result_bytes_list = qwen_image_edit_remote(
            base_image=base_image,
            ref_images=ref_images,
            prompt=active_prompt,
            negative_prompt=negative_prompt.strip() if negative_prompt else "",
            seed=actual_seed,
            steps=int(steps),
            true_cfg_scale=float(true_cfg_scale),
            guidance_scale=float(guidance_scale),
            num_images=count,
            custom_width=int(custom_width),
            custom_height=int(custom_height),
        )
    except Exception as e:
        raise gr.Error(f"Qwen Image Edit generation failed: {e}")

    progress(0.95, desc="Receiving result images...")
    pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in result_bytes_list]
    
    seeds_str = str(actual_seed) if count == 1 else f"{actual_seed} .. {actual_seed + count - 1}"
    first_img = pil_images[0] if pil_images else None

    progress(1.0, desc="Done!")
    return pil_images, seeds_str, first_img


# ── AI Video Director & Prompt Reasoning ──────────────────────────────────────

def generate_director_prompt(
    base_prompt: str,
    direction_style: str,
    camera_movement: str,
    motion_intensity: int,
    duration_seconds: float,
):
    """
    Synthesizes a production-grade, temporally-staged prompt tailored for Wan 2.2 DiT.
    Decomposes the generation into invariant anchors and multi-phase timelines to eliminate
    facial morphing, pixel tearing, and temporal drift.
    """
    clean_base = base_prompt.strip() if base_prompt else "cinematic natural motion"
    clean_base = clean_base.replace("make this image come alive, ", "").replace("cinematic motion, ", "")

    intensity_desc = {
        1: "subtle, micro-movements only, whisper-quiet motion",
        2: "gentle, controlled, organic natural motion",
        3: "moderate, expressive, fluid cinematic motion",
        4: "energetic, pronounced dynamic motion",
        5: "high-energy, rapid, dramatic motion",
    }.get(int(motion_intensity), "gentle, controlled, organic natural motion")

    dur = float(duration_seconds)
    p1_end = round(dur * 0.35, 1)
    p2_end = round(dur * 0.8, 1)

    if "Portrait & Identity Lock" in direction_style:
        invariants = (
            "[Subject Invariant Anchors]: strictly preserve source subject's facial identity, exact bone structure, "
            "eye shape, nose geometry, jawline, and skin texture without alteration or morphing throughout."
        )
        timeline = (
            f"[Timeline Stages]: "
            f"Phase 1 (0.0s–{p1_end}s): natural rhythmic breathing, steady eye contact, locked head pose; "
            f"Phase 2 ({p1_end}s–{p2_end}s): subtle 5-degree head tilt, soft natural smile, subtle hair flutter ({intensity_desc}); "
            f"Phase 3 ({p2_end}s–{dur}s): smooth return to resting pose, serene expression, invariant facial features."
        )
        rec_guidance = 2.25
        rec_steps = 10
    elif "Expressive Head Turn" in direction_style:
        invariants = (
            "[Subject Invariant Anchors]: consistent facial structure, identical facial keypoints and eyes, "
            "photorealistic human face, zero facial warping."
        )
        timeline = (
            f"[Timeline Stages]: "
            f"Phase 1 (0.0s–{p1_end}s): subject begins looking forward with gentle expression; "
            f"Phase 2 ({p1_end}s–{p2_end}s): smooth continuous head rotation towards the side, natural blinking ({intensity_desc}); "
            f"Phase 3 ({p2_end}s–{dur}s): settling at 3/4 profile angle, calm steady gaze, locked facial proportions."
        )
        rec_guidance = 2.0
        rec_steps = 9
    elif "Cinematic Camera Orbit" in direction_style:
        invariants = (
            "[Spatial & Subject Invariants]: main subject pose stabilized, solid 3D depth, "
            "coherent environment geometry, static subject features."
        )
        timeline = (
            f"[Timeline Stages]: "
            f"Phase 1 (0.0s–{p1_end}s): smooth camera arc initiation, parallax effect across background; "
            f"Phase 2 ({p1_end}s–{p2_end}s): steady continuous orbital rotation, maintaining perfect focus ({intensity_desc}); "
            f"Phase 3 ({p2_end}s–{dur}s): gentle deceleration, stable cinematic framing."
        )
        rec_guidance = 2.0
        rec_steps = 8
    elif "Micro-Motion Natural" in direction_style:
        invariants = (
            "[Scene Invariants]: photorealistic fidelity, unchanged character anatomy and facial features, "
            "no structural shifting or morphing."
        )
        timeline = (
            f"[Timeline Stages]: "
            f"Phase 1 (0.0s–{p1_end}s): subtle ambient breeze, gentle fabric ripple; "
            f"Phase 2 ({p1_end}s–{p2_end}s): soft hair strands moving naturally, calm breathing ({intensity_desc}); "
            f"Phase 3 ({p2_end}s–{dur}s): steady resting pose, smooth continuous atmosphere."
        )
        rec_guidance = 1.75
        rec_steps = 8
    else:  # Dynamic Action
        invariants = (
            "[Kinematic Invariants]: cohesive physical trajectory, stable silhouette, "
            "zero frame tearing, sharp motion vectors."
        )
        timeline = (
            f"[Timeline Stages]: "
            f"Phase 1 (0.0s–{p1_end}s): initial acceleration with realistic inertia; "
            f"Phase 2 ({p1_end}s–{p2_end}s): fluid kinematic motion through the frame ({intensity_desc}); "
            f"Phase 3 ({p2_end}s–{dur}s): continuous momentum, clean temporal consistency."
        )
        rec_guidance = 2.0
        rec_steps = 8

    camera_clause = f"[Camera Trajectory]: {camera_movement}, fixed focal length, 4k ultra-detailed, photorealistic cinematic lighting."

    director_prompt = f"{clean_base}, {invariants} {timeline} {camera_clause}"

    director_negative = (
        "face morphing, changing face, altered facial features, changing identity, inconsistent face, "
        "shifting features, melting face, face distortion, facial transformation, mutating face, "
        "asymmetrical eyes, cross-eyed, wandering gaze, sudden smile distortion, "
        "temporal flickering, pixel noise, pixel degradation, digital artifacts, warping, tearing, "
        "blurry, low quality, worst quality, bad anatomy, deformed"
    )

    try:
        gr.Info("🪄 AI Director Prompt Synthesized! Locked invariant anchors and temporal phases.")
    except Exception:
        pass

    return director_prompt, director_negative, rec_guidance, rec_steps


# ── Video Generation ──────────────────────────────────────────────────────────

def generate_video(
    input_image,
    last_image,
    prompt,
    steps=8,
    negative_prompt=default_negative_prompt,
    duration_seconds=3.5,
    guidance_scale=2.0,
    guidance_scale_2=1.5,
    seed=42,
    randomize_seed=False,
    quality=6,
    scheduler="UniPCMultistep",
    flow_shift=3.0,
    frame_multiplier=16,
    safe_mode=False,
    video_component=True,
    lora_repo_or_path="",
    lora_scale=0.8,
    deflicker_frames=True,
    anti_noise_guard=True,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        raise gr.Error("Please upload or edit an input image first.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    processed_last_image = None
    if last_image:
        processed_last_image = resize_and_crop_to_match(last_image, resized_image)

    if anti_noise_guard:
        guard_cfg = StudioPipeline.apply_anti_noise_and_rotation_guard(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "flow_shift": flow_shift,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
            },
            duration_seconds=duration_seconds,
            has_last_frame=(processed_last_image is not None),
            anti_noise_guard=True,
        )
        prompt = guard_cfg.get("prompt", prompt)
        negative_prompt = guard_cfg.get("negative_prompt", negative_prompt)
        steps = guard_cfg.get("steps", steps)
        flow_shift = guard_cfg.get("flow_shift", flow_shift)
        guidance_scale = guard_cfg.get("guidance_scale", guidance_scale)
        guidance_scale_2 = guard_cfg.get("guidance_scale_2", guidance_scale_2)

    video_path, task_name = run_remote_inference(
        resized_image=resized_image,
        processed_last_image=processed_last_image,
        prompt=prompt,
        steps=steps,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        current_seed=current_seed,
        scheduler_name=scheduler,
        flow_shift=flow_shift,
        frame_multiplier=frame_multiplier,
        quality=quality,
        duration_seconds=duration_seconds,
        safe_mode=safe_mode,
        lora_repo_or_path=lora_repo_or_path,
        lora_scale=lora_scale,
        deflicker_frames=deflicker_frames,
        progress=progress
    )

    print(f"Modal generation complete: {task_name}")
    return (video_path if video_component else None), video_path, current_seed


def extract_selected_image(selected_img, gallery=None):
    img = selected_img
    if img is None and gallery:
        if isinstance(gallery, list) and len(gallery) > 0:
            img = gallery[0]
        else:
            img = gallery

    if img is None:
        return None

    if isinstance(img, tuple):
        img = img[0]
    elif isinstance(img, dict) and "image" in img:
        img = img["image"]

    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:
            return img
    return img


def route_qwen_to_base(selected_img, gallery=None):
    return extract_selected_image(selected_img, gallery)


def route_qwen_to_ref1(selected_img, gallery=None):
    return extract_selected_image(selected_img, gallery)


def route_qwen_to_ref2(selected_img, gallery=None):
    return extract_selected_image(selected_img, gallery)


def route_qwen_to_video(selected_img, gallery=None):
    return extract_selected_image(selected_img, gallery)


def route_qwen_to_video_last(selected_img, gallery=None):
    return extract_selected_image(selected_img, gallery)


def use_generated_as_video_input(generated_image):
    return extract_selected_image(generated_image, None)


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8em !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 0.95em;
    margin-bottom: 16px !important;
}

.qwen-panel {
    background: linear-gradient(145deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 0 30px rgba(139,92,246,0.1);
    margin-bottom: 16px !important;
}

.wan-panel {
    background: linear-gradient(145deg, rgba(14,165,233,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(14,165,233,0.35);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 0 30px rgba(14,165,233,0.1);
    margin-bottom: 16px !important;
}

.studio-panel {
    background: linear-gradient(145deg, rgba(236,72,153,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(236,72,153,0.35);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 0 30px rgba(236,72,153,0.1);
    margin-bottom: 16px !important;
}

.tool-accordion {
    border-radius: 14px !important;
    margin-bottom: 16px !important;
}

.toolbar-row {
    margin-bottom: 12px !important;
    display: flex;
    gap: 8px !important;
}

.section-header {
    font-size: 1.05em !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
    color: #e2e8f0 !important;
}

.step-hint {
    color: #94a3b8;
    font-size: 0.85em;
    margin-top: 2px;
    margin-bottom: 8px;
}

.director-panel {
    background: linear-gradient(145deg, rgba(168,85,247,0.12), rgba(99,102,241,0.08));
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px !important;
}

#hidden-timestamp, #hidden-timestamp-last {
    opacity: 0; height: 0px; width: 0px;
    margin: 0; padding: 0;
    overflow: hidden; position: absolute; pointer-events: none;
}

/* Drag and Drop Styling & Visual Feedback */
.drag-active-highlight {
    outline: 3px dashed #38bdf8 !important;
    outline-offset: -3px !important;
    background-color: rgba(56, 189, 248, 0.22) !important;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.6) !important;
    border-radius: 10px !important;
    transition: all 0.15s ease-in-out !important;
}

.gallery img, .thumbnail-item img, [data-testid="gallery"] img {
    cursor: grab !important;
    -webkit-user-drag: element !important;
}

.gallery img:active {
    cursor: grabbing !important;
}
"""

DRAG_DROP_JS = """
<script>
(function() {
    console.log("[WanStudio] Initializing Drag & Drop handler for Image Editor...");
    let draggedImageSrc = null;

    // 1. Capture dragstart on any gallery image or preview
    document.addEventListener('dragstart', function(e) {
        let img = null;
        if (e.target && e.target.tagName === 'IMG') {
            img = e.target;
        } else if (e.target && e.target.querySelector) {
            img = e.target.querySelector('img');
        }

        if (img && img.src) {
            draggedImageSrc = img.src;
            if (e.dataTransfer) {
                e.dataTransfer.setData('text/plain', img.src);
                e.dataTransfer.setData('text/uri-list', img.src);
                e.dataTransfer.effectAllowed = 'copyMove';
            }
        }
    }, true);

    // 2. Highlight dropzones on dragover
    document.addEventListener('dragover', function(e) {
        const dropzone = e.target.closest('.image-container, .dropzone, [data-testid="image"], .upload-container, .gradio-image, .image-frame');
        if (dropzone) {
            e.preventDefault();
            if (e.dataTransfer) {
                e.dataTransfer.dropEffect = 'copy';
            }
            dropzone.classList.add('drag-active-highlight');
        }
    }, true);

    // 3. Remove highlight on dragleave
    document.addEventListener('dragleave', function(e) {
        const dropzone = e.target.closest('.image-container, .dropzone, [data-testid="image"], .upload-container, .gradio-image, .image-frame');
        if (dropzone && (!e.relatedTarget || !dropzone.contains(e.relatedTarget))) {
            dropzone.classList.remove('drag-active-highlight');
        }
    }, true);

    // 4. Handle drop event
    document.addEventListener('drop', async function(e) {
        const dropzone = e.target.closest('.image-container, .dropzone, [data-testid="image"], .upload-container, .gradio-image, .image-frame');
        if (dropzone) {
            dropzone.classList.remove('drag-active-highlight');
            
            // If dragging native files from desktop file explorer, let browser handle
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                return;
            }

            let src = draggedImageSrc;
            if (!src && e.dataTransfer) {
                src = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
            }

            if (src) {
                e.preventDefault();
                e.stopPropagation();
                try {
                    const response = await fetch(src);
                    const blob = await response.blob();
                    const file = new File([blob], "input_dropped.png", { type: blob.type || "image/png" });
                    
                    const fileInput = dropzone.querySelector('input[type="file"]') || 
                                     dropzone.closest('.gradio-image, .block, [data-testid="image"]')?.querySelector('input[type="file"]');
                    
                    if (fileInput) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                        fileInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                } catch (err) {
                    console.error("[WanStudio] Error loading dropped image:", err);
                }
            }
        }
    }, true);

    // Ensure all gallery output images have draggable="true" and grab cursor
    setInterval(function() {
        document.querySelectorAll('.gallery img, [data-testid="gallery"] img, .thumbnail-item img, .preview img').forEach(function(img) {
            if (!img.getAttribute('draggable') || img.getAttribute('draggable') !== 'true') {
                img.setAttribute('draggable', 'true');
            }
        });
    }, 1000);
})();
</script>
"""

# ── Studio (Tool 3) Handlers ──────────────────────────────────────────────────
# Kept at module level so they can be unit-tested without launching Gradio.

INITIAL_STUDIO_SEEDS = generate_three_studio_seeds()


def roll_three_studio_seeds():
    seeds = generate_three_studio_seeds()
    return seeds["z_image_seed"], seeds["qwen_seed"], seeds["wan_seed"]


def _studio_progress(progress):
    def cb(pct, msg):
        try:
            progress(pct, desc=msg)
        except Exception:
            pass
    return cb


def _studio_gallery_items(story_data):
    chars = [
        (c.get("portrait_path") or c.get("portrait_pil"), f"{c.get('name')} (Seed: {c.get('portrait_seed')})")
        for c in (story_data or {}).get("characters", []) if (c.get("portrait_path") or c.get("portrait_pil"))
    ]
    scenes = []
    for s in (story_data or {}).get("scenes", []):
        kf = s.get("keyframe_path") or s.get("keyframe_pil")
        if kf:
            scenes.append(
                (kf, f"Scene {s.get('scene_number')}: {s.get('title')} (Seed: {s.get('keyframe_seed', '')})")
            )
        last_kf = s.get("last_keyframe_path") or s.get("last_keyframe_pil")
        if last_kf:
            end_seed = s.get("last_keyframe_seed", s.get("keyframe_seed", ""))
            scenes.append(
                (last_kf, f"Scene {s.get('scene_number')}: {s.get('title')} — End Frame (Seed: {end_seed})")
            )
    return chars, scenes


def _gallery_index_to_scene_index(gallery_idx: int, story_data: dict) -> int:
    scenes = (story_data or {}).get("scenes", [])
    if not scenes:
        return 0
    cursor = 0
    for idx, s in enumerate(scenes):
        count = (1 if (s.get("keyframe_path") or s.get("keyframe_pil")) else 0) + (1 if (s.get("last_keyframe_path") or s.get("last_keyframe_pil")) else 0)
        if count > 0:
            if gallery_idx < cursor + count:
                return idx
            cursor += count
    return max(0, min(len(scenes) - 1, int(gallery_idx)))


def _studio_director_summary(scenes):
    rows = []
    for s in scenes or []:
        d = s.get("director") or {}
        if d:
            rows.append(
                f"| {s.get('scene_number')} | {s.get('title', '')} | {d.get('camera_movement', '')} | "
                f"{d.get('motion_intensity', '')} | {d.get('num_frames', '')} | {d.get('source', '')} |"
            )
    if not rows:
        return ""
    return ("\n\n| # | Shot | Camera | Intensity | Frames | Director |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows))


def _character_choices(story_data):
    chars = (story_data or {}).get("characters", [])
    return [f"{i+1}. {c.get('name', f'Character {i+1}')} ({c.get('id', f'char_{i+1}')})" for i, c in enumerate(chars)]


def _scene_choices(story_data):
    scenes = (story_data or {}).get("scenes", [])
    return [f"Scene {s.get('scene_number', i+1)}: {s.get('title', f'Shot {i+1}')}" for i, s in enumerate(scenes)]


def _parse_selector_index(choice_label, choices_list):
    if not choices_list:
        return 0
    if isinstance(choice_label, int):
        return max(0, min(len(choices_list) - 1, choice_label))
    if choice_label in choices_list:
        return choices_list.index(choice_label)
    if isinstance(choice_label, str):
        for idx, c in enumerate(choices_list):
            if c.startswith(choice_label) or choice_label in c:
                return idx
    return 0


def studio_load_character_fields(choice_label, current_state, default_seed=42):
    chars = (current_state or {}).get("characters", [])
    choices = _character_choices(current_state)
    if not chars:
        return "", "", "", "", "", "", int(default_seed), None
    idx = _parse_selector_index(choice_label, choices)
    c = chars[idx]
    img = c.get("portrait_path") if c.get("portrait_path") and os.path.exists(c.get("portrait_path")) else None
    return (
        c.get("name", ""),
        c.get("face", ""),
        c.get("hair", ""),
        c.get("clothing", ""),
        c.get("signature_details", ""),
        c.get("z_image_prompt", ""),
        int(c.get("portrait_seed", default_seed)),
        img,
    )


def studio_save_character_text_edits(
    choice_label, name, face, hair, clothing, sig_details, z_prompt, current_state
):
    chars = list((current_state or {}).get("characters", []))
    choices = _character_choices(current_state)
    if not chars:
        return StudioPipeline.serializable(current_state or {}), "### ⚠️ Generate a story first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    c = dict(chars[idx])
    c["name"] = name
    c["face"] = face
    c["hair"] = hair
    c["clothing"] = clothing
    c["signature_details"] = sig_details
    c["z_image_prompt"] = z_prompt
    chars[idx] = c
    current_state["characters"] = chars
    return (
        StudioPipeline.serializable(current_state),
        f"### ✅ Saved description updates for Character #{idx+1} (`{name}`).",
        current_state,
    )


def studio_redo_single_character(
    choice_label, name, face, hair, clothing, sig_details, z_prompt, char_seed,
    api_key, model, current_state, progress=gr.Progress(track_tqdm=True),
):
    chars = (current_state or {}).get("characters", [])
    choices = _character_choices(current_state)
    if not chars:
        return [], None, {}, "### ⚠️ Please run Phase 1 (Story Generation) first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        updated_char = pipeline.regenerate_single_character(
            story_data=current_state,
            char_index=idx,
            updated_fields={
                "name": name,
                "face": face,
                "hair": hair,
                "clothing": clothing,
                "signature_details": sig_details,
                "z_image_prompt": z_prompt,
            },
            seed=int(char_seed),
            progress_cb=_studio_progress(progress),
        )
        char_items, _ = _studio_gallery_items(current_state)
        status = (
            f"### 🔄 Character Redo Complete: `{updated_char.get('name')}`\n"
            f"Regenerated with Z-Image Turbo using seed `{int(char_seed)}`."
        )
        return char_items, updated_char.get("portrait_path"), StudioPipeline.serializable(current_state), status, current_state
    except Exception as e:
        char_items, _ = _studio_gallery_items(current_state)
        return char_items, None, StudioPipeline.serializable(current_state), f"### ❌ Character Redo Error\n`{e}`", current_state


def studio_load_scene_fields(choice_label, current_state, default_seed=42):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return "", "", "", "", int(default_seed), None
    idx = _parse_selector_index(choice_label, choices)
    s = scenes[idx]
    img = s.get("keyframe_path") if s.get("keyframe_path") and os.path.exists(s.get("keyframe_path")) else None
    char_ids_str = ", ".join(s.get("character_ids", []))
    return (
        s.get("title", ""),
        s.get("scenario", ""),
        s.get("qwen_scene_prompt", ""),
        char_ids_str,
        int(s.get("keyframe_seed", default_seed)),
        img,
    )


def studio_load_scene_fields_dual(choice_label, current_state, default_seed=42):
    base_fields = studio_load_scene_fields(choice_label, current_state, default_seed=default_seed)
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return (*base_fields, "", None, int((default_seed + 10007) % (2**31 - 1)), "N/A")
    idx = _parse_selector_index(choice_label, choices)
    s = scenes[idx]
    end_img = (
        s.get("last_keyframe_path")
        if s.get("last_keyframe_path") and os.path.exists(s.get("last_keyframe_path"))
        else None
    )
    start_seed = int(s.get("keyframe_seed", default_seed))
    end_seed = int(s.get("last_keyframe_seed", (start_seed + 10007) % (2**31 - 1)))
    delta_val = s.get("pose_delta")
    delta_str = f"🏃 Pose Delta: {delta_val:.1f} (High Motion)" if delta_val is not None else "Pending generation"
    return (*base_fields, s.get("qwen_end_scene_prompt", ""), end_img, end_seed, delta_str)


def studio_save_scene_text_edits(
    choice_label, title, scenario, qwen_prompt, char_ids_str, current_state, qwen_end_prompt=None
):
    scenes = list((current_state or {}).get("scenes", []))
    choices = _scene_choices(current_state)
    if not scenes:
        return StudioPipeline.serializable(current_state or {}), "### ⚠️ Generate a story first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    s = dict(scenes[idx])
    s["title"] = title
    s["scenario"] = scenario
    s["qwen_scene_prompt"] = qwen_prompt
    if qwen_end_prompt is not None:
        s["qwen_end_scene_prompt"] = qwen_end_prompt
    if char_ids_str and str(char_ids_str).strip():
        s["character_ids"] = [x.strip() for x in str(char_ids_str).split(",") if x.strip()]
    scenes[idx] = s
    current_state["scenes"] = scenes
    return (
        StudioPipeline.serializable(current_state),
        f"### ✅ Saved description updates for Scene #{idx+1} (`{title}`).",
        current_state,
    )


def studio_redo_single_scene(
    choice_label, title, scenario, qwen_prompt, char_ids_str, scene_seed,
    aspect_ratio, api_key, model, current_state,
    qwen_end_prompt=None, which_frame="both", auto_qa_retry=False,
    scene_end_seed=None, direct_end_frame=True,
    progress=gr.Progress(track_tqdm=True),
):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return [], None, None, {}, "### ⚠️ Please run Phase 1 & Phase 2 first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    char_ids = [x.strip() for x in str(char_ids_str or "").split(",") if x.strip()] or scenes[idx].get("character_ids", [])
    updated_fields = {
        "title": title,
        "scenario": scenario,
        "qwen_scene_prompt": qwen_prompt,
        "character_ids": char_ids,
    }
    if qwen_end_prompt is not None:
        updated_fields["qwen_end_scene_prompt"] = qwen_end_prompt

    start_seed = int(scene_seed)
    end_s = int(scene_end_seed) if (scene_end_seed is not None and int(scene_end_seed) != start_seed) else (start_seed + 10007) % (2**31 - 1)

    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        updated_scene = pipeline.regenerate_single_scene_keyframe(
            story_data=current_state,
            scene_index=idx,
            updated_fields=updated_fields,
            seed=start_seed,
            end_seed=end_s,
            which_frame=str(which_frame or "both"),
            generate_end_frame=True,
            direct_end_frame=bool(direct_end_frame),
            auto_qa_retry=bool(auto_qa_retry),
            aspect_ratio=aspect_ratio,
            progress_cb=_studio_progress(progress),
        )
        _, scene_items = _studio_gallery_items(current_state)
        qa_note = ""
        if updated_scene.get("qa_report"):
            qa = updated_scene["qa_report"]
            qa_note = f"\n- **Gemma Vision QA Score:** `{qa.get('score')}/10` ({'✅ Passed' if qa.get('passed') else '⚠️ Flagged'}) — {qa.get('issues')}"
        delta_val = updated_scene.get("pose_delta")
        delta_note = f"\n- **🏃 Pose Motion Delta:** `{delta_val:.1f}` ({'✅ High Motion' if (delta_val or 0) >= 18.0 else '⚠️ Low Motion'})" if delta_val is not None else ""
        status = (
            f"### 🔄 Scene #{idx+1} Keyframe Redo Complete (`{which_frame}`): `{updated_scene.get('title')}`\n"
            f"- **Start Seed:** `{start_seed}` | **End Seed:** `{updated_scene.get('last_keyframe_seed', end_s)}`"
            + delta_note
            + qa_note
        )
        return scene_items, updated_scene.get("keyframe_path"), updated_scene.get("last_keyframe_path"), StudioPipeline.serializable(current_state), status, current_state
    except Exception as e:
        _, scene_items = _studio_gallery_items(current_state)
        return scene_items, None, None, StudioPipeline.serializable(current_state), f"### ❌ Scene Redo Error\n`{e}`", current_state


def studio_verify_scene_qa(choice_label, api_key, model, current_state):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return StudioPipeline.serializable(current_state or {}), "### ⚠️ Please generate scene keyframes first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    pipeline = StudioPipeline(api_key=api_key, model=model)
    report = pipeline.verify_scene_frames_with_vlm(
        scenes[idx],
        current_state.get("characters", []),
        art_style=current_state.get("art_style", ""),
    )
    badge = "✅ PASSED" if report.get("passed") else "⚠️ NEEDS ATTENTION"
    status = (
        f"### 🔍 Gemma 4: 31B Vision QA Critic — Scene #{idx+1} ({badge} · Score `{report.get('score')}/10`)\n"
        f"- **Findings:** {report.get('issues')}\n"
        + (f"- **Suggested Start Prompt Fix:** `{report['suggested_start_prompt']}`\n" if report.get("suggested_start_prompt") else "")
        + (f"- **Suggested End Prompt Fix:** `{report['suggested_end_prompt']}`" if report.get("suggested_end_prompt") else "")
    )
    return StudioPipeline.serializable(current_state), status, current_state


def studio_direct_single_scene_end_frame(choice_label, api_key, model, current_state):
    scenes = list((current_state or {}).get("scenes", []))
    choices = _scene_choices(current_state)
    if not scenes:
        return "", StudioPipeline.serializable(current_state or {}), "### ⚠️ Please generate scene keyframes first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    scene = dict(scenes[idx])

    keyframe_pil = scene.get("keyframe_pil")
    if keyframe_pil is None and scene.get("keyframe_path") and os.path.exists(scene["keyframe_path"]):
        try:
            keyframe_pil = Image.open(scene["keyframe_path"]).convert("RGB")
        except Exception:
            keyframe_pil = None
    if keyframe_pil is None:
        return scene.get("qwen_end_scene_prompt", ""), StudioPipeline.serializable(current_state), "### ⚠️ Please generate the Start Frame (0.0s) first so Gemma Director can inspect it.", current_state

    pipeline = StudioPipeline(api_key=api_key, model=model)
    directed_prompt = pipeline.direct_scene_end_frame_with_gemma(
        scene=scene,
        keyframe_pil=keyframe_pil,
        characters=current_state.get("characters", []),
        story_meta=current_state.get("metadata", {}),
        art_style=current_state.get("art_style", ""),
        use_vision=True,
    )
    if directed_prompt:
        scene["qwen_end_scene_prompt"] = directed_prompt
        scenes[idx] = scene
        current_state["scenes"] = scenes
        status = (
            f"### 🎬 Gemma Director Directed Scene #{idx+1} End Frame Pose:\n"
            f"> {directed_prompt}\n\n"
            f"Prompt updated in the editor! Click **'🔄 Redo Selected Scene Frame(s)'** with mode='end' to generate the updated frame."
        )
        return directed_prompt, StudioPipeline.serializable(current_state), status, current_state
    else:
        return scene.get("qwen_end_scene_prompt", ""), StudioPipeline.serializable(current_state), "### ⚠️ Could not direct end frame.", current_state


def studio_load_clip_fields(choice_label, current_state, default_seed=42):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return "", "", "", int(default_seed), None
    idx = _parse_selector_index(choice_label, choices)
    s = scenes[idx]
    d = s.get("director") or {}
    vid = s.get("video_path") if s.get("video_path") and os.path.exists(s.get("video_path")) else None
    return (
        s.get("wan_motion_prompt", d.get("subject_action", "")),
        d.get("camera_movement", s.get("camera_movement", "Static Tripod Shot")),
        d.get("wan_prompt", ""),
        int(s.get("video_seed", default_seed)),
        vid,
    )


def studio_redirect_single_clip(
    choice_label, motion_desc, camera_move, use_vision, api_key, model, current_state
):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return "", "", {}, "### ⚠️ Please run earlier phases first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    if motion_desc:
        scenes[idx]["wan_motion_prompt"] = motion_desc
    if camera_move:
        scenes[idx]["camera_movement"] = camera_move
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        plan = pipeline.direct_single_scene_animation(current_state, idx, use_vision=bool(use_vision))
        status = (
            f"### 🎬 Gemma Vision Re-Directed Scene #{idx+1}\n"
            f"Camera: `{plan.get('camera_movement')}` | Intensity: `{plan.get('motion_intensity')}` | Source: `{plan.get('source')}`"
        )
        return (
            plan.get("camera_movement", camera_move),
            plan.get("wan_prompt", ""),
            StudioPipeline.serializable(current_state),
            status,
            current_state,
        )
    except Exception as e:
        return camera_move, "", StudioPipeline.serializable(current_state), f"### ❌ Director Error\n`{e}`", current_state


def studio_redo_single_clip(
    choice_label, motion_desc, camera_move, compiled_wan_prompt, clip_seed,
    re_run_director, use_vision, aspect_ratio, api_key, model, current_state,
    use_last_frame=True, enable_rife_2x=False, anti_noise_guard=True,
    progress=gr.Progress(track_tqdm=True),
):
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes:
        return None, None, {}, "### ⚠️ Please generate scene keyframes first.", current_state
    idx = _parse_selector_index(choice_label, choices)
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        updated_scene = pipeline.regenerate_single_scene_video(
            story_data=current_state,
            scene_index=idx,
            updated_fields={
                "wan_motion_prompt": motion_desc,
                "camera_movement": camera_move,
                "wan_prompt": compiled_wan_prompt,
            },
            seed=int(clip_seed),
            aspect_ratio=aspect_ratio,
            re_run_director=bool(re_run_director),
            use_vision=bool(use_vision),
            use_last_frame=bool(use_last_frame),
            enable_rife_2x=bool(enable_rife_2x),
            anti_noise_guard=bool(anti_noise_guard),
            auto_stitch=True,
            progress_cb=_studio_progress(progress),
        )
        clip_path = updated_scene.get("video_path")
        final_movie = current_state.get("final_movie_path")
        flf_tag = "FLF2V (First+Last Frame)" if updated_scene.get("used_last_frame") else "I2V (First Frame)"
        status = (
            f"### 🔄 Scene #{idx+1} Video Clip Redo Complete!\n"
            f"- **Clip:** `{pathlib.Path(clip_path).name if clip_path else 'N/A'}` (Seed `{int(clip_seed)}`, Mode: `{flf_tag}`)\n"
            f"- **Final Movie Re-Stitched:** `{pathlib.Path(final_movie).name if final_movie else 'Pending remaining clips'}`"
        )
        return (
            clip_path,
            final_movie,
            StudioPipeline.serializable(current_state),
            status,
            current_state,
        )
    except Exception as e:
        return (
            None,
            current_state.get("final_movie_path"),
            StudioPipeline.serializable(current_state),
            f"### ❌ Clip Redo Error\n`{e}`",
            current_state,
        )


def studio_run_full(
    prompt, art_style, aspect_ratio, seed, num_scenes, duration, api_key, model,
    use_vision=True, qwen_seed=None, wan_seed=None, progress=gr.Progress(track_tqdm=True),
):
    pipeline = StudioPipeline(api_key=api_key, model=model)
    z_seed = int(seed)
    q_seed = int(qwen_seed if qwen_seed is not None else seed)
    w_seed = int(wan_seed if wan_seed is not None else seed)
    try:
        story_data = pipeline.run_full_pipeline(
            user_prompt=prompt,
            shared_seed=z_seed,
            z_image_seed=z_seed,
            qwen_seed=q_seed,
            wan_seed=w_seed,
            num_scenes=int(num_scenes),
            target_duration=float(duration),
            aspect_ratio=aspect_ratio,
            art_style=art_style,
            use_vision_director=bool(use_vision),
            progress_cb=_studio_progress(progress),
        )
        char_items, scene_items = _studio_gallery_items(story_data)
        final_video = story_data.get("final_movie_path")
        status_msg = (
            f"### 🎬 Production Complete!\n\n"
            f"- **Screenplay:** '{story_data.get('title')}'\n"
            f"- **Characters:** {len(char_items)} (Z-Image Turbo, locked seed `{z_seed}`)\n"
            f"- **Scene Keyframes (Start + End):** {len(scene_items)} (Qwen Multi-Reference, locked seed `{q_seed}`)\n"
            f"- **Video Clips:** {len(story_data.get('clip_paths', []))} (Wan 2.2 FLF2V, locked seed `{w_seed}`)\n"
            f"- **Final Movie:** `{pathlib.Path(final_video).name if final_video else 'N/A'}`"
            + _studio_director_summary(story_data.get("scenes"))
        )
        return final_video, char_items, scene_items, StudioPipeline.serializable(story_data), status_msg, story_data
    except Exception as e:
        return None, [], [], {}, f"### ❌ Studio Production Error\n`{e}`", {}


def studio_run_step1(prompt, art_style, num_scenes, duration, api_key, model, progress=gr.Progress(track_tqdm=True)):
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        story_data = pipeline.generate_story_and_characters(
            prompt=prompt,
            num_scenes=int(num_scenes),
            target_duration=float(duration),
            art_style=art_style,
            progress_cb=_studio_progress(progress),
        )
        status_msg = (
            f"### 📝 Step 1 Complete: Screenplay & Characters Generated\n"
            f"**Title:** *{story_data.get('title')}* — Extracted **{len(story_data.get('characters', []))}** character(s) and "
            f"**{len(story_data.get('scenes', []))}** shot(s) of ≤5s each (with First + Last Frame beats).\n"
            f"Review/edit below, then click **Next: Validate Story ➔ Phase 2 (Characters)**."
        )
        return StudioPipeline.serializable(story_data), status_msg, story_data
    except Exception as e:
        return {}, f"### ❌ Step 1 Error\n`{e}`", {}


def studio_run_step2(seed, api_key, model, current_state, progress=gr.Progress(track_tqdm=True)):
    if not current_state or "characters" not in current_state:
        return [], "### ⚠️ Please run Step 1 (Story & Characters) first.", current_state
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        chars = pipeline.generate_character_portraits(
            characters=current_state["characters"],
            shared_seed=int(seed),
            art_style=current_state.get("art_style", ""),
            progress_cb=_studio_progress(progress),
        )
        current_state["characters"] = chars
        char_items, _ = _studio_gallery_items(current_state)
        return char_items, (
            f"### 👤 Step 2 Complete: {len(chars)} Character Portraits\n"
            f"All characters generated with locked Z-Image seed `{int(seed)}`. "
            f"Select any character on the right to edit descriptions/seed and redo individually, or click **Next ➔ Phase 3**."
        ), current_state
    except Exception as e:
        return [], f"### ❌ Step 2 Error\n`{e}`", current_state


def studio_run_step3(
    aspect_ratio, seed, api_key, model, current_state,
    generate_end_frame=True, auto_qa_retry=False, chain_previous_end_frame=False,
    direct_end_frame=True,
    progress=gr.Progress(track_tqdm=True),
):
    if not current_state or "scenes" not in current_state or not any(
        c.get("portrait_path") or c.get("portrait_pil") for c in current_state.get("characters", [])
    ):
        return [], "### ⚠️ Please run Step 1 & Step 2 first.", current_state
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        scenes = pipeline.generate_scene_keyframes(
            scenes=current_state["scenes"],
            characters=current_state["characters"],
            aspect_ratio=aspect_ratio,
            shared_seed=int(seed),
            generate_end_frame=bool(generate_end_frame),
            chain_previous_end_frame=bool(chain_previous_end_frame),
            auto_qa_retry=bool(auto_qa_retry),
            art_style=current_state.get("art_style", ""),
            direct_end_frame=bool(direct_end_frame),
            progress_cb=_studio_progress(progress),
        )
        current_state["scenes"] = scenes
        _, scene_items = _studio_gallery_items(current_state)
        chain_note = " with Seamless Shot Chaining enabled" if chain_previous_end_frame else ""
        direct_note = " (Gemma Vision End Frame Director active)" if direct_end_frame else ""
        return scene_items, (
            f"### 🖼️ Step 3 Complete: {len(scenes)} Scenes ({len(scene_items)} Start & End Keyframes){chain_note}{direct_note}\n"
            f"All Start & End frames generated with locked Qwen seed `{int(seed)}`, character multi-references, and CIE-Lab color matching. "
            f"Select any scene on the right to edit its Start/End prompts or redo individually, or click **Next ➔ Phase 4**."
        ), current_state
    except Exception as e:
        return [], f"### ❌ Step 3 Error\n`{e}`", current_state


def studio_run_director(api_key, model, use_vision, current_state, progress=gr.Progress(track_tqdm=True)):
    scenes = (current_state or {}).get("scenes", [])
    if not scenes or not all(s.get("keyframe_path") or s.get("keyframe_pil") for s in scenes):
        return StudioPipeline.serializable(current_state or {}), \
            "### ⚠️ Please run Step 3 (Scene Keyframes) first — the Director needs to see each keyframe.", current_state
    pipeline = StudioPipeline(api_key=api_key, model=model)
    try:
        current_state["scenes"] = pipeline.direct_scene_animations(
            scenes=scenes,
            characters=current_state.get("characters", []),
            story_meta=current_state,
            use_vision=bool(use_vision),
            progress_cb=_studio_progress(progress),
        )
        return StudioPipeline.serializable(current_state), (
            f"### 🎬 Step 4 Complete: Director planned {len(scenes)} shots (inspecting Start + End keyframes)"
            + _studio_director_summary(current_state["scenes"])
        ), current_state
    except Exception as e:
        return StudioPipeline.serializable(current_state), f"### ❌ Director Error\n`{e}`", current_state


def studio_run_step4(
    aspect_ratio, seed, api_key, model, use_vision, current_state,
    use_last_frame=True, enable_rife_2x=False, anti_noise_guard=True,
    progress=gr.Progress(track_tqdm=True),
):
    scenes = (current_state or {}).get("scenes", [])
    if not scenes or not all(s.get("keyframe_path") or s.get("keyframe_pil") for s in scenes):
        return None, "### ⚠️ Please run Step 3 (Scene Keyframes) first.", current_state
    pipeline = StudioPipeline(api_key=api_key, model=model)
    cb = _studio_progress(progress)
    try:
        # Never animate without a shot plan: direct any scenes the user skipped
        if not all(s.get("director") for s in scenes):
            scenes = pipeline.direct_scene_animations(
                scenes=scenes, characters=current_state.get("characters", []),
                story_meta=current_state, use_vision=bool(use_vision), progress_cb=cb,
            )
            current_state["scenes"] = scenes
        clips = pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio=aspect_ratio,
            base_seed=int(seed),
            use_last_frame=bool(use_last_frame),
            enable_rife_2x=bool(enable_rife_2x),
            anti_noise_guard=bool(anti_noise_guard),
            progress_cb=cb,
        )
        current_state["clip_paths"] = clips
        final_movie = pipeline.stitch_clips(
            clip_paths=clips, movie_title=current_state.get("title", "Studio_Movie"), progress_cb=cb,
        )
        current_state["final_movie_path"] = final_movie
        current_state["manifest_path"] = pipeline.save_manifest(current_state)
        flf_mode = "First+Last Frame (FLF2V)" if use_last_frame else "First Frame Only (I2V)"
        fps_str = "32fps RIFE 2x" if enable_rife_2x else "16fps Native"
        guard_str = " | Anti-Noise & Rotation Guard: ON" if anti_noise_guard else ""
        return final_movie, (
            f"### 🎬 Step 5 Complete: Movie Stitched! (Locked Wan Seed `{int(seed)}`, Mode: `{flf_mode}`, `{fps_str}`{guard_str})\n"
            f"Final file: `{pathlib.Path(final_movie).name}`"
            + _studio_director_summary(scenes)
        ), current_state
    except Exception as e:
        return None, f"### ❌ Step 5 Error\n`{e}`", current_state


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Wan 2.2 & Qwen AI Filmmaking Studio") as demo:

    gr.Markdown("## 🎬 Wan 2.2 AI Filmmaking Studio", elem_classes="main-title")
    gr.Markdown(
        "**💰 GPU Cost-Saver Mode Active:** Models run strictly **On-Demand ($0 Idle)** with `max_containers=1` & 15s auto-scaledown   |   "
        "**3-Seed Consistency Engine** (`Z-Image Seed` · `Qwen Scene Seed` · `Wan Video Seed`)",
        elem_classes="subtitle"
    )

    with gr.Row():
        z_image_hb_btn = gr.Button("💤 Z-Image Turbo Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        qwen_hb_btn = gr.Button("💤 Qwen Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        wan_hb_btn = gr.Button("💤 Wan 2.2 Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        clear_outputs_btn = gr.Button("🗑️ Clear All Outputs", variant="stop", scale=1)

    with gr.Row(elem_classes="toolbar-row"):
        expand_all_btn = gr.Button("🔽 Expand All Tools", variant="secondary", size="sm")
        collapse_all_btn = gr.Button("🔼 Collapse All Tools", variant="secondary", size="sm")

    # ── Tool 3 (Primary): 4-Phase Interactive AI Filmmaking Studio Wizard ─────
    with gr.Accordion(
        "🎬 AI Filmmaking Studio — 4-Phase Interactive Story-to-Movie Wizard (Gemma 4: 31B ➔ Z-Image ➔ Qwen ➔ Wan 2.2)",
        open=True,
        elem_classes="tool-accordion studio-panel",
    ) as studio_accordion:
        gr.Markdown(
            "### 🎬 4-Phase Interactive Studio · Validate & Redo Every Character, Scene, and 5s Video Clip",
            elem_classes="section-header",
        )

        # Global 3-Seed Bar & Global Studio Configuration
        with gr.Group():
            with gr.Row():
                studio_seed = gr.Number(
                    value=INITIAL_STUDIO_SEEDS["z_image_seed"],
                    precision=0,
                    label="🎲 Seed #1: Z-Image Turbo (All Characters)",
                    scale=1,
                )
                studio_qwen_seed = gr.Number(
                    value=INITIAL_STUDIO_SEEDS["qwen_seed"],
                    precision=0,
                    label="🎲 Seed #2: Qwen Image Edit (All Scene Images)",
                    scale=1,
                )
                studio_wan_seed = gr.Number(
                    value=INITIAL_STUDIO_SEEDS["wan_seed"],
                    precision=0,
                    label="🎲 Seed #3: Wan 2.2 (All 5s Video Clips)",
                    scale=1,
                )
                studio_roll_seeds_btn = gr.Button("🎲 Roll 3 New Random Seeds", variant="secondary", scale=1)

            with gr.Row():
                studio_art_style = gr.Dropdown(
                    label="Art Style",
                    choices=[
                        "2D vibrant cartoon animation",
                        "Cyberpunk anime aesthetic",
                        "3D Pixar / Disney animation style",
                        "Photorealistic cinematic",
                        "Watercolor children's book illustration",
                    ],
                    value="2D vibrant cartoon animation",
                    scale=1,
                )
                studio_aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio / Format",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="9:16 Vertical (Shorts/Reels)",
                    scale=1,
                )
                studio_num_scenes = gr.Slider(
                    minimum=3, maximum=8, step=1, value=6,
                    label="Scenes Count (≤5s each)",
                    scale=1,
                )
                studio_total_duration = gr.Slider(
                    minimum=15.0, maximum=40.0, step=2.5, value=30.0,
                    label="Total Movie Duration (s)",
                    scale=1,
                )

        studio_status = gr.Markdown(
            "### 🟢 Phase 1 Ready — Enter your story idea below and click **1️⃣ Generate Story (Gemma 4: 31B)**."
        )

        # ── 4-Phase Wizard Tabs ───────────────────────────────────────────────
        with gr.Tabs(selected="phase1") as studio_wizard_tabs:

            # ══════════════════════════════════════════════════════════════════
            # PHASE 1: STORY & SCREENPLAY
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("1️⃣ Phase 1: Story & Screenplay", id="phase1") as phase1_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        studio_prompt = gr.Textbox(
                            label="Story Premise / Idea",
                            value="Two best friends, Barnaby the fluffy brown bear and Pippin the white rabbit, discover a basket of sweet wild berries in an enchanted forest meadow and learn that sharing brings true happiness.",
                            placeholder="Describe your story premise or thematic concept...",
                            lines=4,
                        )
                        studio_use_vision = gr.Checkbox(
                            label="🎬 Enable Gemma 4: 31B Multimodal Vision Director (inspects each keyframe before animating)",
                            value=True,
                        )
                        with gr.Accordion("⚙️ Ollama Cloud API & One-Click Auto Mode", open=False):
                            with gr.Row():
                                studio_ollama_key = gr.Textbox(
                                    label="Ollama Cloud API Key",
                                    value=DEFAULT_OLLAMA_API_KEY,
                                    type="password",
                                    scale=2,
                                )
                                studio_ollama_model = gr.Textbox(
                                    label="Ollama Model",
                                    value=DEFAULT_OLLAMA_MODEL,
                                    scale=1,
                                )
                            studio_run_full_btn = gr.Button(
                                "🚀 Run Full Automated Pipeline (All 4 Phases Non-Stop)",
                                variant="secondary",
                            )

                        studio_step1_btn = gr.Button(
                            "1️⃣ Generate Story & Extract Characters (Gemma 4: 31B)",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        studio_story_title_edit = gr.Textbox(
                            label="Movie Title (Editable)",
                            placeholder="Generated story title will appear here...",
                        )
                        studio_json_display = gr.JSON(label="Generated Screenplay, Cast & Scene Breakdown")
                        studio_phase1_next_btn = gr.Button(
                            "✅ Validate Story & Next ➔ Phase 2: Characters",
                            variant="primary",
                            size="lg",
                        )

            # ══════════════════════════════════════════════════════════════════
            # PHASE 2: CHARACTER STUDIO (Z-IMAGE TURBO)
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("2️⃣ Phase 2: Character Studio (Z-Image Turbo)", id="phase2") as phase2_tab:
                with gr.Row():
                    studio_step2_btn = gr.Button(
                        "🎨 Generate All Characters (Using Locked Z-Image Seed)",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    studio_phase2_back_btn = gr.Button("⬅️ Back to Phase 1", variant="secondary", scale=1)
                    studio_phase2_next_btn = gr.Button(
                        "✅ Validate Characters & Next ➔ Phase 3: Scene Images",
                        variant="primary",
                        scale=2,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        studio_char_gallery = gr.Gallery(
                            label="👤 Generated Character Reference Sheets (Click a character to inspect/redo)",
                            columns=[2],
                            height=420,
                            object_fit="contain",
                            preview=True,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🛠️ Individual Character Inspector & Redo")
                        studio_char_selector = gr.Dropdown(
                            label="Select Character to Edit / Redo",
                            choices=[],
                            interactive=True,
                        )
                        studio_char_preview = gr.Image(
                            label="Selected Character Portrait",
                            height=200,
                            interactive=False,
                        )
                        with gr.Row():
                            studio_char_name_edit = gr.Textbox(label="Character Name", scale=1)
                            studio_char_seed_edit = gr.Number(
                                label="Character Seed (Change to vary portrait)",
                                value=INITIAL_STUDIO_SEEDS["z_image_seed"],
                                precision=0,
                                scale=1,
                            )
                        studio_char_face_edit = gr.Textbox(label="Face Details (eyes, expression, muzzle/features)")
                        studio_char_hair_edit = gr.Textbox(label="Hair / Fur / Feathers Details")
                        studio_char_cloth_edit = gr.Textbox(label="Clothing & Outfit Details")
                        studio_char_sig_edit = gr.Textbox(label="Signature Props / Accessories")
                        studio_char_prompt_edit = gr.Textbox(
                            label="Z-Image Prompt Override",
                            lines=2,
                        )
                        with gr.Row():
                            studio_char_save_btn = gr.Button("💾 Save Description Changes Only", variant="secondary")
                            studio_char_redo_btn = gr.Button(
                                "🔄 Redo This Character Image (Z-Image Turbo)",
                                variant="primary",
                            )

            # ══════════════════════════════════════════════════════════════════
            # PHASE 3: SCENE / SCENARIO IMAGES (QWEN MULTI-REF — FIRST + LAST FRAME)
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("3️⃣ Phase 3: Scene / Scenario Images (Qwen Dual-Keyframe: First + Last Frame)", id="phase3") as phase3_tab:
                with gr.Row():
                    studio_step3_btn = gr.Button(
                        "🖼️ Generate All Scene First + Last Frames (Locked Qwen Seed + Character Refs + CIE-Lab Match)",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    studio_phase3_back_btn = gr.Button("⬅️ Back to Phase 2", variant="secondary", scale=1)
                    studio_phase3_next_btn = gr.Button(
                        "✅ Validate Scene Images & Next ➔ Phase 4: Video Clips",
                        variant="primary",
                        scale=2,
                    )

                with gr.Row():
                    studio_scene_direct_end_chk = gr.Checkbox(
                        label="🎬 Auto-Direct Last Frame with Gemma Vision (Ensures High-Delta Dynamic Pose at 5.0s)",
                        value=True,
                        info="Gemma 4: 31B Vision inspects the Start Frame (0.0s) and directs an explicit limb, weapon, and stance progression at 5.0s.",
                    )
                    studio_scene_qa_chk = gr.Checkbox(
                        label="🔍 Auto-Verify & Self-Correct Scenes with Gemma 4: 31B Vision QA Critic (Scores 1-10, retries once if < 7)",
                        value=False,
                    )
                    studio_scene_chain_chk = gr.Checkbox(
                        label="🔗 Seamless Continuous Shot Chaining (Scene N End Frame ➔ Scene N+1 Start Frame — Zero Degradation)",
                        value=False,
                        info="Reuses previous scene's clean Qwen end frame as the next scene's start frame to guarantee 100% visual continuity and cut generation time.",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        studio_scenes_gallery = gr.Gallery(
                            label="🖼️ Generated Scene Keyframes — First (0.0s) & Last (5.0s) Frames (Click any frame to inspect/redo)",
                            columns=[2, 2],
                            height=500,
                            object_fit="contain",
                            preview=True,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🛠️ Individual Scene / Scenario Dual-Keyframe Inspector & Redo")
                        studio_scene_selector = gr.Dropdown(
                            label="Select Scene to Edit / Redo",
                            choices=[],
                            interactive=True,
                        )
                        with gr.Row():
                            studio_scene_preview = gr.Image(
                                label="🟢 First Frame (Start @ 0.0s)",
                                height=200,
                                interactive=False,
                                scale=1,
                            )
                            studio_scene_end_preview = gr.Image(
                                label="🏁 Last Frame (End @ 5.0s — CIE-Lab Matched)",
                                height=200,
                                interactive=False,
                                scale=1,
                            )
                        with gr.Row():
                            studio_scene_title_edit = gr.Textbox(label="Scene Title", scale=2)
                            studio_scene_seed_edit = gr.Number(
                                label="Start Frame (0.0s) Seed",
                                value=INITIAL_STUDIO_SEEDS["qwen_seed"],
                                precision=0,
                                scale=1,
                            )
                            studio_scene_end_seed_edit = gr.Number(
                                label="End Frame (5.0s) Seed",
                                value=INITIAL_STUDIO_SEEDS["qwen_seed"] + 10007,
                                precision=0,
                                scale=1,
                            )
                        with gr.Row():
                            studio_scene_delta_badge = gr.Textbox(
                                label="🏃 Motion Delta Metric (Start vs End)",
                                value="Pending generation",
                                interactive=False,
                                scale=1,
                            )
                        studio_scene_scenario_edit = gr.Textbox(
                            label="Scene Scenario / Story Beat Description",
                            lines=2,
                        )
                        studio_scene_prompt_edit = gr.Textbox(
                            label="🟢 First Frame Qwen Visual Prompt (Start Pose @ 0.0s)",
                            lines=2,
                        )
                        studio_scene_end_prompt_edit = gr.Textbox(
                            label="🏁 Last Frame Qwen Visual Prompt (End Pose @ 5.0s — Anchored on First Frame)",
                            lines=2,
                        )
                        with gr.Row():
                            studio_scene_chars_edit = gr.Textbox(
                                label="Character IDs in Scene (comma-separated, e.g. char_1, char_2)",
                                scale=2,
                            )
                            studio_scene_redo_mode = gr.Dropdown(
                                label="Frames to Redo",
                                choices=["both", "start", "end"],
                                value="both",
                                scale=1,
                            )
                        with gr.Row():
                            studio_scene_save_btn = gr.Button("💾 Save Scene Description Only", variant="secondary")
                            studio_scene_direct_end_btn = gr.Button("🎬 Direct End Frame with Gemma Vision", variant="secondary")
                            studio_scene_qa_btn = gr.Button("🔍 Inspect Scene with Gemma Vision QA Critic", variant="secondary")
                            studio_scene_redo_btn = gr.Button(
                                "🔄 Redo Selected Scene Frame(s) (Qwen Multi-Ref)",
                                variant="primary",
                            )

            # ══════════════════════════════════════════════════════════════════
            # PHASE 4: VIDEO CLIPS & FINAL MOVIE (GEMMA DIRECTOR + WAN 2.2 FLF2V)
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("4️⃣ Phase 4: 5s Video Clips & Final Movie (Wan 2.2 FLF2V + 32fps RIFE)", id="phase4") as phase4_tab:
                with gr.Row():
                    studio_step_director_btn = gr.Button(
                        "🎬 Step 4A: Run Dual-Frame Gemma Vision Director on All Scenes",
                        variant="secondary",
                        scale=1,
                    )
                    studio_step4_btn = gr.Button(
                        "🎥 Step 4B: Generate All 5s Video Clips & Stitch Final Movie (Locked Wan Seed)",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    studio_phase4_back_btn = gr.Button("⬅️ Back to Phase 3", variant="secondary", scale=1)

                with gr.Row():
                    studio_use_last_frame_chk = gr.Checkbox(
                        label="📌 Use First + Last Frame in Wan 2.2 (FLF2V Mode — Locks Start & End Poses)",
                        value=True,
                    )
                    studio_rife_2x_chk = gr.Checkbox(
                        label="✨ Enable 32fps RIFE 2x Smooth Motion Interpolation (16fps ➔ 32fps)",
                        value=True,
                    )
                    studio_anti_noise_chk = gr.Checkbox(
                        label="🛡️ Anti-Noise & Rotation Identity Guard (5s Noise Clamp & Turn Lock)",
                        value=True,
                        info="Clamps CFG <= 2.0 and sets FlowShift 4.5 on long clips to stop noise dissolution; locks facial identity during rotations.",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        studio_final_video = gr.Video(
                            label="🎬 Final Stitched Movie (All Clips Combined)",
                            autoplay=True,
                            interactive=False,
                            height=360,
                        )
                        studio_clip_preview = gr.Video(
                            label="🎞️ Selected Individual 5-Second Scene Clip Preview",
                            autoplay=False,
                            interactive=False,
                            height=260,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 🛠️ Individual 5s Video Clip Inspector & Redo")
                        studio_clip_selector = gr.Dropdown(
                            label="Select 5s Scene Clip to Inspect / Edit / Redo",
                            choices=[],
                            interactive=True,
                        )
                        with gr.Row():
                            studio_clip_camera_edit = gr.Textbox(
                                label="Camera Movement",
                                value="Slow Cinematic Push-In",
                                scale=1,
                            )
                            studio_clip_seed_edit = gr.Number(
                                label="Clip Seed (Change to vary motion trajectory)",
                                value=INITIAL_STUDIO_SEEDS["wan_seed"],
                                precision=0,
                                scale=1,
                            )
                        studio_clip_motion_edit = gr.Textbox(
                            label="Video Motion Description (What happens in this 5-second clip)",
                            lines=3,
                            placeholder="Describe the character movement, action, and expression for this 5s clip...",
                        )
                        studio_clip_wan_prompt_edit = gr.Textbox(
                            label="Full Compiled Director / Wan 2.2 Prompt (Editable)",
                            lines=4,
                        )
                        studio_clip_rerundir_chk = gr.Checkbox(
                            label="🪄 Ask Gemma 4: 31B Vision Director to re-plan this clip before animating",
                            value=False,
                        )
                        with gr.Row():
                            studio_clip_director_btn = gr.Button(
                                "🪄 Re-Direct Only This Clip (Gemma Vision)",
                                variant="secondary",
                            )
                            studio_clip_redo_btn = gr.Button(
                                "🔄 Redo This 5s Video Clip & Re-Stitch Movie",
                                variant="primary",
                            )

    # ── Tool 1: Qwen Image Edit Rapid AIO Generator / Editor (Standalone) ─────
    with gr.Accordion("🖼️ Standalone Tool 1 — Qwen Image Edit Rapid AIO (Image Generator & Editor)", open=False, elem_classes="tool-accordion qwen-panel") as qwen_accordion:
        gr.Markdown("### 🖼️ Qwen Image Edit Rapid AIO · Image Generator & Editor", elem_classes="section-header")
        gr.Markdown(
            "Upload an **image to edit** (or leave empty for text-to-image). "
            "Optionally add **reference images** for character / style transfer. "
            "Outputs save automatically to `./outputs/`.",
            elem_classes="step-hint"
        )

        with gr.Row():
            # ── Left column: inputs ───────────────────────────────────────────
            with gr.Column(scale=1):
                qwen_base_image = gr.Image(
                    type="pil", label="📷 Base Image (optional — for image editing)",
                    sources=["upload", "clipboard"], elem_id="qwen-base-image"
                )
                with gr.Row():
                    qwen_ref_image_1 = gr.Image(
                        type="pil", label="🔗 Reference 1 (optional)",
                        sources=["upload", "clipboard"], scale=1, elem_id="qwen-ref1-image"
                    )
                    qwen_ref_image_2 = gr.Image(
                        type="pil", label="🔗 Reference 2 (optional)",
                        sources=["upload", "clipboard"], scale=1, elem_id="qwen-ref2-image"
                    )

                qwen_prompt = gr.Textbox(
                    label="Prompt / Instruction",
                    value=default_prompt_qwen,
                    placeholder="Describe how to edit the image or what to generate...",
                    lines=3,
                )

                qwen_negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value=default_negative_prompt,
                    placeholder="Describe what to avoid in the edited image...",
                    lines=3,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        qwen_seed = gr.Slider(
                            minimum=-1, maximum=MAX_SEED, step=1, value=-1,
                            label="Seed (-1 = random)"
                        )
                        qwen_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    qwen_num_images = gr.Slider(minimum=1, maximum=6, step=1, value=1, label="Number of Images (1–6)")
                    qwen_steps = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Inference Steps")
                    qwen_true_cfg = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=4.0, label="True CFG Scale")
                    qwen_guidance = gr.Slider(minimum=0.0, maximum=5.0, step=0.1, value=1.0, label="Guidance Scale")
                    with gr.Row():
                        qwen_custom_width = gr.Slider(minimum=512, maximum=2048, step=64, value=1024, label="Width (px)")
                        qwen_custom_height = gr.Slider(minimum=512, maximum=2048, step=64, value=1024, label="Height (px)")

                qwen_generate_btn = gr.Button("✨ Generate with Qwen Image Edit", variant="primary")

            # ── Right column: output ─────────────────────────────────────────
            with gr.Column(scale=1):
                qwen_output_gallery = gr.Gallery(
                    label="🖼️ Generated / Edited Result(s)",
                    columns=[2, 3],
                    rows=[1, 2],
                    height=440,
                    object_fit="contain",
                    preview=True,
                    interactive=False,
                )
                qwen_output_seed = gr.Textbox(label="Used Seed(s)", interactive=False)
                qwen_selected_image = gr.State(value=None)
                
                gr.Markdown(
                    "💡 **Drag & Drop:** You can drag any generated image directly into any input box across all tools, "
                    "or click an image and use the transfer buttons below:",
                    elem_classes="step-hint"
                )
                with gr.Row():
                    use_as_base_btn = gr.Button("⬅️ Use as Base Image", variant="primary", scale=1)
                    use_as_ref1_btn = gr.Button("🔗 Set as Ref 1", variant="secondary", scale=1)
                    use_as_ref2_btn = gr.Button("🔗 Set as Ref 2", variant="secondary", scale=1)

                with gr.Row():
                    use_qwen_as_video_btn = gr.Button(
                        "🎥 Send to Wan 2.2 (First Frame)",
                        variant="secondary",
                        scale=1
                    )
                    use_qwen_as_video_last_btn = gr.Button(
                        "📌 Send to Wan 2.2 (Last Frame)",
                        variant="secondary",
                        scale=1
                    )

    # ── Tool 2: Wan 2.2 Video Generator (Standalone) ──────────────────────────
    with gr.Accordion("🎥 Standalone Tool 2 — Wan 2.2 · High-Motion Video Generator", open=False, elem_classes="tool-accordion wan-panel") as wan_accordion:
        gr.Markdown("### 🎥 Wan 2.2 Video Generator", elem_classes="section-header")
        gr.Markdown(
            "Upload or route an image from Tool 1 (Qwen Image Edit). Add a motion prompt and click **Generate Video**.\n\n"
            "💡 **Facial Consistency & Stability Tip:** To prevent the subject's face from altering after 3–4 seconds, "
            "keep duration within **3.0s – 5.0s**, leave Guidance Scale at **1.5 – 2.5** so the negative prompt is actively enforced, "
            "and use the **AI Video Director** below or provide a **Last Frame** to anchor facial identity.",
            elem_classes="step-hint"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image_component = gr.Image(
                    type="pil", label="Input Image for Video (First Frame)",
                    sources=["upload", "clipboard"], elem_id="wan-input-image"
                )

                # ── AI Video Director & Prompt Reasoning Assistant ──
                with gr.Accordion("🪄 AI Video Director & Prompt Reasoning (Zero-Drift Assistant)", open=False, elem_classes="director-panel"):
                    gr.Markdown(
                        "**The AI Director decomposes your prompt into a structured temporal script.** "
                        "Diffusion Transformers (DiT) lock facial identity and prevent pixel degradation when guided by invariant anchors and multi-phase timelines.",
                        elem_classes="step-hint"
                    )
                    with gr.Row():
                        director_style = gr.Dropdown(
                            label="Direction Focus / Style",
                            choices=[
                                "👤 Portrait & Identity Lock (Prevents face drift & morphing)",
                                "✨ Expressive Head Turn & Smile (Gradual head turn, locked identity)",
                                "🎬 Cinematic Camera Orbit (Locked subject, camera trajectory)",
                                "🌬️ Micro-Motion Natural (Subtle breathing, wind, hair sway)",
                                "🚀 Dynamic Action & Flow (Steady kinematics, no pixel tears)"
                            ],
                            value="👤 Portrait & Identity Lock (Prevents face drift & morphing)",
                            scale=2
                        )
                        director_camera = gr.Dropdown(
                            label="Camera Movement",
                            choices=[
                                "Static Tripod Shot (Highest facial stability)",
                                "Slow Cinematic Push-In / Dolly",
                                "Gentle Orbital Tracking",
                                "Smooth Horizontal Pan",
                                "Subtle Handheld Drift"
                            ],
                            value="Static Tripod Shot (Highest facial stability)",
                            scale=1
                        )
                    with gr.Row():
                        director_intensity = gr.Slider(
                            minimum=1, maximum=5, step=1, value=2,
                            label="Motion Intensity (1=Subtle, 5=Dynamic)",
                            scale=1
                        )
                        director_apply_btn = gr.Button("✨ Synthesize Temporally-Anchored Prompt", variant="primary", scale=2)

                prompt_input = gr.Textbox(label="Motion Prompt", value=default_prompt_i2v, lines=3)
                duration_seconds_input = gr.Slider(
                    minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=DEFAULT_DURATION,
                    label="Duration (seconds)",
                    info="Native Wan 2.2 window: 1.0s to 5.0s (17 to 81 frames at 16fps) to prevent pixel degradation and facial drift."
                )
                frame_multi = gr.Dropdown(
                    choices=[FIXED_FPS, FIXED_FPS * 2, FIXED_FPS * 4, FIXED_FPS * 8],
                    value=FIXED_FPS,
                    label="Video Fluidity (FPS)",
                    info="Extra frames generated using RIFE flow estimation."
                )
                safe_mode_checkbox = gr.Checkbox(
                    label="Safe Mode", value=True,
                    info="Requests extra processing-time safety margins."
                )

                with gr.Accordion("Advanced Settings", open=False):
                    last_image_component = gr.Image(
                        type="pil", label="Last Frame (Optional — Anchors End Pose)",
                        sources=["upload", "clipboard"], elem_id="wan-last-image"
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt", value=default_negative_prompt, lines=3
                    )
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="Video Quality")
                    seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                    steps_slider = gr.Slider(
                        minimum=4, maximum=24, step=1, value=8, label="Inference Steps",
                        info="8–10 steps locks facial identity and prevents drift on Lightning models."
                    )
                    guidance_scale_input = gr.Slider(
                        minimum=1.0, maximum=7.0, step=0.25, value=2.0, label="Guidance Scale (High Noise)",
                        info="Controls prompt adherence. Values 1.5–2.5 activate negative prompt face-preservation."
                    )
                    guidance_scale_2_input = gr.Slider(
                        minimum=1.0, maximum=7.0, step=0.25, value=1.5, label="Guidance Scale 2 (Low Noise)",
                        info="Stabilizes fine facial features and surface textures in later denoising steps."
                    )
                    with gr.Row():
                        lora_repo_or_path_input = gr.Textbox(
                            label="LoRA Model (Hugging Face ID or path)",
                            value="",
                            placeholder="e.g. user/wan2.2-face-consistency (optional)",
                            scale=2
                        )
                        lora_scale_slider = gr.Slider(
                            minimum=0.0, maximum=1.5, step=0.05, value=0.8,
                            label="LoRA Scale / Weight",
                            scale=1
                        )
                    deflicker_checkbox = gr.Checkbox(
                        label="✨ Temporal Face & Texture Deflicker",
                        value=True,
                        info="Applies edge-preserving temporal filtering to eliminate pixel micro-jitter and lock facial features across frames before RIFE."
                    )
                    anti_noise_checkbox = gr.Checkbox(
                        label="🛡️ Anti-Noise & Rotation Identity Guard (5s Stabilization)",
                        value=True,
                        info="Stabilizes long clips (>= 4.5s) with FlowShift 4.5 & steps >= 10, clamps CFG <= 2.0 to prevent latent blowups, and locks facial proportions during head/body turns."
                    )
                    scheduler_dropdown = gr.Dropdown(label="Scheduler", choices=SCHEDULERS, value="UniPCMultistep")
                    flow_shift_slider = gr.Slider(minimum=0.5, maximum=15.0, step=0.1, value=3.0, label="Flow Shift")
                    play_result_video = gr.Checkbox(label="Display result", value=True, interactive=True)

                generate_button = gr.Button("🎬 Generate Video (Wan 2.2)", variant="primary")

            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Generated Video", autoplay=True,
                    sources=["upload"], interactive=True, elem_id="generated-video"
                )
                with gr.Row():
                    grab_frame_btn = gr.Button("📸 Use Current Frame as Input", variant="secondary")
                    grab_last_frame_btn = gr.Button("📌 Use Current Frame as Last Frame", variant="secondary")
                    timestamp_box = gr.Number(value=0, label="Timestamp", visible=False, elem_id="hidden-timestamp")
                    timestamp_last_box = gr.Number(value=0, label="Timestamp Last", visible=False, elem_id="hidden-timestamp-last")
                file_output = gr.File(label="Download Video")

    # ── Global Toolbar Events (Expand/Collapse All Tools) ─────────────────────
    expand_all_btn.click(
        fn=lambda: (gr.update(open=True), gr.update(open=True), gr.update(open=True)),
        inputs=[],
        outputs=[qwen_accordion, wan_accordion, studio_accordion],
    )

    collapse_all_btn.click(
        fn=lambda: (gr.update(open=False), gr.update(open=False), gr.update(open=False)),
        inputs=[],
        outputs=[qwen_accordion, wan_accordion, studio_accordion],
    )

    # ── Events: Qwen Image Edit ───────────────────────────────────────────────

    def on_qwen_gallery_select(evt: gr.SelectData, gallery):
        if evt.value:
            val = evt.value
            if isinstance(val, dict) and "image" in val:
                val = val["image"]
            elif isinstance(val, dict) and "path" in val:
                val = val["path"]
            if isinstance(val, tuple):
                val = val[0]
            if isinstance(val, str):
                try:
                    return Image.open(val).convert("RGB")
                except Exception:
                    pass
            return val
        if gallery and len(gallery) > evt.index:
            return extract_selected_image(gallery[evt.index], None)
        return None

    qwen_output_gallery.select(
        fn=on_qwen_gallery_select,
        inputs=[qwen_output_gallery],
        outputs=[qwen_selected_image],
    )

    qwen_generate_btn.click(
        fn=run_qwen_generate,
        inputs=[
            qwen_base_image, qwen_ref_image_1, qwen_ref_image_2,
            qwen_prompt, qwen_negative_prompt,
            qwen_seed, qwen_randomize_seed,
            qwen_steps, qwen_true_cfg, qwen_guidance,
            qwen_num_images,
            qwen_custom_width, qwen_custom_height,
        ],
        outputs=[qwen_output_gallery, qwen_output_seed, qwen_selected_image],
    )

    use_as_base_btn.click(
        fn=route_qwen_to_base,
        inputs=[qwen_selected_image, qwen_output_gallery],
        outputs=[qwen_base_image],
    )

    use_as_ref1_btn.click(
        fn=route_qwen_to_ref1,
        inputs=[qwen_selected_image, qwen_output_gallery],
        outputs=[qwen_ref_image_1],
    )

    use_as_ref2_btn.click(
        fn=route_qwen_to_ref2,
        inputs=[qwen_selected_image, qwen_output_gallery],
        outputs=[qwen_ref_image_2],
    )

    use_qwen_as_video_btn.click(
        fn=route_qwen_to_video,
        inputs=[qwen_selected_image, qwen_output_gallery],
        outputs=[input_image_component],
    )

    use_qwen_as_video_last_btn.click(
        fn=route_qwen_to_video_last,
        inputs=[qwen_selected_image, qwen_output_gallery],
        outputs=[last_image_component],
    )

    director_apply_btn.click(
        fn=generate_director_prompt,
        inputs=[prompt_input, director_style, director_camera, director_intensity, duration_seconds_input],
        outputs=[prompt_input, negative_prompt_input, guidance_scale_input, steps_slider],
    )

    # ── Events: Wan 2.2 Video ─────────────────────────────────────────────────

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider, frame_multi,
        safe_mode_checkbox, play_result_video,
        lora_repo_or_path_input, lora_scale_slider, deflicker_checkbox,
        anti_noise_checkbox,
    ]

    generate_button.click(
        fn=generate_video,
        inputs=ui_inputs,
        outputs=[video_output, file_output, seed_input],
    )

    grab_frame_btn.click(
        fn=None, inputs=None, outputs=[timestamp_box], js=get_timestamp_js
    )

    timestamp_box.change(
        fn=extract_frame,
        inputs=[video_output, timestamp_box],
        outputs=[input_image_component],
    )

    grab_last_frame_btn.click(
        fn=None, inputs=None, outputs=[timestamp_last_box], js=get_timestamp_js
    )

    timestamp_last_box.change(
        fn=extract_frame,
        inputs=[video_output, timestamp_last_box],
        outputs=[last_image_component],
    )

    # ── Events: 4-Phase Studio Wizard ─────────────────────────────────────────

    studio_pipeline_state = gr.State(value={})

    def _roll_and_sync_seeds():
        z_s, q_s, w_s = roll_three_studio_seeds()
        return z_s, q_s, w_s, z_s, q_s, w_s

    studio_roll_seeds_btn.click(
        fn=_roll_and_sync_seeds,
        inputs=[],
        outputs=[
            studio_seed, studio_qwen_seed, studio_wan_seed,
            studio_char_seed_edit, studio_scene_seed_edit, studio_clip_seed_edit,
        ],
    )

    # Phase 1: Generate Story & populate dropdowns for Phases 2, 3, 4
    def _ui_step1(prompt, art_style, num_scenes, duration, api_key, model, z_seed, q_seed, w_seed, progress=gr.Progress(track_tqdm=True)):
        json_data, status_msg, state = studio_run_step1(
            prompt, art_style, num_scenes, duration, api_key, model, progress=progress
        )
        char_opts = _character_choices(state)
        scene_opts = _scene_choices(state)
        first_char = char_opts[0] if char_opts else None
        first_scene = scene_opts[0] if scene_opts else None
        c_name, c_face, c_hair, c_cloth, c_sig, c_prompt, c_seed, c_img = studio_load_character_fields(
            first_char, state, default_seed=z_seed
        )
        (
            s_title, s_scen, s_qprompt, s_cids, s_seed, s_img,
            s_qendprompt, s_end_img, s_end_seed, s_delta,
        ) = studio_load_scene_fields_dual(first_scene, state, default_seed=q_seed)
        v_mot, v_cam, v_wprompt, v_seed, v_vid = studio_load_clip_fields(
            first_scene, state, default_seed=w_seed
        )
        return (
            json_data,
            status_msg,
            state,
            state.get("title", ""),
            gr.update(choices=char_opts, value=first_char),
            c_name, c_face, c_hair, c_cloth, c_sig, c_prompt, c_seed, c_img,
            gr.update(choices=scene_opts, value=first_scene),
            s_title, s_scen, s_qprompt, s_qendprompt, s_cids, s_seed, s_img, s_end_img, s_end_seed, s_delta,
            gr.update(choices=scene_opts, value=first_scene),
            v_mot, v_cam, v_wprompt, v_seed, v_vid,
        )

    studio_step1_btn.click(
        fn=_ui_step1,
        inputs=[
            studio_prompt, studio_art_style, studio_num_scenes,
            studio_total_duration, studio_ollama_key, studio_ollama_model,
            studio_seed, studio_qwen_seed, studio_wan_seed,
        ],
        outputs=[
            studio_json_display, studio_status, studio_pipeline_state,
            studio_story_title_edit,
            studio_char_selector,
            studio_char_name_edit, studio_char_face_edit, studio_char_hair_edit,
            studio_char_cloth_edit, studio_char_sig_edit, studio_char_prompt_edit,
            studio_char_seed_edit, studio_char_preview,
            studio_scene_selector,
            studio_scene_title_edit, studio_scene_scenario_edit, studio_scene_prompt_edit,
            studio_scene_end_prompt_edit, studio_scene_chars_edit, studio_scene_seed_edit,
            studio_scene_preview, studio_scene_end_preview,
            studio_scene_end_seed_edit, studio_scene_delta_badge,
            studio_clip_selector,
            studio_clip_motion_edit, studio_clip_camera_edit, studio_clip_wan_prompt_edit,
            studio_clip_seed_edit, studio_clip_preview,
        ],
    )

    def _hb_button_updates():
        import backend
        z_btn = (
            gr.update(value="💓 Z-Image Turbo Keep-Warm: Active (Phase 2)", variant="primary")
            if backend.z_image_heartbeat_enabled
            else gr.update(value="💤 Z-Image Turbo Keep-Warm: Paused ($0 Idle)", variant="secondary")
        )
        q_btn = (
            gr.update(value="💓 Qwen Keep-Warm: Active (Phase 3)", variant="primary")
            if backend.qwen_heartbeat_enabled
            else gr.update(value="💤 Qwen Keep-Warm: Paused ($0 Idle)", variant="secondary")
        )
        w_btn = (
            gr.update(value="💓 Wan 2.2 Keep-Warm: Active (Phase 4)", variant="primary")
            if backend.wan_heartbeat_enabled
            else gr.update(value="💤 Wan 2.2 Keep-Warm: Paused ($0 Idle)", variant="secondary")
        )
        return z_btn, q_btn, w_btn

    def _switch_to_phase_gpu(phase_num: int):
        import backend
        backend.activate_studio_phase(phase_num, trigger_warmup=True)
        return _hb_button_updates()

    # Phase 1 -> Phase 2 validation & navigation
    def _validate_phase1(title_edit, current_state):
        import backend
        backend.activate_studio_phase(2, trigger_warmup=True)
        if current_state and title_edit and str(title_edit).strip():
            current_state["title"] = str(title_edit).strip()
        return (
            gr.update(selected="phase2"),
            "### ✅ Phase 1 Validated — Now in **Phase 2: Character Studio** (🔥 Z-Image Turbo GPU warming up; Qwen & Wan paused at $0).",
            current_state,
            *_hb_button_updates(),
        )

    studio_phase1_next_btn.click(
        fn=_validate_phase1,
        inputs=[studio_story_title_edit, studio_pipeline_state],
        outputs=[studio_wizard_tabs, studio_status, studio_pipeline_state, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    studio_phase2_back_btn.click(
        fn=lambda: (gr.update(selected="phase1"), *_switch_to_phase_gpu(1)),
        inputs=[],
        outputs=[studio_wizard_tabs, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    def _validate_phase2():
        import backend
        backend.activate_studio_phase(3, trigger_warmup=True)
        return (
            gr.update(selected="phase3"),
            "### ✅ Phase 2 Validated — Now in **Phase 3: Scene Images** (🔥 Qwen GPU warming up; Z-Image & Wan paused at $0).",
            *_hb_button_updates(),
        )

    studio_phase2_next_btn.click(
        fn=_validate_phase2,
        inputs=[],
        outputs=[studio_wizard_tabs, studio_status, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    studio_phase3_back_btn.click(
        fn=lambda: (gr.update(selected="phase2"), *_switch_to_phase_gpu(2)),
        inputs=[],
        outputs=[studio_wizard_tabs, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    def _validate_phase3():
        import backend
        backend.activate_studio_phase(4, trigger_warmup=True)
        return (
            gr.update(selected="phase4"),
            "### ✅ Phase 3 Validated — Now in **Phase 4: Video Clips & Final Movie** (🔥 Wan 2.2 GPU warming up; Z-Image & Qwen paused at $0).",
            *_hb_button_updates(),
        )

    studio_phase3_next_btn.click(
        fn=_validate_phase3,
        inputs=[],
        outputs=[studio_wizard_tabs, studio_status, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    studio_phase4_back_btn.click(
        fn=lambda: (gr.update(selected="phase3"), *_switch_to_phase_gpu(3)),
        inputs=[],
        outputs=[studio_wizard_tabs, z_image_hb_btn, qwen_hb_btn, wan_hb_btn],
    )

    # Also switch active GPU automatically if user clicks any Phase Tab directly
    phase1_tab.select(fn=lambda: _switch_to_phase_gpu(1), inputs=[], outputs=[z_image_hb_btn, qwen_hb_btn, wan_hb_btn])
    phase2_tab.select(fn=lambda: _switch_to_phase_gpu(2), inputs=[], outputs=[z_image_hb_btn, qwen_hb_btn, wan_hb_btn])
    phase3_tab.select(fn=lambda: _switch_to_phase_gpu(3), inputs=[], outputs=[z_image_hb_btn, qwen_hb_btn, wan_hb_btn])
    phase4_tab.select(fn=lambda: _switch_to_phase_gpu(4), inputs=[], outputs=[z_image_hb_btn, qwen_hb_btn, wan_hb_btn])

    # Phase 2: Generate All Characters & Individual Character Inspector
    def _ui_step2(seed, api_key, model, current_state, selected_char, progress=gr.Progress(track_tqdm=True)):
        char_items, status, state = studio_run_step2(seed, api_key, model, current_state, progress=progress)
        c_name, c_face, c_hair, c_cloth, c_sig, c_prompt, c_seed, c_img = studio_load_character_fields(
            selected_char, state, default_seed=seed
        )
        return char_items, status, state, c_seed, c_img

    studio_step2_btn.click(
        fn=_ui_step2,
        inputs=[studio_seed, studio_ollama_key, studio_ollama_model, studio_pipeline_state, studio_char_selector],
        outputs=[studio_char_gallery, studio_status, studio_pipeline_state, studio_char_seed_edit, studio_char_preview],
    )

    studio_char_selector.change(
        fn=studio_load_character_fields,
        inputs=[studio_char_selector, studio_pipeline_state, studio_seed],
        outputs=[
            studio_char_name_edit, studio_char_face_edit, studio_char_hair_edit,
            studio_char_cloth_edit, studio_char_sig_edit, studio_char_prompt_edit,
            studio_char_seed_edit, studio_char_preview,
        ],
    )

    def _on_char_gallery_select(evt: gr.SelectData, current_state, default_seed):
        choices = _character_choices(current_state)
        if not choices or evt.index >= len(choices):
            return gr.update(), "", "", "", "", "", "", int(default_seed), None
        chosen = choices[evt.index]
        fields = studio_load_character_fields(chosen, current_state, default_seed)
        return (gr.update(value=chosen), *fields)

    studio_char_gallery.select(
        fn=_on_char_gallery_select,
        inputs=[studio_pipeline_state, studio_seed],
        outputs=[
            studio_char_selector,
            studio_char_name_edit, studio_char_face_edit, studio_char_hair_edit,
            studio_char_cloth_edit, studio_char_sig_edit, studio_char_prompt_edit,
            studio_char_seed_edit, studio_char_preview,
        ],
    )

    studio_char_save_btn.click(
        fn=studio_save_character_text_edits,
        inputs=[
            studio_char_selector, studio_char_name_edit, studio_char_face_edit,
            studio_char_hair_edit, studio_char_cloth_edit, studio_char_sig_edit,
            studio_char_prompt_edit, studio_pipeline_state,
        ],
        outputs=[studio_json_display, studio_status, studio_pipeline_state],
    )

    studio_char_redo_btn.click(
        fn=studio_redo_single_character,
        inputs=[
            studio_char_selector, studio_char_name_edit, studio_char_face_edit,
            studio_char_hair_edit, studio_char_cloth_edit, studio_char_sig_edit,
            studio_char_prompt_edit, studio_char_seed_edit,
            studio_ollama_key, studio_ollama_model, studio_pipeline_state,
        ],
        outputs=[
            studio_char_gallery, studio_char_preview, studio_json_display,
            studio_status, studio_pipeline_state,
        ],
    )

    # Phase 3: Generate All Scene Images (First + Last Frames) & Individual Scene Inspector
    def _ui_step3(aspect_ratio, q_seed, api_key, model, current_state, selected_scene, auto_qa_chk=False, chain_chk=False, direct_end_chk=True, progress=gr.Progress(track_tqdm=True)):
        scene_items, status, state = studio_run_step3(
            aspect_ratio, q_seed, api_key, model, current_state,
            auto_qa_retry=bool(auto_qa_chk),
            chain_previous_end_frame=bool(chain_chk),
            direct_end_frame=bool(direct_end_chk),
            progress=progress,
        )
        fields = studio_load_scene_fields_dual(
            selected_scene, state, default_seed=q_seed
        )
        return scene_items, status, state, fields[4], fields[5], fields[7], fields[8], fields[9]

    studio_step3_btn.click(
        fn=_ui_step3,
        inputs=[
            studio_aspect_ratio, studio_qwen_seed, studio_ollama_key,
            studio_ollama_model, studio_pipeline_state, studio_scene_selector,
            studio_scene_qa_chk, studio_scene_chain_chk, studio_scene_direct_end_chk,
        ],
        outputs=[
            studio_scenes_gallery, studio_status, studio_pipeline_state,
            studio_scene_seed_edit, studio_scene_preview, studio_scene_end_preview,
            studio_scene_end_seed_edit, studio_scene_delta_badge,
        ],
    )

    def studio_load_scene_end_fields(choice_label, current_state, default_seed=42):
        fields = studio_load_scene_fields_dual(
            choice_label, current_state, default_seed=default_seed
        )
        return fields[6], fields[7], fields[8], fields[9]

    studio_scene_selector.change(
        fn=studio_load_scene_fields,
        inputs=[studio_scene_selector, studio_pipeline_state, studio_qwen_seed],
        outputs=[
            studio_scene_title_edit, studio_scene_scenario_edit, studio_scene_prompt_edit,
            studio_scene_chars_edit, studio_scene_seed_edit, studio_scene_preview,
        ],
    ).then(
        fn=studio_load_scene_end_fields,
        inputs=[studio_scene_selector, studio_pipeline_state, studio_qwen_seed],
        outputs=[studio_scene_end_prompt_edit, studio_scene_end_preview, studio_scene_end_seed_edit, studio_scene_delta_badge],
    )

    def _on_scene_gallery_select(evt: gr.SelectData, current_state, default_seed):
        choices = _scene_choices(current_state)
        scene_idx = _gallery_index_to_scene_index(evt.index, current_state)
        if not choices or scene_idx >= len(choices):
            return gr.update(), "", "", "", "", int(default_seed), None
        chosen = choices[scene_idx]
        fields = studio_load_scene_fields(chosen, current_state, default_seed)
        return (gr.update(value=chosen), *fields)

    studio_scenes_gallery.select(
        fn=_on_scene_gallery_select,
        inputs=[studio_pipeline_state, studio_qwen_seed],
        outputs=[
            studio_scene_selector,
            studio_scene_title_edit, studio_scene_scenario_edit, studio_scene_prompt_edit,
            studio_scene_chars_edit, studio_scene_seed_edit, studio_scene_preview,
        ],
    ).then(
        fn=studio_load_scene_end_fields,
        inputs=[studio_scene_selector, studio_pipeline_state, studio_qwen_seed],
        outputs=[studio_scene_end_prompt_edit, studio_scene_end_preview, studio_scene_end_seed_edit, studio_scene_delta_badge],
    )

    studio_scene_save_btn.click(
        fn=studio_save_scene_text_edits,
        inputs=[
            studio_scene_selector, studio_scene_title_edit, studio_scene_scenario_edit,
            studio_scene_prompt_edit, studio_scene_chars_edit, studio_pipeline_state,
            studio_scene_end_prompt_edit,
        ],
        outputs=[studio_json_display, studio_status, studio_pipeline_state],
    )

    studio_scene_direct_end_btn.click(
        fn=studio_direct_single_scene_end_frame,
        inputs=[
            studio_scene_selector, studio_ollama_key, studio_ollama_model, studio_pipeline_state,
        ],
        outputs=[studio_scene_end_prompt_edit, studio_json_display, studio_status, studio_pipeline_state],
    )

    studio_scene_qa_btn.click(
        fn=studio_verify_scene_qa,
        inputs=[
            studio_scene_selector, studio_ollama_key, studio_ollama_model, studio_pipeline_state,
        ],
        outputs=[studio_json_display, studio_status, studio_pipeline_state],
    )

    studio_scene_redo_btn.click(
        fn=studio_redo_single_scene,
        inputs=[
            studio_scene_selector, studio_scene_title_edit, studio_scene_scenario_edit,
            studio_scene_prompt_edit, studio_scene_chars_edit, studio_scene_seed_edit,
            studio_aspect_ratio, studio_ollama_key, studio_ollama_model, studio_pipeline_state,
            studio_scene_end_prompt_edit, studio_scene_redo_mode, studio_scene_qa_chk,
            studio_scene_end_seed_edit, studio_scene_direct_end_chk,
        ],
        outputs=[
            studio_scenes_gallery, studio_scene_preview, studio_scene_end_preview, studio_json_display,
            studio_status, studio_pipeline_state,
        ],
    ).then(
        fn=studio_load_scene_end_fields,
        inputs=[studio_scene_selector, studio_pipeline_state, studio_qwen_seed],
        outputs=[studio_scene_end_prompt_edit, studio_scene_end_preview, studio_scene_end_seed_edit, studio_scene_delta_badge],
    )

    # Phase 4: Director, Generate All Video Clips, and Individual Clip Redo
    def _ui_director(api_key, model, use_vision, current_state, selected_clip, w_seed, progress=gr.Progress(track_tqdm=True)):
        json_out, status, state = studio_run_director(api_key, model, use_vision, current_state, progress=progress)
        v_mot, v_cam, v_wprompt, v_seed, v_vid = studio_load_clip_fields(selected_clip, state, default_seed=w_seed)
        return json_out, status, state, v_mot, v_cam, v_wprompt

    studio_step_director_btn.click(
        fn=_ui_director,
        inputs=[
            studio_ollama_key, studio_ollama_model, studio_use_vision,
            studio_pipeline_state, studio_clip_selector, studio_wan_seed,
        ],
        outputs=[
            studio_json_display, studio_status, studio_pipeline_state,
            studio_clip_motion_edit, studio_clip_camera_edit, studio_clip_wan_prompt_edit,
        ],
    )

    def _ui_step4(
        aspect_ratio, w_seed, api_key, model, use_vision, current_state, selected_clip,
        use_last_frame=True, enable_rife_2x=True, anti_noise_guard=True,
        progress=gr.Progress(track_tqdm=True),
    ):
        final_vid, status, state = studio_run_step4(
            aspect_ratio, w_seed, api_key, model, use_vision, current_state,
            use_last_frame=bool(use_last_frame),
            enable_rife_2x=bool(enable_rife_2x),
            anti_noise_guard=bool(anti_noise_guard),
            progress=progress,
        )
        v_mot, v_cam, v_wprompt, v_seed, v_vid = studio_load_clip_fields(selected_clip, state, default_seed=w_seed)
        return final_vid, status, state, v_mot, v_cam, v_wprompt, v_seed, v_vid

    studio_step4_btn.click(
        fn=_ui_step4,
        inputs=[
            studio_aspect_ratio, studio_wan_seed, studio_ollama_key,
            studio_ollama_model, studio_use_vision, studio_pipeline_state, studio_clip_selector,
            studio_use_last_frame_chk, studio_rife_2x_chk, studio_anti_noise_chk,
        ],
        outputs=[
            studio_final_video, studio_status, studio_pipeline_state,
            studio_clip_motion_edit, studio_clip_camera_edit, studio_clip_wan_prompt_edit,
            studio_clip_seed_edit, studio_clip_preview,
        ],
    )

    studio_clip_selector.change(
        fn=studio_load_clip_fields,
        inputs=[studio_clip_selector, studio_pipeline_state, studio_wan_seed],
        outputs=[
            studio_clip_motion_edit, studio_clip_camera_edit, studio_clip_wan_prompt_edit,
            studio_clip_seed_edit, studio_clip_preview,
        ],
    )

    studio_clip_director_btn.click(
        fn=studio_redirect_single_clip,
        inputs=[
            studio_clip_selector, studio_clip_motion_edit, studio_clip_camera_edit,
            studio_use_vision, studio_ollama_key, studio_ollama_model, studio_pipeline_state,
        ],
        outputs=[
            studio_clip_camera_edit, studio_clip_wan_prompt_edit,
            studio_json_display, studio_status, studio_pipeline_state,
        ],
    )

    studio_clip_redo_btn.click(
        fn=studio_redo_single_clip,
        inputs=[
            studio_clip_selector, studio_clip_motion_edit, studio_clip_camera_edit,
            studio_clip_wan_prompt_edit, studio_clip_seed_edit,
            studio_clip_rerundir_chk, studio_use_vision, studio_aspect_ratio,
            studio_ollama_key, studio_ollama_model, studio_pipeline_state,
            studio_use_last_frame_chk, studio_rife_2x_chk, studio_anti_noise_chk,
        ],
        outputs=[
            studio_clip_preview, studio_final_video, studio_json_display,
            studio_status, studio_pipeline_state,
        ],
    )

    studio_run_full_btn.click(
        fn=studio_run_full,
        inputs=[
            studio_prompt, studio_art_style, studio_aspect_ratio,
            studio_seed, studio_num_scenes, studio_total_duration,
            studio_ollama_key, studio_ollama_model, studio_use_vision,
            studio_qwen_seed, studio_wan_seed,
        ],
        outputs=[
            studio_final_video, studio_char_gallery, studio_scenes_gallery,
            studio_json_display, studio_status, studio_pipeline_state,
        ],
    )

    # ── Heartbeat Toggles (All 3 Models Active by Default on Startup) ─────────
    async def toggle_z_image_heartbeat():
        import backend
        new_state = not backend.z_image_heartbeat_enabled
        set_z_image_heartbeat_state(new_state)
        if new_state:
            return gr.update(value="💓 Z-Image Turbo Keep-Warm: Active", variant="primary")
        else:
            return gr.update(value="💤 Z-Image Turbo Keep-Warm: Paused", variant="secondary")

    async def toggle_wan_heartbeat():
        import backend
        new_state = not backend.wan_heartbeat_enabled
        set_wan_heartbeat_state(new_state)
        if new_state:
            return gr.update(value="💓 Wan 2.2 Keep-Warm: Active", variant="primary")
        else:
            return gr.update(value="💤 Wan 2.2 Keep-Warm: Paused", variant="secondary")

    async def toggle_qwen_heartbeat():
        import backend
        new_state = not backend.qwen_heartbeat_enabled
        set_qwen_heartbeat_state(new_state)
        if new_state:
            return gr.update(value="💓 Qwen Keep-Warm: Active", variant="primary")
        else:
            return gr.update(value="💤 Qwen Keep-Warm: Paused", variant="secondary")

    z_image_hb_btn.click(
        fn=toggle_z_image_heartbeat,
        inputs=[],
        outputs=[z_image_hb_btn],
    )

    wan_hb_btn.click(
        fn=toggle_wan_heartbeat,
        inputs=[],
        outputs=[wan_hb_btn],
    )

    qwen_hb_btn.click(
        fn=toggle_qwen_heartbeat,
        inputs=[],
        outputs=[qwen_hb_btn],
    )

    clear_outputs_btn.click(
        fn=clear_all_outputs,
        inputs=[],
        outputs=[],
    )

    # ── Periodic Client Heartbeat Timer ──────────────────────────────────────
    async def handle_client_heartbeat():
        record_frontend_heartbeat()
        return None

    heartbeat_timer = gr.Timer(value=25, active=True)
    heartbeat_timer.tick(
        fn=handle_client_heartbeat,
        inputs=None,
        outputs=None,
    )


if __name__ == "__main__":
    share_lan = os.environ.get("WAN_SHARE_LAN") == "1"
    server_name = "0.0.0.0" if share_lan else None

    # Start background heartbeat to keep Z-Image Turbo, Qwen, and Wan 2.2 warm while frontend is open
    start_keepalive_thread()

    demo.queue().launch(
        server_name=server_name,
        show_error=True,
        css=CSS,
        head=DRAG_DROP_JS,
    )

