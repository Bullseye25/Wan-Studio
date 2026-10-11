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
    set_qwen_heartbeat_state
)
from studio_pipeline import (
    StudioPipeline,
    ASPECT_RATIO_PRESETS,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_MODEL,
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


def extract_frame(video_path, timestamp):
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_num = int(float(timestamp) * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame_num >= total_frames:
        target_frame_num = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Wan 2.2 & Qwen Image Studio") as demo:

    gr.Markdown("## 🎨 Wan 2.2 & Qwen Image Studio", elem_classes="main-title")
    gr.Markdown(
        "**Tool 1 🖼️** — Qwen Image Edit Rapid AIO   |   "
        "**Tool 2 🎥** — Wan 2.2 Video Generator   |   "
        "**Tool 3 🎬** — Studio Automated Director",
        elem_classes="subtitle"
    )

    with gr.Row():
        wan_hb_btn = gr.Button("💤 Wan 2.2 Keep-Warm: Paused", variant="secondary", scale=1)
        qwen_hb_btn = gr.Button("💤 Qwen Keep-Warm: Paused", variant="secondary", scale=1)
        clear_outputs_btn = gr.Button("🗑️ Clear All Outputs", variant="stop", scale=1)

    with gr.Row(elem_classes="toolbar-row"):
        expand_all_btn = gr.Button("🔽 Expand All Tools", variant="secondary", size="sm")
        collapse_all_btn = gr.Button("🔼 Collapse All Tools", variant="secondary", size="sm")

    # ── Tool 1: Qwen Image Edit Rapid AIO Generator / Editor ──────────────────
    with gr.Accordion("🖼️ Tool 1 — Qwen Image Edit Rapid AIO (Image Generator & Editor)", open=True, elem_classes="tool-accordion qwen-panel") as qwen_accordion:
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

    # ── Tool 2: Wan 2.2 Video Generator ───────────────────────────────────────
    with gr.Accordion("🎥 Tool 2 — Wan 2.2 · High-Motion Video Generator", open=True, elem_classes="tool-accordion wan-panel") as wan_accordion:
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

    # ── Tool 3: Studio Multi-Agent Automated Story & Video Production Studio ────
    with gr.Accordion("🎬 Tool 3 — Studio Automated Director (Story ➔ Z-Image Turbo ➔ Qwen Multi-Ref ➔ Wan 2.2)", open=True, elem_classes="tool-accordion studio-panel") as studio_accordion:
        gr.Markdown("### 🎬 Studio Automated Director · End-to-End Multi-Agent Video Pipeline", elem_classes="section-header")
        gr.Markdown(
            "1. **Story Architect (Gemma 4: 31B)** generates a structured 30s screenplay and extracts character visual details.\n"
            "2. **Character Generator (Z-Image Turbo)** synthesizes portraits using an **identical locked seed** for maximum consistency.\n"
            "3. **Scene Synthesizer (Qwen Image Edit)** transfers character faces, hair, and clothes as multi-references across all scenes.\n"
            "4. **Scene Animator (Wan 2.2)** animates clips adhering to 4k+1 frame math and CFG 2.0 stability.\n"
            "5. **Film Assembly (FFmpeg)** concatenates all clips into a seamless final 30-second movie!",
            elem_classes="step-hint"
        )

        with gr.Row():
            with gr.Column(scale=1):
                studio_prompt = gr.Textbox(
                    label="Story Premise / Idea",
                    value="Two best friends, Barnaby the fluffy brown bear and Pippin the white rabbit, discover a basket of sweet wild berries in an enchanted forest meadow and learn that sharing brings true happiness.",
                    placeholder="Describe your story premise or thematic concept...",
                    lines=3,
                )
                with gr.Row():
                    studio_art_style = gr.Dropdown(
                        label="Art Style",
                        choices=[
                            "2D vibrant cartoon animation",
                            "Cyberpunk anime aesthetic",
                            "3D Pixar / Disney animation style",
                            "Photorealistic cinematic",
                            "Watercolor children's book illustration"
                        ],
                        value="2D vibrant cartoon animation",
                        scale=1
                    )
                    studio_aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio / Format",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="9:16 Vertical (Shorts/Reels)",
                        scale=1
                    )
                with gr.Row():
                    studio_seed = gr.Slider(
                        minimum=0, maximum=MAX_SEED, step=1, value=42,
                        label="Character Locked Seed (Ensures Identical Faces)",
                        scale=2
                    )
                    studio_num_scenes = gr.Slider(
                        minimum=3, maximum=6, step=1, value=5,
                        label="Scenes Count",
                        scale=1
                    )
                    studio_total_duration = gr.Slider(
                        minimum=15.0, maximum=30.0, step=2.5, value=30.0,
                        label="Total Duration (s)",
                        scale=1
                    )

                with gr.Accordion("⚙️ AI Model & API Configuration", open=False):
                    with gr.Row():
                        studio_ollama_key = gr.Textbox(
                            label="Ollama Cloud API Key",
                            value=DEFAULT_OLLAMA_API_KEY,
                            type="password",
                            scale=2
                        )
                        studio_ollama_model = gr.Textbox(
                            label="Ollama Model",
                            value=DEFAULT_OLLAMA_MODEL,
                            scale=1
                        )

                studio_run_full_btn = gr.Button("🚀 Run Full Automated 30s Pipeline", variant="primary", size="lg")

                with gr.Accordion("🛠️ Step-by-Step Production Controls", open=False):
                    gr.Markdown("Inspect or execute each production stage individually:", elem_classes="step-hint")
                    with gr.Row():
                        studio_step1_btn = gr.Button("1️⃣ Story & Character Extraction", variant="secondary")
                        studio_step2_btn = gr.Button("2️⃣ Z-Image Turbo Portraits", variant="secondary")
                    with gr.Row():
                        studio_step3_btn = gr.Button("3️⃣ Qwen Scene Keyframes", variant="secondary")
                        studio_step4_btn = gr.Button("4️⃣ Animate & Stitch Wan 2.2", variant="secondary")

            with gr.Column(scale=1):
                studio_status = gr.Markdown("### 🟢 Pipeline Ready\nEnter a prompt and click **Run Full Automated 30s Pipeline**.")
                studio_final_video = gr.Video(label="🎬 Final Assembled 30-Second Movie", autoplay=True, interactive=False)

                with gr.Accordion("👤 Character Reference Portraits (Z-Image Turbo)", open=True):
                    studio_char_gallery = gr.Gallery(
                        label="Consistent Character References (Locked Seed)",
                        columns=[2, 3],
                        height=200,
                        object_fit="contain",
                        preview=True
                    )

                with gr.Accordion("🖼️ Synthesized Scene Keyframes (Qwen Multi-Reference)", open=True):
                    studio_scenes_gallery = gr.Gallery(
                        label="Scene Frames with Anchored Identity",
                        columns=[2, 3],
                        height=200,
                        object_fit="contain",
                        preview=True
                    )

                with gr.Accordion("📝 Screenplay & Character JSON Data", open=False):
                    studio_json_display = gr.JSON(label="Gemma 4: 31B Screenplay Data")

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
        lora_repo_or_path_input, lora_scale_slider, deflicker_checkbox
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

    # ── Events: Studio Multi-Agent Automated Pipeline ─────────────────────────

    studio_pipeline_state = gr.State(value={})

    def studio_run_full(
        prompt, art_style, aspect_ratio, seed, num_scenes, duration, api_key, model,
        progress=gr.Progress(track_tqdm=True)
    ):
        pipeline = StudioPipeline(api_key=api_key, model=model)

        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        try:
            story_data = pipeline.run_full_pipeline(
                user_prompt=prompt,
                shared_seed=int(seed),
                num_scenes=int(num_scenes),
                target_duration=float(duration),
                aspect_ratio=aspect_ratio,
                art_style=art_style,
                progress_cb=progress_callback,
            )

            char_items = [
                (c["portrait_path"], f"{c.get('name')} (Seed: {c.get('portrait_seed')})")
                for c in story_data.get("characters", [])
                if "portrait_path" in c
            ]
            scene_items = [
                (s["keyframe_path"], f"Scene {s.get('scene_number')}: {s.get('title')}")
                for s in story_data.get("scenes", [])
                if "keyframe_path" in s
            ]
            final_video = story_data.get("final_movie_path")

            clean_json = {
                "title": story_data.get("title"),
                "art_style": story_data.get("art_style"),
                "theme": story_data.get("theme"),
                "characters": [
                    {k: v for k, v in c.items() if k not in ["portrait_pil", "portrait_path"]}
                    for c in story_data.get("characters", [])
                ],
                "scenes": [
                    {k: v for k, v in s.items() if k not in ["keyframe_pil"]}
                    for s in story_data.get("scenes", [])
                ],
                "final_movie_path": final_video,
            }

            status_msg = (
                f"### 🎬 Production Complete!\n\n"
                f"- **Screenplay:** '{story_data.get('title')}'\n"
                f"- **Characters Synthesized:** {len(char_items)} (Z-Image Turbo, locked seed {seed})\n"
                f"- **Scene Keyframes:** {len(scene_items)} (Qwen Multi-Reference Transfer)\n"
                f"- **Final 30s Movie:** `{pathlib.Path(final_video).name if final_video else 'N/A'}`"
            )
            return final_video, char_items, scene_items, clean_json, status_msg, story_data
        except Exception as e:
            err_msg = f"### ❌ Studio Production Error\n`{str(e)}`"
            return None, [], [], {}, err_msg, {}

    def studio_run_step1(prompt, art_style, num_scenes, duration, api_key, model, progress=gr.Progress(track_tqdm=True)):
        pipeline = StudioPipeline(api_key=api_key, model=model)

        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        try:
            story_data = pipeline.generate_story_and_characters(
                prompt=prompt,
                num_scenes=int(num_scenes),
                target_duration=float(duration),
                art_style=art_style,
                progress_cb=progress_callback,
            )
            clean_json = {
                "title": story_data.get("title"),
                "art_style": story_data.get("art_style"),
                "theme": story_data.get("theme"),
                "characters": story_data.get("characters", []),
                "scenes": story_data.get("scenes", []),
            }
            status_msg = (
                f"### 📝 Step 1 Complete: Screenplay & Characters Generated\n"
                f"Extracted {len(story_data.get('characters', []))} character(s) and {len(story_data.get('scenes', []))} scene(s)."
            )
            return clean_json, status_msg, story_data
        except Exception as e:
            return {}, f"### ❌ Step 1 Error\n`{str(e)}`", {}

    def studio_run_step2(seed, api_key, model, current_state, progress=gr.Progress(track_tqdm=True)):
        if not current_state or "characters" not in current_state:
            return [], "### ⚠️ Please run Step 1 (Story & Characters) first before generating character portraits.", current_state
        pipeline = StudioPipeline(api_key=api_key, model=model)

        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        try:
            chars = pipeline.generate_character_portraits(
                characters=current_state["characters"],
                shared_seed=int(seed),
                progress_cb=progress_callback,
            )
            current_state["characters"] = chars
            char_items = [
                (c["portrait_path"], f"{c.get('name')} (Seed: {c.get('portrait_seed')})")
                for c in chars
                if "portrait_path" in c
            ]
            status_msg = (
                f"### 👤 Step 2 Complete: {len(chars)} Character Portraits Synthesized\n"
                f"All characters locked to seed `{seed}`."
            )
            return char_items, status_msg, current_state
        except Exception as e:
            return [], f"### ❌ Step 2 Error\n`{str(e)}`", current_state

    def studio_run_step3(aspect_ratio, seed, api_key, model, current_state, progress=gr.Progress(track_tqdm=True)):
        if not current_state or "characters" not in current_state or "scenes" not in current_state:
            return [], "### ⚠️ Please run Step 1 & Step 2 first before generating scene keyframes.", current_state
        pipeline = StudioPipeline(api_key=api_key, model=model)

        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        try:
            scenes = pipeline.generate_scene_keyframes(
                scenes=current_state["scenes"],
                characters=current_state["characters"],
                aspect_ratio=aspect_ratio,
                shared_seed=int(seed),
                progress_cb=progress_callback,
            )
            current_state["scenes"] = scenes
            scene_items = [
                (s["keyframe_path"], f"Scene {s.get('scene_number')}: {s.get('title')}")
                for s in scenes
                if "keyframe_path" in s
            ]
            status_msg = (
                f"### 🖼️ Step 3 Complete: {len(scenes)} Scene Keyframes Synthesized\n"
                f"Character identities locked using multi-reference transfer."
            )
            return scene_items, status_msg, current_state
        except Exception as e:
            return [], f"### ❌ Step 3 Error\n`{str(e)}`", current_state

    def studio_run_step4(aspect_ratio, api_key, model, current_state, progress=gr.Progress(track_tqdm=True)):
        if not current_state or "scenes" not in current_state:
            return None, "### ⚠️ Please run Step 3 (Scene Keyframes) first before animating and stitching.", current_state
        pipeline = StudioPipeline(api_key=api_key, model=model)

        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        try:
            clips = pipeline.generate_scene_videos(
                scenes=current_state["scenes"],
                aspect_ratio=aspect_ratio,
                progress_cb=progress_callback,
            )
            current_state["clip_paths"] = clips
            final_movie = pipeline.stitch_clips(
                clip_paths=clips,
                movie_title=current_state.get("title", "Studio_Movie"),
                progress_cb=progress_callback,
            )
            current_state["final_movie_path"] = final_movie
            status_msg = (
                f"### 🎬 Step 4 Complete: Movie Stitched Successfully!\n"
                f"Final file: `{pathlib.Path(final_movie).name}`."
            )
            return final_movie, status_msg, current_state
        except Exception as e:
            return None, f"### ❌ Step 4 Error\n`{str(e)}`", current_state

    studio_run_full_btn.click(
        fn=studio_run_full,
        inputs=[
            studio_prompt, studio_art_style, studio_aspect_ratio,
            studio_seed, studio_num_scenes, studio_total_duration,
            studio_ollama_key, studio_ollama_model
        ],
        outputs=[
            studio_final_video, studio_char_gallery, studio_scenes_gallery,
            studio_json_display, studio_status, studio_pipeline_state
        ]
    )

    studio_step1_btn.click(
        fn=studio_run_step1,
        inputs=[
            studio_prompt, studio_art_style, studio_num_scenes,
            studio_total_duration, studio_ollama_key, studio_ollama_model
        ],
        outputs=[studio_json_display, studio_status, studio_pipeline_state]
    )

    studio_step2_btn.click(
        fn=studio_run_step2,
        inputs=[studio_seed, studio_ollama_key, studio_ollama_model, studio_pipeline_state],
        outputs=[studio_char_gallery, studio_status, studio_pipeline_state]
    )

    studio_step3_btn.click(
        fn=studio_run_step3,
        inputs=[studio_aspect_ratio, studio_seed, studio_ollama_key, studio_ollama_model, studio_pipeline_state],
        outputs=[studio_scenes_gallery, studio_status, studio_pipeline_state]
    )

    studio_step4_btn.click(
        fn=studio_run_step4,
        inputs=[studio_aspect_ratio, studio_ollama_key, studio_ollama_model, studio_pipeline_state],
        outputs=[studio_final_video, studio_status, studio_pipeline_state]
    )

    # ── Heartbeat Toggles ────────────────────────────────────────────────────
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

    # Start background heartbeat to prevent Modal GPU containers from scaling down
    start_keepalive_thread()

    demo.queue().launch(
        server_name=server_name,
        show_error=True,
        css=CSS,
        head=DRAG_DROP_JS,
    )
