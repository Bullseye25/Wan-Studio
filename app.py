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
    VERIFIED_OLLAMA_MODELS,
    generate_three_studio_seeds,
    restore_extracted_video_frame,
)
from sprites_pipeline import SpritesPipeline, GENRE_PRESETS
from sprite_cutter import process_video_to_sprite_sheet

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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
code, pre, .mono-font { font-family: 'JetBrains Mono', monospace !important; }

body, .gradio-container {
    background: radial-gradient(circle at 50% 0%, #17153a 0%, #0a0b10 55%, #030508 100%) !important;
    min-height: 100vh;
    color: #f8fafc;
}

/* Glassmorphic Panels */
.glass-panel {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    padding: 16px !important;
}

.studio-panel {
    background: linear-gradient(145deg, rgba(30, 27, 75, 0.45), rgba(15, 23, 42, 0.65)) !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 35px rgba(168, 85, 247, 0.08) !important;
    padding: 18px !important;
}

.sprites-panel {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(15, 23, 42, 0.65)) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 35px rgba(16, 185, 129, 0.08) !important;
    padding: 18px !important;
}

.qwen-panel {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.12), rgba(15, 23, 42, 0.65)) !important;
    border: 1px solid rgba(99, 102, 241, 0.28) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 35px rgba(99, 102, 241, 0.08) !important;
    padding: 18px !important;
}

.wan-panel {
    background: linear-gradient(145deg, rgba(14, 165, 233, 0.12), rgba(15, 23, 42, 0.65)) !important;
    border: 1px solid rgba(14, 165, 233, 0.28) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 35px rgba(14, 165, 233, 0.08) !important;
    padding: 18px !important;
}

/* Header Branding */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2.2em !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 4px !important;
}

.subtitle {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 0.95em !important;
    margin-bottom: 14px !important;
}

/* Pipeline Progression Rail */
.pipeline-progression-rail {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
    margin-bottom: 16px !important;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

/* Top Level Main Workspace Tabs */
.main-workspace-tabs > .tab-nav {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    margin-bottom: 16px !important;
}

.main-workspace-tabs > .tab-nav button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    color: #94a3b8 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease-in-out !important;
}

.main-workspace-tabs > .tab-nav button.selected {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(168, 85, 247, 0.22)) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.45) !important;
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.3) !important;
}

/* Buttons Glow & Polish */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover, .gr-button-primary:hover {
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.55) !important;
    transform: translateY(-1px) !important;
}

button.secondary, .gr-button-secondary {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.22) !important;
}

/* Tool Accordion Styling (fallback/nested) */
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
    font-size: 1.15em !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em;
}

.step-hint {
    color: #94a3b8;
    font-size: 0.88em;
    margin-top: 2px;
    margin-bottom: 10px;
    line-height: 1.5;
}

.director-panel {
    background: linear-gradient(145deg, rgba(168, 85, 247, 0.12), rgba(99, 102, 241, 0.08)) !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    border-radius: 14px !important;
    padding: 14px !important;
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
            f"### 🔄 Character Redo Complete (9:16 Full-Body): `{updated_char.get('name')}`\n"
            f"Regenerated in 9:16 vertical full-body orientation (576x1024) with Z-Image Turbo using seed `{int(char_seed)}`."
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
    chain_previous_end_frame=False,
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
            chain_previous_end_frame=bool(chain_previous_end_frame),
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
        chain_note = f"\n- **🔗 Shot Continuity:** Chained seamlessly from Scene #{idx} End Frame (Zero Drift)" if (chain_previous_end_frame and idx > 0) else ""
        status = (
            f"### 🎯 Scene #{idx+1} Keyframe Redo Complete (`{which_frame}`): `{updated_scene.get('title')}`\n"
            f"- **Start Seed:** `{start_seed}` | **End Seed:** `{updated_scene.get('last_keyframe_seed', end_s)}`"
            + delta_note
            + qa_note
            + chain_note
        )
        return scene_items, updated_scene.get("keyframe_path"), updated_scene.get("last_keyframe_path"), StudioPipeline.serializable(current_state), status, current_state
    except Exception as e:
        _, scene_items = _studio_gallery_items(current_state)
        return scene_items, None, None, StudioPipeline.serializable(current_state), f"### ❌ Scene Generation Error\n`{e}`", current_state


def studio_step_to_next_scene(choice_label, current_state, default_seed=42):
    """Advances selection to the next scene in the story and populates its dual-keyframe fields."""
    scenes = (current_state or {}).get("scenes", [])
    choices = _scene_choices(current_state)
    if not scenes or not choices:
        return (
            gr.update(choices=[], value=None),
            "", "", "", "", int(default_seed), None,
            "", None, int((default_seed + 10007) % (2**31 - 1)), "Pending generation",
            "### ⚠️ No scenes available.",
        )
    current_idx = _parse_selector_index(choice_label, choices)
    next_idx = (current_idx + 1) % len(choices)
    next_choice = choices[next_idx]
    fields = studio_load_scene_fields_dual(next_choice, current_state, default_seed=default_seed)
    status = (
        f"### 🎬 Switched to Scene #{next_idx + 1}/{len(choices)}: `{scenes[next_idx].get('title', '')}`\n"
        f"Review prompts and click **'🎯 Generate Selected Scene (Scene-by-Scene Mode)'**."
    )
    return (
        gr.update(choices=choices, value=next_choice),
        fields[0], fields[1], fields[2], fields[3], fields[4], fields[5],
        fields[6], fields[7], fields[8], fields[9],
        status,
    )


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


# ── Sprites Creator Studio Handlers ──────────────────────────────────────────

def sprites_ui_refine_character(user_concept, genre, art_style, api_key, model):
    pipeline = SpritesPipeline(api_key=api_key, model=model)
    try:
        refined = pipeline.refine_character_with_gemma(
            user_input=user_concept or "Arcade protagonist",
            genre=genre,
            art_style=art_style,
        )
        arch = SpritesPipeline.detect_character_combat_archetype(refined, genre=genre)
        status = (
            f"### ✨ Gemma Refined 2D Game Character: `{refined.get('name')}`\n"
            f"- **Genre:** `{genre}` | **Style:** `{art_style}`\n"
            f"- **Detected Combat Archetype:** `{arch['role_title']}` | **Weapon/Gear:** `{arch['weapon_label']}`\n"
            f"- **Choreography Focus:** {arch['combat_style_summary']}"
        )
        return (
            refined.get("name", "Hero"),
            refined.get("title", ""),
            refined.get("appearance", ""),
            refined.get("signature_gear", ""),
            refined.get("z_image_prompt", ""),
            status,
        )
    except Exception as e:
        return "Hero", "Arcade Fighter", user_concept, "Standard Gear", "", f"### ⚠️ Gemma Refine Error\n`{e}`"


def sprites_ui_generate_master_sprite(
    name, title, appearance, gear, z_prompt, seed, genre, art_style, api_key, model, current_state, user_concept=""
):
    pipeline = SpritesPipeline(api_key=api_key, model=model)

    # Smart auto-enrichment if the user typed a concept (e.g. "Samurai with a sword") and clicked Generate directly
    eff_name = (name or "").strip()
    eff_title = (title or "").strip()
    eff_appearance = (appearance or "").strip()
    eff_gear = (gear or "").strip()
    eff_prompt = (z_prompt or "").strip()
    concept_str = (user_concept or "").strip()

    if concept_str and (not eff_appearance or not eff_gear or eff_name in ("", "Pepsi_Hero")):
        refined = pipeline.refine_character_with_gemma(
            user_input=concept_str,
            genre=genre,
            art_style=art_style,
        )
        if not eff_name or eff_name == "Pepsi_Hero":
            eff_name = refined.get("name") or eff_name
        if not eff_title or eff_title == "Endless Obstacle Runner":
            eff_title = refined.get("title") or eff_title
        if not eff_appearance:
            eff_appearance = refined.get("appearance") or concept_str
        if not eff_gear:
            eff_gear = refined.get("signature_gear") or ""
        if not eff_prompt:
            eff_prompt = refined.get("z_image_prompt") or ""

    char_data = {
        "name": eff_name or "Hero",
        "title": eff_title or "Arcade Fighter",
        "appearance": eff_appearance or concept_str or "",
        "signature_gear": eff_gear or "",
        "user_concept": concept_str,
        "z_image_prompt": eff_prompt or f"2D game character sheet of {eff_name or 'Hero'} {('holding ' + eff_gear) if eff_gear else ''}",
        "genre": genre,
        "art_style": art_style,
    }
    try:
        pil_img, local_path = pipeline.synthesize_master_character_sprite(char_data, seed=int(seed))
        char_data["master_sprite_pil"] = pil_img
        char_data["master_sprite_path"] = local_path
        char_data["master_seed"] = int(seed)

        # Smart Character & Weapon-Aware Move-Set Generation (uses Gemma when api_key is provided, contextual weapon-aware fallback otherwise)
        moves = pipeline.generate_smart_moveset_with_gemma(
            character_data=char_data,
            genre=genre,
            use_llm=bool(api_key),
        )
        char_data["moves"] = moves
        arch = SpritesPipeline.detect_character_combat_archetype(char_data, genre=genre)
        char_data["combat_archetype"] = arch

        current_state = current_state or {}
        current_state["character"] = char_data

        move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
        status = (
            f"### 🎨 Master 2D Sprite & Smart Choreography Ready: `{char_data['name']}` (Seed {seed})\n"
            f"- **Weapon / Gear Detected:** `{arch['weapon_label']}` (`{arch['role_title']}`)\n"
            f"- **Smart Move-Set:** Tailored `{len(moves)}` character-specific moves & pre-animation instructions for `{genre}`."
        )
        first_choice = move_choices[0] if move_choices else None
        return (
            local_path,
            gr.update(choices=move_choices, value=first_choice),
            gr.update(choices=move_choices, value=first_choice),
            gr.update(choices=move_choices, value=first_choice),
            moves,
            status,
            current_state,
        )
    except Exception as e:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), [], f"### ❌ Sprite Generation Error\n`{e}`", current_state


def sprites_ui_smart_choreograph_moveset(
    char_state, genre, combat_notes, name, title, appearance, gear, api_key, model
):
    """
    Uses Gemma 4: 31B to intelligently generate or re-architect the character's entire move-set
    and pre-animation instructions based on their weapon/gear, appearance, and combat style notes.
    """
    char_state = char_state or {}
    character_data = dict(char_state.get("character") or {})
    if name and str(name).strip():
        character_data["name"] = str(name).strip()
    if title and str(title).strip():
        character_data["title"] = str(title).strip()
    if appearance and str(appearance).strip():
        character_data["appearance"] = str(appearance).strip()
    if gear and str(gear).strip():
        character_data["signature_gear"] = str(gear).strip()
    character_data["genre"] = genre

    pipeline = SpritesPipeline(api_key=api_key, model=model)
    moves = pipeline.generate_smart_moveset_with_gemma(
        character_data=character_data,
        genre=genre,
        combat_notes=combat_notes or "",
        use_llm=bool(api_key),
    )
    character_data["moves"] = moves
    arch = SpritesPipeline.detect_character_combat_archetype(character_data, genre=genre)
    character_data["combat_archetype"] = arch
    char_state["character"] = character_data

    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    first_choice = move_choices[0] if move_choices else None
    first_move = moves[0] if moves else {}
    first_type_label = (
        "🔁 Seamless Looping Cycle" if first_move.get("type") == "loop" else "🎯 Action (Return to Idle Stance)"
    )

    status = (
        f"### 🧠 Gemma Smart Choreography Complete for `{character_data.get('name', 'Hero')}`!\n"
        f"- **Combat Archetype:** `{arch['role_title']}` | **Weapon / Gear:** `{arch['weapon_label']}`\n"
        f"- **Style Focus:** {arch['combat_style_summary']}\n"
        f"- **Tailored Move-Set:** Generated `{len(moves)}` moves with weapon-locked pre-animation Wan 2.2 instructions."
    )
    return (
        gr.update(choices=move_choices, value=first_choice),
        gr.update(choices=move_choices, value=first_choice),
        gr.update(choices=move_choices, value=first_choice),
        moves,
        first_move.get("name", ""),
        first_type_label,
        first_move.get("description", ""),
        first_move.get("refined_prompt", ""),
        status,
        char_state,
    )


def sprites_ui_select_move_for_edit(choice_label, char_state):
    """Loads the selected move's pre-animation choreography instructions into the Phase 2 editor."""
    info, vpath = sprites_ui_load_move_details(choice_label, char_state)
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    moves = character_data.get("moves", [])
    for m in moves:
        label = f"{m['name']} ({m['type'].upper()})"
        if choice_label == label or m["name"] in str(choice_label):
            type_label = (
                "🔁 Seamless Looping Cycle" if m.get("type") == "loop" else "🎯 Action (Return to Idle Stance)"
            )
            return (
                m.get("name", ""),
                type_label,
                m.get("description", ""),
                m.get("refined_prompt", ""),
                info,
                vpath,
            )
    return "", "🎯 Action (Return to Idle Stance)", "", "", info, vpath


def sprites_ui_rechoreograph_selected_move(
    choice_label, edit_name, edit_type, edit_desc, genre, api_key, model, char_state
):
    """Uses Gemma 4: 31B to rewrite the selected move's pre-animation instructions for the character's weapon/gear."""
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    moves = list(character_data.get("moves", []))
    if not character_data or not moves:
        return "", "", gr.update(), gr.update(), gr.update(), [], "### ⚠️ Please generate a character first in Phase 1.", char_state

    idx = 0
    for i, m in enumerate(moves):
        label = f"{m['name']} ({m['type'].upper()})"
        if choice_label == label or m["name"] in str(choice_label):
            idx = i
            break

    target_move = dict(moves[idx])
    anim_name = (edit_name or target_move.get("name") or "Move").strip()
    anim_type = "loop" if "loop" in str(edit_type).lower() else "action"
    raw_desc = (edit_desc or target_move.get("description") or anim_name).strip()

    pipeline = SpritesPipeline(api_key=api_key, model=model)
    plan = pipeline.refine_custom_animation_with_gemma(
        character_data=character_data,
        genre=genre,
        anim_name=anim_name,
        raw_description=raw_desc,
        anim_type=anim_type,
    )
    new_desc = plan.get("choreography_prompt") or plan.get("action_summary") or raw_desc
    new_prompt = plan.get("wan_prompt") or ""

    target_move["name"] = anim_name
    target_move["type"] = anim_type
    target_move["description"] = new_desc
    target_move["refined_prompt"] = new_prompt
    moves[idx] = target_move
    character_data["moves"] = moves
    char_state["character"] = character_data

    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    selected_label = move_choices[idx]
    status = (
        f"### ✨ Gemma Re-Choreographed Move: `{anim_name}` ({anim_type.upper()})\n"
        f"- **Summary:** {plan.get('action_summary', new_desc)}"
    )
    return (
        new_desc,
        new_prompt,
        gr.update(choices=move_choices, value=selected_label),
        gr.update(choices=move_choices, value=selected_label),
        gr.update(choices=move_choices, value=selected_label),
        moves,
        status,
        char_state,
    )


def sprites_ui_save_selected_move(
    choice_label, edit_name, edit_type, edit_desc, edit_prompt, genre, char_state
):
    """Saves manual edits to a move's name, motion rule, choreography description, and Wan prompt."""
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    moves = list(character_data.get("moves", []))
    if not character_data or not moves:
        return gr.update(), gr.update(), gr.update(), [], "### ⚠️ No active move-set to update.", char_state

    idx = 0
    for i, m in enumerate(moves):
        label = f"{m['name']} ({m['type'].upper()})"
        if choice_label == label or m["name"] in str(choice_label):
            idx = i
            break

    target_move = dict(moves[idx])
    anim_name = (edit_name or target_move.get("name") or f"Move {idx+1}").strip()
    anim_type = "loop" if "loop" in str(edit_type).lower() else "action"
    desc_str = (edit_desc or target_move.get("description") or "").strip()
    prompt_str = (edit_prompt or "").strip()
    if not prompt_str:
        prompt_str = SpritesPipeline.build_pre_animation_wan_prompt(
            character_data=character_data,
            genre=genre,
            move_name=anim_name,
            move_type=anim_type,
            move_description=desc_str,
        )

    target_move["name"] = anim_name
    target_move["type"] = anim_type
    target_move["description"] = desc_str
    target_move["refined_prompt"] = prompt_str
    moves[idx] = target_move
    character_data["moves"] = moves
    char_state["character"] = character_data

    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    selected_label = move_choices[idx]
    status = f"### 💾 Saved Pre-Animation Instructions for `{anim_name}` ({anim_type.upper()})."
    return (
        gr.update(choices=move_choices, value=selected_label),
        gr.update(choices=move_choices, value=selected_label),
        gr.update(choices=move_choices, value=selected_label),
        moves,
        status,
        char_state,
    )


def sprites_ui_validate_phase1_to_phase2(
    char_state, genre, art_style, name, title, appearance, gear, z_prompt, user_concept, api_key, model
):
    """
    Validates Phase 1 character details and ensures Phase 2 is populated with smart,
    weapon-aware moves and pre-animation instructions before switching to Phase 2.
    """
    char_state = char_state or {}
    character_data = dict(char_state.get("character") or {})

    prev_gear = character_data.get("signature_gear", "")
    prev_appearance = character_data.get("appearance", "")

    if name and str(name).strip():
        character_data["name"] = str(name).strip()
    if title and str(title).strip():
        character_data["title"] = str(title).strip()
    if appearance and str(appearance).strip():
        character_data["appearance"] = str(appearance).strip()
    elif user_concept and str(user_concept).strip() and not character_data.get("appearance"):
        character_data["appearance"] = str(user_concept).strip()
    if gear and str(gear).strip():
        character_data["signature_gear"] = str(gear).strip()
    if z_prompt and str(z_prompt).strip():
        character_data["z_image_prompt"] = str(z_prompt).strip()
    if user_concept and str(user_concept).strip():
        character_data["user_concept"] = str(user_concept).strip()
    character_data["genre"] = genre
    character_data["art_style"] = art_style

    moves = character_data.get("moves") or []
    gear_changed = (character_data.get("signature_gear", "") != prev_gear) or (character_data.get("appearance", "") != prev_appearance)

    if not moves or gear_changed:
        pipeline = SpritesPipeline(api_key=api_key, model=model)
        moves = pipeline.generate_smart_moveset_with_gemma(
            character_data=character_data,
            genre=genre,
            use_llm=bool(api_key),
        )
        character_data["moves"] = moves

    arch = SpritesPipeline.detect_character_combat_archetype(character_data, genre=genre)
    character_data["combat_archetype"] = arch
    char_state["character"] = character_data

    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    first_choice = move_choices[0] if move_choices else None
    first_move = moves[0] if moves else {}
    first_type_label = (
        "🔁 Seamless Looping Cycle" if first_move.get("type") == "loop" else "🎯 Action (Return to Idle Stance)"
    )
    combat_directive_default = f"{arch['role_title']} wielding {arch['weapon_label']} — {arch['combat_style_summary']}"

    status = (
        f"### ⚔️ Phase 2 Ready: Smart Choreography Loaded for `{character_data.get('name', 'Hero')}`\n"
        f"- **Detected Weapon / Gear:** `{arch['weapon_label']}` (`{arch['role_title']}`)\n"
        f"- **Pre-Animation Instructions:** `{len(moves)}` moves tailored to `{arch['weapon_label']}` (no mismatched animations)."
    )
    return (
        gr.update(selected="sprites_phase2"),
        gr.update(choices=move_choices, value=first_choice),
        gr.update(choices=move_choices, value=first_choice),
        gr.update(choices=move_choices, value=first_choice),
        moves,
        combat_directive_default,
        first_move.get("name", ""),
        first_type_label,
        first_move.get("description", ""),
        first_move.get("refined_prompt", ""),
        status,
        char_state,
    )


def sprites_ui_refine_custom_anim(char_state, genre, custom_name, custom_desc, custom_type, api_key, model):
    character_data = (char_state or {}).get("character", {})
    if not character_data:
        return "", f"### ⚠️ Please generate a master character sprite first in Phase 1."
    anim_type = "loop" if "loop" in str(custom_type).lower() else "action"
    pipeline = SpritesPipeline(api_key=api_key, model=model)
    try:
        plan = pipeline.refine_custom_animation_with_gemma(
            character_data=character_data,
            genre=genre,
            anim_name=custom_name or "Custom Move",
            raw_description=custom_desc or "Dynamic movement",
            anim_type=anim_type,
        )
        status = f"### ✨ Gemma Directed Move: `{custom_name}` ({anim_type.upper()})\n{plan.get('action_summary')}"
        return plan.get("wan_prompt", ""), status
    except Exception as e:
        return "", f"### ⚠️ Gemma Move Refine Error\n`{e}`"


def sprites_ui_add_custom_anim(char_state, custom_name, custom_desc, custom_type, refined_prompt):
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    if not character_data:
        return gr.update(), gr.update(), gr.update(), [], "### ⚠️ Please generate a master character sprite first in Phase 1.", char_state

    moves = list(character_data.get("moves", []))
    anim_type = "loop" if "loop" in str(custom_type).lower() else "action"
    new_id = f"custom_move_{len(moves) + 1}"
    name_str = custom_name.strip() if custom_name and custom_name.strip() else f"Custom Move {len(moves)+1}"
    genre = character_data.get("genre", "Custom / Hybrid Genre")
    desc_str = (custom_desc or "").strip()
    prompt_str = (refined_prompt or "").strip()
    if not prompt_str:
        prompt_str = SpritesPipeline.build_pre_animation_wan_prompt(
            character_data=character_data,
            genre=genre,
            move_name=name_str,
            move_type=anim_type,
            move_description=desc_str or name_str,
        )
    new_move = {
        "id": new_id,
        "name": name_str,
        "type": anim_type,
        "description": desc_str,
        "refined_prompt": prompt_str,
        "wan_seed": 42 + len(moves) * 1007,
        "video_path": None,
        "sheet_path": None,
        "json_path": None,
    }
    moves.append(new_move)
    character_data["moves"] = moves
    char_state["character"] = character_data

    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    chosen = move_choices[-1]
    status = f"### ➕ Added Custom Move: `{name_str}` ({anim_type.upper()}) to move-set (Total moves: {len(moves)})."
    return (
        gr.update(choices=move_choices, value=chosen),
        gr.update(choices=move_choices, value=chosen),
        gr.update(choices=move_choices, value=chosen),
        moves,
        status,
        char_state,
    )


def sprites_ui_load_move_details(choice_label, char_state):
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    moves = character_data.get("moves", [])
    if not moves:
        return "No moves found. Please generate a character in Phase 1.", None

    target_move = moves[0]
    if choice_label:
        for m in moves:
            label = f"{m['name']} ({m['type'].upper()})"
            if choice_label == label or m["name"] in str(choice_label):
                target_move = m
                break

    v_path = target_move.get("video_path")
    rule = "🔁 Infinite Seamless Loop" if target_move.get("type") == "loop" else "🎯 Action & Combo (Returns to Idle Posture @ Final Frame)"
    info = (
        f"### 🎯 Selected Move: `{target_move['name']}` ({target_move.get('type', 'action').upper()})\n"
        f"- **Motion Rule:** {rule}\n"
        f"- **Choreography:** {target_move.get('description', '')}\n"
        f"- **Wan 2.2 Prompt:** `{target_move.get('refined_prompt', '')}`\n"
        f"- **Animation Status:** {'✅ Rendered (' + os.path.basename(v_path) + ')' if (v_path and os.path.exists(v_path)) else '⏳ Not yet animated (Click Animate below)'}"
    )
    return info, v_path if (v_path and os.path.exists(v_path)) else None


def sprites_ui_transition_phase2_to_phase3(char_state, current_choice):
    """Smoothly transitions to Phase 3, ensuring dropdown and info are populated."""
    char_state = char_state or {}
    character_data = char_state.get("character") or {}
    moves = character_data.get("moves") or []
    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    selected_val = current_choice if (current_choice in move_choices) else (move_choices[0] if move_choices else None)
    info, vpath = sprites_ui_load_move_details(selected_val, char_state)
    return (
        gr.update(selected="sprites_phase3"),
        gr.update(choices=move_choices, value=selected_val),
        info,
        vpath,
    )


def sprites_ui_transition_phase3_to_phase4(char_state, current_choice):
    """Smoothly transitions to Phase 4, ensuring cut dropdown is populated."""
    char_state = char_state or {}
    character_data = char_state.get("character") or {}
    moves = character_data.get("moves") or []
    move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
    selected_val = current_choice if (current_choice in move_choices) else (move_choices[0] if move_choices else None)
    return (
        gr.update(selected="sprites_phase4"),
        gr.update(choices=move_choices, value=selected_val),
    )


def sprites_ui_animate_move(choice_label, anim_seed, char_state, progress=gr.Progress(track_tqdm=True)):
    char_state = char_state or {}
    character_data = char_state.get("character") or {}
    moves = character_data.get("moves") or []

    # Auto-heal: if character_data exists but has no moves, generate contextual moves
    if character_data and not moves:
        genre = character_data.get("genre", "Action Runner (Pepsiman / Crash Bandicoot)")
        pipeline = SpritesPipeline()
        moves = pipeline.get_smart_contextual_moves(character_data, genre=genre)
        character_data["moves"] = moves
        char_state["character"] = character_data

    if not character_data or not moves:
        msg = "### ⚠️ Please complete Phase 1 and Phase 2 first.\nEnsure you have configured a character concept and move-set."
        return None, gr.update(), msg, msg, char_state

    # Resolve selected move index
    idx = 0
    if choice_label:
        for i, m in enumerate(moves):
            label = f"{m['name']} ({m['type'].upper()})"
            if choice_label == label or m["name"] in str(choice_label):
                idx = i
                break
    target_move = moves[idx]
    actual_label = f"{target_move['name']} ({target_move['type'].upper()})"

    pipeline = SpritesPipeline()
    try:
        def cb(pct, msg):
            try:
                progress(pct, desc=msg)
            except Exception:
                pass

        seed_val = int(anim_seed) if (anim_seed is not None and str(anim_seed).strip() != "") else 42
        video_path = pipeline.synthesize_animation_clip(
            character_data=character_data,
            move_data=target_move,
            wan_seed=seed_val,
            progress_cb=cb,
        )
        target_move["video_path"] = video_path
        moves[idx] = target_move
        character_data["moves"] = moves
        char_state["character"] = character_data

        move_choices = [f"{m['name']} ({m['type'].upper()})" for m in moves]
        mode_str = "🔁 Seamless Infinite Loop" if target_move["type"] == "loop" else "🎯 Return-to-Idle Recovery (FLF2V Anchor)"
        info = (
            f"### 🎬 Synthesized Move Animation: `{target_move['name']}` ({target_move['type'].upper()})\n"
            f"- **Motion Mode:** `{mode_str}` | **Wan Seed:** `{seed_val}`\n"
            f"- **Saved Video:** `{pathlib.Path(video_path).name}`\n"
            f"- **Choreography:** {target_move.get('description', '')}\n"
            f"- **Ready for Export:** Proceed to **Phase 4** to cut into transparent sprite sheet!"
        )
        return (
            video_path,
            gr.update(choices=move_choices, value=actual_label),
            info,
            info,
            char_state,
        )
    except Exception as e:
        err_msg = f"### ❌ Animation Synthesis Error\n`{e}`"
        return None, gr.update(), err_msg, err_msg, char_state


def sprites_ui_cut_sprite_sheet(choice_label, target_fps, chroma_tol, char_state):
    char_state = char_state or {}
    character_data = char_state.get("character", {})
    moves = character_data.get("moves", [])
    if not character_data or not moves:
        return None, {}, None, None, "### ⚠️ Please animate a move in Phase 3 first.", char_state

    idx = 0
    for i, m in enumerate(moves):
        label = f"{m['name']} ({m['type'].upper()})"
        if choice_label == label or m["name"] in str(choice_label):
            idx = i
            break
    target_move = moves[idx]
    video_path = target_move.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return None, {}, None, None, f"### ⚠️ Please animate `{target_move['name']}` in Phase 3 first.", char_state

    pipeline = SpritesPipeline()
    try:
        res = pipeline.cut_and_export_sprite_sheet(
            character_data=character_data,
            move_data=target_move,
            target_fps=int(target_fps),
            chroma_tolerance=float(chroma_tol),
        )
        moves[idx] = target_move
        character_data["moves"] = moves
        char_state["character"] = character_data

        status = (
            f"### ✂️ Sprite Sheet Packed: `{target_move['name']}` ({res['frame_count']} frames @ {target_fps} FPS)\n"
            f"- **PNG Atlas:** `{pathlib.Path(res['sheet_path']).name}`\n"
            f"- **Engine Manifest:** `{pathlib.Path(res['json_path']).name}`"
        )
        return res["sheet_image"], res["metadata"], res["sheet_path"], res["json_path"], status, char_state
    except Exception as e:
        return None, {}, None, None, f"### ❌ Sprite Cutter Error\n`{e}`", char_state


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Wan 2.2 & Qwen AI Filmmaking Studio") as demo:

    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 12px; padding-top: 6px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 20px; padding: 4px 14px; margin-bottom: 8px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
                <span style="color: #38bdf8; font-size: 0.82em; font-weight: 700; letter-spacing: 0.05em;">SERVERLESS ON-DEMAND GPU CLUSTER · ZERO IDLE COST</span>
            </div>
            <h1 class="main-title">🎬 Wan 2.2 Generative AI Studio</h1>
            <p class="subtitle">Next-Gen Multi-Modal Production Suite · Screenplay to Final Movie · 2D Game Sprites · Qwen Edit · Wan 2.2 I2V</p>
        </div>
        """
    )

    with gr.Row():
        z_image_hb_btn = gr.Button("💤 Z-Image Turbo Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        qwen_hb_btn = gr.Button("💤 Qwen Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        wan_hb_btn = gr.Button("💤 Wan 2.2 Keep-Warm: Paused ($0 Idle)", variant="secondary", scale=1)
        clear_outputs_btn = gr.Button("🗑️ Clear All Outputs", variant="stop", scale=1)

    with gr.Row(elem_classes="toolbar-row"):
        nav_studio_btn = gr.Button("🎬 Story Studio", variant="primary", size="sm", scale=1)
        nav_sprites_btn = gr.Button("🎮 2D Sprites Engine", variant="secondary", size="sm", scale=1)
        nav_qwen_btn = gr.Button("🖼️ Image Generator & Editor", variant="secondary", size="sm", scale=1)
        nav_wan_btn = gr.Button("🎥 Cinematic Video Studio", variant="secondary", size="sm", scale=1)
        expand_all_btn = gr.Button("🔽 Expand All Tools", variant="secondary", size="sm", visible=False)
        collapse_all_btn = gr.Button("🔼 Collapse All Tools", variant="secondary", size="sm", visible=False)

    with gr.Tabs(selected="tab_studio", elem_classes="main-workspace-tabs") as main_workspace_tabs:
        # ── Tool 1: AI Story-to-Movie Studio (Flagship 4-Phase Wizard) ─────────
        with gr.TabItem("🎬 AI Story-to-Movie Studio", id="tab_studio") as tab_studio:
            with gr.Group(elem_classes="studio-panel") as studio_accordion:
                gr.HTML(
                    """
                    <div class="pipeline-progression-rail">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.05em;">🎬 5-STEP MOVIE PIPELINE:</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 0.88em; flex-wrap: wrap;">
                            <span style="color: #38bdf8; background: rgba(56, 189, 248, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.3);">1️⃣ Screenplay</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #a855f7; background: rgba(168, 85, 247, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(168, 85, 247, 0.3);">2️⃣ Character Cast</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #ec4899; background: rgba(236, 72, 153, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(236, 72, 153, 0.3);">3️⃣ Dual Keyframes</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">4️⃣ Wan 2.2 Video</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #f59e0b; background: rgba(245, 158, 11, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3);">5️⃣ Film Stitch</span>
                        </div>
                    </div>
                    """
                )
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
                                        studio_ollama_model = gr.Dropdown(
                                            label="🧠 Ollama Cloud Model",
                                            choices=VERIFIED_OLLAMA_MODELS,
                                            value=DEFAULT_OLLAMA_MODEL,
                                            allow_custom_value=True,
                                            scale=2,
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
                                "🎨 Generate All Characters (9:16 Full-Body · Locked Z-Image Seed)",
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
                                    label="👤 Generated Character Reference Sheets (9:16 Vertical Head-to-Toe · Click to inspect/redo)",
                                    columns=[2],
                                    height=460,
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
                                    label="Selected Character Portrait (9:16 Full-Body)",
                                    height=320,
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
                                        "🔄 Redo This Character Image (9:16 Z-Image Turbo)",
                                        variant="primary",
                                    )

                    # ══════════════════════════════════════════════════════════════════
                    # PHASE 3: SCENE / SCENARIO IMAGES (QWEN MULTI-REF — FIRST + LAST FRAME)
                    # ══════════════════════════════════════════════════════════════════
                    with gr.TabItem("3️⃣ Phase 3: Scene / Scenario Images (Qwen Dual-Keyframe: First + Last Frame)", id="phase3") as phase3_tab:
                        with gr.Row():
                            studio_step3_single_btn = gr.Button(
                                "🎯 Generate Selected Scene (Scene-by-Scene Mode)",
                                variant="primary",
                                size="lg",
                                scale=2,
                            )
                            studio_step3_btn = gr.Button(
                                "⚡ Generate ALL Scenes at Once (Batch Mode)",
                                variant="secondary",
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
                                        "🎨 Generate / Redo Selected Scene Frame(s)",
                                        variant="primary",
                                    )
                                    studio_scene_next_btn = gr.Button(
                                        "⏭️ Next Scene ➔",
                                        variant="secondary",
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

        # ── Tool 2: Universal 2D Game Sprites Creator Studio ──────────────────
        with gr.TabItem("🎮 2D Game Sprites Engine", id="tab_sprites") as tab_sprites:
            with gr.Group(elem_classes="sprites-panel") as sprites_accordion:
                gr.HTML(
                    """
                    <div class="pipeline-progression-rail">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="background: linear-gradient(135deg, #10b981, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.05em;">🎮 2D SPRITE PIPELINE:</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 0.88em; flex-wrap: wrap;">
                            <span style="color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">1️⃣ Concept Genesis</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #06b6d4; background: rgba(6, 182, 212, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.3);">2️⃣ Move Choreography</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #8b5cf6; background: rgba(139, 92, 246, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3);">3️⃣ Wan Animation</span>
                            <span style="color: #64748b;">➔</span>
                            <span style="color: #f59e0b; background: rgba(245, 158, 11, 0.15); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(245, 158, 11, 0.3);">4️⃣ Transparent Atlas Export</span>
                        </div>
                    </div>
                    """
                )
                gr.Markdown(
                    "### 🎮 Universal 2D Game Sprites Creator · Gemma 4: 31B ➔ Z-Image Turbo ➔ Wan 2.2 ➔ Transparent Sprite Sheet",
                    elem_classes="section-header",
                )

                sprites_state_store = gr.State({})

                with gr.Group():
                    with gr.Row():
                        sprites_genre = gr.Dropdown(
                            label="🕹️ Game Genre / Perspective Archetype",
                            choices=list(GENRE_PRESETS.keys()),
                            value="Action Runner (Pepsiman / Crash Bandicoot)",
                            scale=2,
                        )
                        sprites_art_style = gr.Dropdown(
                            label="🎨 Art Aesthetic Style",
                            choices=[
                                "16-bit arcade pixel art style",
                                "Hand-drawn 2D anime cel style",
                                "2.5D stylized rendered sprite",
                                "Classic retro 8-bit NES style",
                                "High-res modern vector arcade style",
                            ],
                            value="16-bit arcade pixel art style",
                            scale=1,
                        )
                        sprites_char_seed = gr.Number(
                            value=42, precision=0, label="🎲 Character Seed", scale=1
                        )
                        sprites_anim_seed = gr.Number(
                            value=8888, precision=0, label="🎲 Animation Seed", scale=1
                        )

                    with gr.Accordion("🔑 Gemma 4: 31B Cloud API Settings", open=False):
                        with gr.Row():
                            sprites_ollama_key = gr.Textbox(
                                label="Ollama API Key",
                                value=DEFAULT_OLLAMA_API_KEY,
                                type="password",
                                scale=3,
                            )
                            sprites_ollama_model = gr.Dropdown(
                                label="🧠 Ollama Cloud Model",
                                choices=VERIFIED_OLLAMA_MODELS,
                                value=DEFAULT_OLLAMA_MODEL,
                                allow_custom_value=True,
                                scale=2,
                            )

                sprites_status = gr.Markdown("### 🕹️ Ready. Enter a character concept below to begin Phase 1.")

                with gr.Tabs() as sprites_wizard_tabs:
                    # ─────────────────────────────────────────────────────────────
                    # Phase 1: Character Genesis
                    # ─────────────────────────────────────────────────────────────
                    with gr.TabItem("1️⃣ Phase 1: Character Genesis", id="sprites_phase1"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sprites_user_concept = gr.Textbox(
                                    label="Character Concept / Details",
                                    placeholder="e.g. Silver metallic runner with blue logo on chest, athletic build, running sneakers, energetic hero personality",
                                    lines=3,
                                )
                                with gr.Row():
                                    sprites_refine_char_btn = gr.Button("✨ Refine Character with Gemma 4: 31B", variant="secondary")
                                    sprites_gen_char_btn = gr.Button("🎨 Generate Master 2D Sprite", variant="primary")

                                with gr.Row():
                                    sprites_char_name = gr.Textbox(label="Character Name", value="Pepsi_Hero", scale=1)
                                    sprites_char_title = gr.Textbox(label="Title / Archetype", value="Endless Obstacle Runner", scale=1)
                                sprites_char_appearance = gr.Textbox(label="Visual Description & Outfit", lines=2)
                                sprites_char_gear = gr.Textbox(label="Signature Gear / Item", lines=1)
                                sprites_char_prompt = gr.Textbox(label="Z-Image Prompt (Chroma Key Green #00FF00)", lines=2)

                            with gr.Column(scale=1):
                                sprites_char_preview = gr.Image(
                                    label="Master 2D Sprite Preview (Isolated on Solid Chroma Green)",
                                    height=380,
                                    interactive=False,
                                )
                                sprites_phase1_next_btn = gr.Button(
                                    "✅ Validate Character & Proceed to Move Choreography ➔",
                                    variant="primary",
                                    size="lg",
                                )

                    # ─────────────────────────────────────────────────────────────
                    # Phase 2: Move Choreography & Custom Animation Creator
                    # ─────────────────────────────────────────────────────────────
                    with gr.TabItem("2️⃣ Phase 2: Move Choreography & Custom Creator", id="sprites_phase2"):
                        with gr.Group(elem_classes="director-panel"):
                            gr.Markdown(
                                "#### 🧠 Smart AI Character & Weapon Choreographer (Gemma 4: 31B)\n"
                                "Automatically designs moves and pre-animation instructions tailored to your character's equipped weapon, gear, and silhouette "
                                "(e.g., sword slashes & blade guard stances for a swordsman — never mismatched boxing or arm-cannon moves).",
                            )
                            with gr.Row():
                                sprites_combat_notes = gr.Textbox(
                                    label="⚔️ Character Weapon & Combat Style Directives (Auto-Detected or Custom)",
                                    placeholder="e.g. Wields a glowing katana sword — use two-handed sword guard, swift blade dashes, and multi-hit slash combos (no unarmed boxing)...",
                                    scale=3,
                                )
                                sprites_smart_choreograph_btn = gr.Button(
                                    "🧠 Smart Choreograph All Moves with Gemma 4: 31B",
                                    variant="primary",
                                    scale=1,
                                )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### 📋 Active Character Move-Set & Pre-Animation Editor")
                                sprites_move_selector = gr.Dropdown(
                                    label="Select Move to Inspect / Edit Pre-Animation Instructions",
                                    choices=[],
                                    interactive=True,
                                )
                                with gr.Row():
                                    sprites_edit_move_name = gr.Textbox(
                                        label="Move Name",
                                        placeholder="e.g. Katana Guard Idle, 3-Hit Sword Combo...",
                                        scale=1,
                                    )
                                    sprites_edit_move_type = gr.Dropdown(
                                        label="Motion Rule",
                                        choices=["🎯 Action (Return to Idle Stance)", "🔁 Seamless Looping Cycle"],
                                        value="🔁 Seamless Looping Cycle",
                                        scale=1,
                                    )
                                sprites_edit_move_desc = gr.Textbox(
                                    label="Pre-Animation Choreography Instructions (How Character & Weapon Move)",
                                    lines=2,
                                    placeholder="Detailed limb and weapon movement instructions...",
                                )
                                sprites_edit_move_prompt = gr.Textbox(
                                    label="Compiled Wan 2.2 Pre-Animation Prompt (Weapon & Pose Anchored)",
                                    lines=3,
                                    placeholder="Temporally-staged Wan 2.2 prompt with character, weapon, and chroma-key locks...",
                                )
                                with gr.Row():
                                    sprites_rechoreograph_move_btn = gr.Button(
                                        "✨ Re-Direct Selected Move with Gemma",
                                        variant="secondary",
                                    )
                                    sprites_save_move_btn = gr.Button(
                                        "💾 Save Move Changes",
                                        variant="primary",
                                    )
                                with gr.Accordion("📚 Full Move-Set Registry (JSON)", open=False):
                                    sprites_moves_json = gr.JSON(label="Move-Set Registry")

                            with gr.Column(scale=1):
                                gr.Markdown("#### ➕ Add Custom Animation (Weapon & Character Aware)")
                                sprites_custom_name = gr.Textbox(
                                    label="Custom Animation Name",
                                    placeholder="e.g. Crescent Blade Wave, Shield Bash, Arcane Nova, Double Backflip...",
                                )
                                sprites_custom_desc = gr.Textbox(
                                    label="Movement & Behavior Description",
                                    placeholder="Describe what the character does with their weapon/body in plain natural language...",
                                    lines=2,
                                )
                                sprites_custom_type = gr.Radio(
                                    label="Animation Motion Rule",
                                    choices=["🎯 Action (Return to Idle Stance)", "🔁 Seamless Looping Cycle"],
                                    value="🎯 Action (Return to Idle Stance)",
                                )
                                sprites_custom_refined_prompt = gr.Textbox(
                                    label="Refined Wan Choreography Prompt",
                                    lines=3,
                                    placeholder="Click 'Direct & Refine with Gemma' to compile weapon-aware micro-choreography prompt...",
                                )
                                with gr.Row():
                                    sprites_refine_custom_btn = gr.Button("✨ Direct & Refine with Gemma", variant="secondary")
                                    sprites_add_custom_btn = gr.Button("➕ Add to Character's Move-Set", variant="primary")

                        with gr.Row():
                            sprites_phase2_back_btn = gr.Button("⬅️ Back to Phase 1", variant="secondary")
                            sprites_phase2_next_btn = gr.Button("✅ Proceed to Phase 3: Wan 2.2 Animation Studio ➔", variant="primary")

                    # ─────────────────────────────────────────────────────────────
                    # Phase 3: Wan 2.2 Animation Studio
                    # ─────────────────────────────────────────────────────────────
                    with gr.TabItem("3️⃣ Phase 3: Wan 2.2 Animation Studio", id="sprites_phase3"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sprites_anim_target_move = gr.Dropdown(
                                    label="Select Move to Animate with Wan 2.2",
                                    choices=[],
                                    interactive=True,
                                )
                                with gr.Row():
                                    sprites_animate_single_btn = gr.Button("🎬 Animate Selected Move (Wan 2.2)", variant="primary", size="lg")
                                sprites_anim_info = gr.Markdown("Select a move and click animate.")

                            with gr.Column(scale=1):
                                sprites_anim_video_preview = gr.Video(
                                    label="Synthesized Move Animation Clip (Looping MP4)",
                                    height=360,
                                    autoplay=True,
                                    loop=True,
                                )
                        with gr.Row():
                            sprites_phase3_back_btn = gr.Button("⬅️ Back to Phase 2", variant="secondary")
                            sprites_phase3_next_btn = gr.Button("✅ Proceed to Phase 4: Sprite Cutter & Sheet Exporter ➔", variant="primary")

                    # ─────────────────────────────────────────────────────────────
                    # Phase 4: Sprite Cutter & Sheet Exporter
                    # ─────────────────────────────────────────────────────────────
                    with gr.TabItem("4️⃣ Phase 4: Sprite Cutter & Sheet Exporter", id="sprites_phase4"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sprites_cut_target_move = gr.Dropdown(
                                    label="Select Move Animation to Slice into Sprite Sheet",
                                    choices=[],
                                    interactive=True,
                                )
                                with gr.Row():
                                    sprites_target_fps = gr.Dropdown(
                                        label="Target Game Frame Rate",
                                        choices=[8, 12, 16, 24],
                                        value=12,
                                        scale=1,
                                    )
                                    sprites_chroma_tol = gr.Slider(
                                        label="Chroma Removal Tolerance",
                                        minimum=15.0,
                                        maximum=70.0,
                                        value=38.0,
                                        step=1.0,
                                        scale=1,
                                    )
                                sprites_cut_sheet_btn = gr.Button("✂️ Cut Video to Transparent Sprite Sheet & JSON", variant="primary", size="lg")
                                with gr.Row():
                                    sprites_download_sheet = gr.File(label="💾 Download Sprite Sheet (.png)")
                                    sprites_download_json = gr.File(label="📄 Download Engine Manifest (.json)")

                            with gr.Column(scale=1):
                                sprites_sheet_preview = gr.Image(
                                    label="Packed Transparent Sprite Sheet Atlas (PNG)",
                                    height=380,
                                )
                                sprites_json_preview = gr.JSON(label="Generated Engine Metadata (Unity / Godot / Unreal)")
                        with gr.Row():
                            sprites_phase4_back_btn = gr.Button("⬅️ Back to Phase 3", variant="secondary")

        # ── Tool 3: Qwen Image Edit Rapid AIO Generator / Editor (Standalone) ─
        with gr.TabItem("🖼️ Image Generator & Editor", id="tab_qwen") as tab_qwen:
            with gr.Group(elem_classes="qwen-panel") as qwen_accordion:
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

        # ── Tool 4: Wan 2.2 Video Generator (Standalone) ──────────────────────
        with gr.TabItem("🎥 Cinematic Video Studio", id="tab_wan") as tab_wan:
            with gr.Group(elem_classes="wan-panel") as wan_accordion:
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

    # ── Global Navigation Toolbar Events ─────────────────────────────────────
    nav_studio_btn.click(fn=lambda: gr.update(selected="tab_studio"), inputs=[], outputs=[main_workspace_tabs])
    nav_sprites_btn.click(fn=lambda: gr.update(selected="tab_sprites"), inputs=[], outputs=[main_workspace_tabs])
    nav_qwen_btn.click(fn=lambda: gr.update(selected="tab_qwen"), inputs=[], outputs=[main_workspace_tabs])
    nav_wan_btn.click(fn=lambda: gr.update(selected="tab_wan"), inputs=[], outputs=[main_workspace_tabs])

    expand_all_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        inputs=[],
        outputs=[sprites_accordion, qwen_accordion, wan_accordion, studio_accordion],
    )

    collapse_all_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        inputs=[],
        outputs=[sprites_accordion, qwen_accordion, wan_accordion, studio_accordion],
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

    for single_scene_btn in (studio_step3_single_btn, studio_scene_redo_btn):
        single_scene_btn.click(
            fn=studio_redo_single_scene,
            inputs=[
                studio_scene_selector, studio_scene_title_edit, studio_scene_scenario_edit,
                studio_scene_prompt_edit, studio_scene_chars_edit, studio_scene_seed_edit,
                studio_aspect_ratio, studio_ollama_key, studio_ollama_model, studio_pipeline_state,
                studio_scene_end_prompt_edit, studio_scene_redo_mode, studio_scene_qa_chk,
                studio_scene_end_seed_edit, studio_scene_direct_end_chk, studio_scene_chain_chk,
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

    studio_scene_next_btn.click(
        fn=studio_step_to_next_scene,
        inputs=[studio_scene_selector, studio_pipeline_state, studio_qwen_seed],
        outputs=[
            studio_scene_selector,
            studio_scene_title_edit, studio_scene_scenario_edit, studio_scene_prompt_edit,
            studio_scene_chars_edit, studio_scene_seed_edit, studio_scene_preview,
            studio_scene_end_prompt_edit, studio_scene_end_preview, studio_scene_end_seed_edit,
            studio_scene_delta_badge, studio_status,
        ],
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

    # ── Events: Universal 2D Game Sprites Creator Studio ─────────────────────

    # Phase 1: Refine Character & Generate Master Sprite
    sprites_refine_char_btn.click(
        fn=sprites_ui_refine_character,
        inputs=[
            sprites_user_concept, sprites_genre, sprites_art_style,
            sprites_ollama_key, sprites_ollama_model,
        ],
        outputs=[
            sprites_char_name, sprites_char_title, sprites_char_appearance,
            sprites_char_gear, sprites_char_prompt, sprites_status,
        ],
    )

    sprites_gen_char_btn.click(
        fn=sprites_ui_generate_master_sprite,
        inputs=[
            sprites_char_name, sprites_char_title, sprites_char_appearance,
            sprites_char_gear, sprites_char_prompt, sprites_char_seed,
            sprites_genre, sprites_art_style,
            sprites_ollama_key, sprites_ollama_model, sprites_state_store,
            sprites_user_concept,
        ],
        outputs=[
            sprites_char_preview,
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_status,
            sprites_state_store,
        ],
    )

    # Phase 2: Smart AI Character & Weapon Choreographer (Full Move-Set via Gemma)
    sprites_smart_choreograph_btn.click(
        fn=sprites_ui_smart_choreograph_moveset,
        inputs=[
            sprites_state_store, sprites_genre, sprites_combat_notes,
            sprites_char_name, sprites_char_title, sprites_char_appearance,
            sprites_char_gear, sprites_ollama_key, sprites_ollama_model,
        ],
        outputs=[
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_edit_move_name,
            sprites_edit_move_type,
            sprites_edit_move_desc,
            sprites_edit_move_prompt,
            sprites_status,
            sprites_state_store,
        ],
    )

    # Move Inspector: Populate Phase 2 Pre-Animation Editor & Phase 3 Details when a move is selected
    sprites_move_selector.change(
        fn=sprites_ui_select_move_for_edit,
        inputs=[sprites_move_selector, sprites_state_store],
        outputs=[
            sprites_edit_move_name,
            sprites_edit_move_type,
            sprites_edit_move_desc,
            sprites_edit_move_prompt,
            sprites_anim_info,
            sprites_anim_video_preview,
        ],
    )

    # Phase 2: Re-Direct Selected Move with Gemma & Save Manual Edits
    sprites_rechoreograph_move_btn.click(
        fn=sprites_ui_rechoreograph_selected_move,
        inputs=[
            sprites_move_selector,
            sprites_edit_move_name,
            sprites_edit_move_type,
            sprites_edit_move_desc,
            sprites_genre,
            sprites_ollama_key,
            sprites_ollama_model,
            sprites_state_store,
        ],
        outputs=[
            sprites_edit_move_desc,
            sprites_edit_move_prompt,
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_status,
            sprites_state_store,
        ],
    )

    sprites_save_move_btn.click(
        fn=sprites_ui_save_selected_move,
        inputs=[
            sprites_move_selector,
            sprites_edit_move_name,
            sprites_edit_move_type,
            sprites_edit_move_desc,
            sprites_edit_move_prompt,
            sprites_genre,
            sprites_state_store,
        ],
        outputs=[
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_status,
            sprites_state_store,
        ],
    )

    sprites_anim_target_move.change(
        fn=sprites_ui_load_move_details,
        inputs=[sprites_anim_target_move, sprites_state_store],
        outputs=[sprites_anim_info, sprites_anim_video_preview],
    )

    # Phase 2: Refine Custom Move & Add Custom Move
    sprites_refine_custom_btn.click(
        fn=sprites_ui_refine_custom_anim,
        inputs=[
            sprites_state_store, sprites_genre,
            sprites_custom_name, sprites_custom_desc, sprites_custom_type,
            sprites_ollama_key, sprites_ollama_model,
        ],
        outputs=[sprites_custom_refined_prompt, sprites_status],
    )

    sprites_add_custom_btn.click(
        fn=sprites_ui_add_custom_anim,
        inputs=[
            sprites_state_store, sprites_custom_name, sprites_custom_desc,
            sprites_custom_type, sprites_custom_refined_prompt,
        ],
        outputs=[
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_status,
            sprites_state_store,
        ],
    )

    # Phase 3: Animate Selected Move with Wan 2.2
    sprites_animate_single_btn.click(
        fn=sprites_ui_animate_move,
        inputs=[sprites_anim_target_move, sprites_anim_seed, sprites_state_store],
        outputs=[
            sprites_anim_video_preview,
            sprites_cut_target_move,
            sprites_anim_info,
            sprites_status,
            sprites_state_store,
        ],
    )

    # Phase 4: Cut Video to Transparent Sprite Sheet & Engine JSON
    sprites_cut_sheet_btn.click(
        fn=sprites_ui_cut_sprite_sheet,
        inputs=[
            sprites_cut_target_move, sprites_target_fps, sprites_chroma_tol,
            sprites_state_store,
        ],
        outputs=[
            sprites_sheet_preview,
            sprites_json_preview,
            sprites_download_sheet,
            sprites_download_json,
            sprites_status,
            sprites_state_store,
        ],
    )

    # Tab Wizard Navigation Buttons (Phase 1 -> Phase 2 validates & populates smart weapon choreography)
    sprites_phase1_next_btn.click(
        fn=sprites_ui_validate_phase1_to_phase2,
        inputs=[
            sprites_state_store, sprites_genre, sprites_art_style,
            sprites_char_name, sprites_char_title, sprites_char_appearance,
            sprites_char_gear, sprites_char_prompt, sprites_user_concept,
            sprites_ollama_key, sprites_ollama_model,
        ],
        outputs=[
            sprites_wizard_tabs,
            sprites_move_selector,
            sprites_anim_target_move,
            sprites_cut_target_move,
            sprites_moves_json,
            sprites_combat_notes,
            sprites_edit_move_name,
            sprites_edit_move_type,
            sprites_edit_move_desc,
            sprites_edit_move_prompt,
            sprites_status,
            sprites_state_store,
        ],
    )
    sprites_phase2_back_btn.click(
        fn=lambda: gr.update(selected="sprites_phase1"),
        inputs=[],
        outputs=[sprites_wizard_tabs],
    )
    sprites_phase2_next_btn.click(
        fn=sprites_ui_transition_phase2_to_phase3,
        inputs=[sprites_state_store, sprites_move_selector],
        outputs=[
            sprites_wizard_tabs,
            sprites_anim_target_move,
            sprites_anim_info,
            sprites_anim_video_preview,
        ],
    )
    sprites_phase3_back_btn.click(
        fn=lambda: gr.update(selected="sprites_phase2"),
        inputs=[],
        outputs=[sprites_wizard_tabs],
    )
    sprites_phase3_next_btn.click(
        fn=sprites_ui_transition_phase3_to_phase4,
        inputs=[sprites_state_store, sprites_anim_target_move],
        outputs=[
            sprites_wizard_tabs,
            sprites_cut_target_move,
        ],
    )
    sprites_phase4_back_btn.click(
        fn=lambda: gr.update(selected="sprites_phase3"),
        inputs=[],
        outputs=[sprites_wizard_tabs],
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

