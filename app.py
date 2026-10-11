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
    run_remote_inference, qwen_image_edit_remote,
    start_keepalive_thread, record_frontend_heartbeat, set_wan_heartbeat_state,
    set_qwen_heartbeat_state
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

#hidden-timestamp {
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
        "**Tool 2 🎥** — Wan 2.2 Video Generator",
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

                use_qwen_as_video_btn = gr.Button(
                    "🎥 Send to Wan 2.2 (Video Input)",
                    variant="secondary"
                )

    # ── Tool 2: Wan 2.2 Video Generator ───────────────────────────────────────
    with gr.Accordion("🎥 Tool 2 — Wan 2.2 · High-Motion Video Generator", open=True, elem_classes="tool-accordion wan-panel") as wan_accordion:
        gr.Markdown("### 🎥 Wan 2.2 Video Generator", elem_classes="section-header")
        gr.Markdown(
            "Upload or route an image from Tool 1 (Qwen Image Edit). Add a motion prompt and click **Generate Video**.\n\n"
            "💡 **Facial Consistency & Stability Tip:** To prevent the subject's face from altering after 3–4 seconds, "
            "keep duration within **3.0s – 5.0s**, leave Guidance Scale at **1.5 – 2.5** so the negative prompt is actively enforced, "
            "and include *'preserve facial features, consistent facial identity'* in your motion prompt.",
            elem_classes="step-hint"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image_component = gr.Image(
                    type="pil", label="Input Image for Video",
                    sources=["upload", "clipboard"], elem_id="wan-input-image"
                )
                prompt_input = gr.Textbox(label="Motion Prompt", value=default_prompt_i2v)
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
                    timestamp_box = gr.Number(value=0, label="Timestamp", visible=True, elem_id="hidden-timestamp")
                file_output = gr.File(label="Download Video")

    # ── Global Toolbar Events (Expand/Collapse All Tools) ─────────────────────
    expand_all_btn.click(
        fn=lambda: (gr.update(open=True), gr.update(open=True)),
        inputs=[],
        outputs=[qwen_accordion, wan_accordion],
    )

    collapse_all_btn.click(
        fn=lambda: (gr.update(open=False), gr.update(open=False)),
        inputs=[],
        outputs=[qwen_accordion, wan_accordion],
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

    # ── Events: Wan 2.2 Video ─────────────────────────────────────────────────

    ui_inputs = [
        input_image_component, last_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox,
        quality_slider, scheduler_dropdown, flow_shift_slider, frame_multi,
        safe_mode_checkbox, play_result_video,
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
