import os
import random
import pathlib
import time
from PIL import Image, ImageOps
import gradio as gr

import backend

# ─────────────────────────────────────────────────────────────────────────────
# Smart Studio Frontend
# Dedicated Wan 2.2 Video Studio + Hermes 3 Smart Prompt Refiner
# ─────────────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = pathlib.Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FIXED_FPS = 16
MIN_FRAMES_MODEL = 16
MAX_FRAMES_MODEL = 128
MIN_DIM = 480
MAX_DIM = 832
MULTIPLE_OF = 16

DEFAULT_WAN_PROMPT = (
    "A cinematic high-resolution video capturing smooth natural movement, "
    "realistic physics, beautiful ambient lighting, 4k masterwork, preserve face details"
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, worst quality, jpeg artifacts, compression artifacts, "
    "pixelated, out of focus, grainy, noisy, oversaturated, undersaturated, bad anatomy, deformed, "
    "disfigured, extra limbs, missing limbs, extra fingers, missing fingers, fused fingers, mutated hands, "
    "poorly drawn hands, poorly drawn face, malformed limbs, long neck, distorted proportions, "
    "asymmetrical eyes, cross-eyed, watermark, signature, text, logo, cropped, cut off, frame, "
    "border, duplicate, cloned face, mutation, ugly, tiling"
)


# ─── Resolution and Frame Utilities ──────────────────────────────────────────

def resize_image(image: Image.Image) -> Image.Image:
    """Resize image so dimensions are multiples of 16 within [480, 832] range."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    aspect_ratio = w / h

    if w >= h:
        new_w = min(MAX_DIM, max(MIN_DIM, w))
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = min(MAX_DIM, max(MIN_DIM, h))
        new_w = int(new_h * aspect_ratio)

    new_w = max(MIN_DIM, min(MAX_DIM, round(new_w / MULTIPLE_OF) * MULTIPLE_OF))
    new_h = max(MIN_DIM, min(MAX_DIM, round(new_h / MULTIPLE_OF) * MULTIPLE_OF))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def resize_and_crop_to_match(image: Image.Image, reference_image: Image.Image) -> Image.Image:
    """Matches the exact dimensions of reference_image using aspect-preserving crop."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    target_w, target_h = reference_image.size
    target_aspect = target_w / target_h

    w, h = image.size
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        image = image.crop((left, 0, left + new_w, h))
    elif current_aspect < target_aspect:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        image = image.crop((0, top, w, top + new_h))

    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def get_num_frames(duration_seconds: float) -> int:
    """Calculates frame count for Wan (1 + 16 * n)."""
    frames = int(round(float(duration_seconds) * FIXED_FPS))
    frames = max(MIN_FRAMES_MODEL, min(MAX_FRAMES_MODEL, frames))
    return 1 + frames


def extract_frame_from_video(video_path: str, timestamp_sec: float = 0.0):
    """Extracts a frame from a local video file at the specified timestamp."""
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or FIXED_FPS
        frame_idx = int(float(timestamp_sec) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
    except Exception as e:
        print(f"[Smart Studio] Frame extraction error: {e}")
    return None


def list_output_videos():
    """Finds all generated mp4 files sorted by modification time."""
    if not OUTPUTS_DIR.exists():
        return []
    videos = sorted(
        [str(p) for p in OUTPUTS_DIR.glob("*.mp4") if p.is_file()],
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    return videos


# ─── Hermes Prompt Refiner Handler ──────────────────────────────────────────

def on_refine_prompt_click(
    raw_prompt: str,
    style_preset: str,
    motion_dynamics: str,
    camera_movement: str,
    temperature: float,
    progress=gr.Progress(track_tqdm=True)
):
    """Calls Hermes LLM to enhance the prompt."""
    if not raw_prompt or not raw_prompt.strip():
        try:
            gr.Warning("Please enter a concept or prompt in the text box first.")
        except Exception:
            pass
        return "", "", "⚠️ Please enter a prompt."

    try:
        result = backend.refine_prompt_hermes(
            raw_prompt=raw_prompt,
            style_preset=style_preset,
            motion_dynamics=motion_dynamics,
            camera_movement=camera_movement,
            temperature=temperature,
            progress=progress
        )
        refined = result.get("refined_prompt", "")
        negative = result.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        elapsed = result.get("elapsed_seconds", 0.0)
        status = f"✓ Prompt refined by Hermes 3 in {elapsed}s ({style_preset})"
        try:
            gr.Info(status)
        except Exception:
            pass
        return refined, negative, status
    except Exception as e:
        try:
            gr.Error(f"Error refining prompt: {e}")
        except Exception:
            pass
        return raw_prompt, DEFAULT_NEGATIVE_PROMPT, f"❌ Error: {e}"


# ─── Wan Video Generation Handler ───────────────────────────────────────────

def on_generate_video_click(
    input_image,
    last_image,
    prompt: str,
    negative_prompt: str,
    duration: float,
    steps: int,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int,
    randomize_seed: bool,
    scheduler_name: str,
    flow_shift: float,
    frame_multiplier: int,
    progress=gr.Progress(track_tqdm=True)
):
    """Generates Wan 2.2 video on Modal."""
    if input_image is None:
        raise gr.Error("Please upload an input image to generate a video.")

    final_seed = random.randint(0, 2**31 - 1) if randomize_seed else int(seed)
    num_frames = get_num_frames(duration)

    print(f"\n[Smart Studio] New Generation Request: seed={final_seed}, duration={duration}s, steps={steps}")

    # Process and crop images
    processed_first = resize_image(input_image)
    processed_last = None
    if last_image is not None:
        processed_last = resize_and_crop_to_match(last_image, processed_first)

    # Remote inference
    video_path = backend.run_remote_inference(
        resized_image=processed_first,
        processed_last_image=processed_last,
        prompt=prompt or DEFAULT_WAN_PROMPT,
        steps=steps,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        current_seed=final_seed,
        scheduler_name=scheduler_name,
        flow_shift=flow_shift,
        frame_multiplier=frame_multiplier,
        quality=5,
        duration_seconds=duration,
        progress=progress
    )

    info_msg = (
        f"✓ Video generated successfully! (Seed: {final_seed}, "
        f"Resolution: {processed_first.width}x{processed_first.height}, Frames: {num_frames})"
    )
    try:
        gr.Info(info_msg)
    except Exception:
        pass

    updated_videos = list_output_videos()
    return video_path, video_path, final_seed, info_msg, updated_videos


# ─── Keepalive and Outputs Handlers ─────────────────────────────────────────

def on_toggle_heartbeat(current_state: bool):
    new_state = not current_state
    backend.set_smart_heartbeat_state(new_state)
    label = "⚡ Keep Backend Warm: ON" if new_state else "⚡ Keep Backend Warm: OFF"
    variant = "primary" if new_state else "secondary"
    return new_state, gr.update(value=label, variant=variant)


def on_clear_outputs():
    msg = backend.clear_all_outputs()
    try:
        gr.Info(msg)
    except Exception:
        pass
    return None, [], msg


# ─── UI Layout Definition ────────────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    padding: 20px 10px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(30,30,45,0.8), rgba(20,20,30,0.9));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}
.hermes-card {
    border: 1px solid rgba(138, 75, 255, 0.3) !important;
    background: rgba(138, 75, 255, 0.03) !important;
    border-radius: 10px !important;
}
.wan-card {
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    background: rgba(59, 130, 246, 0.03) !important;
    border-radius: 10px !important;
}
"""

def create_ui():
    with gr.Blocks(title="Smart Studio · Wan 2.2 & Hermes 3", css=CUSTOM_CSS, theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate")) as demo:
        heartbeat_state = gr.State(value=False)

        # Header
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🎬 Smart Studio · Wan 2.2 & Hermes 3
                ### Cinematic AI Video Creation with Intelligent LLM Prompt Refinement
                **Backend:** Modal.com Serverless GPU (A100-80GB) · **LLM:** `NousResearch/Hermes-3-Llama-3.1-8B`
                """
            )
            with gr.Row():
                hb_btn = gr.Button("⚡ Keep Backend Warm: OFF", variant="secondary", size="sm")
                clear_btn = gr.Button("🗑️ Clear Outputs", variant="secondary", size="sm")
                backend_status = gr.Markdown("*Status: Ready on Localhost*", elem_id="status-text")

        # ── Section 1: Hermes Smart Prompt Studio ──
        with gr.Accordion("🧠 Tool 1 — Hermes 3 Smart Prompt Generator (NousResearch)", open=True, elem_classes=["hermes-card"]):
            gr.Markdown(
                "Type your core idea or raw prompt below. Hermes 3 will automatically structure and expand it with "
                "cinematic lens properties, natural physics, fluid movement, and lighting schemes tailored for Wan 2.2."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    hermes_raw_prompt = gr.Textbox(
                        label="Raw Concept / Initial Idea",
                        placeholder="e.g., a cybernetic samurai walking through neo-Tokyo in the rain, cherry blossoms falling...",
                        lines=3,
                        value="camera capturing overall people in a complete full frame, 4k resolution, preserve face details"
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        style_preset = gr.Dropdown(
                            label="Cinematic Style Preset",
                            choices=[
                                "Cinematic Film (35mm)",
                                "Hyperrealistic 4K",
                                "Cyberpunk & Sci-Fi",
                                "Anime / Studio Ghibli",
                                "Dark Noir & Atmospheric",
                                "Fantasy & Mythical",
                                "Drone Aerial Nature",
                            ],
                            value="Cinematic Film (35mm)",
                            interactive=True
                        )
                    with gr.Row():
                        motion_dynamics = gr.Dropdown(
                            label="Motion Dynamics",
                            choices=[
                                "Smooth & Flowing Motion",
                                "Subtle & Atmospheric",
                                "High-Speed Dynamic Action",
                                "Epic Slow Motion Sweep",
                            ],
                            value="Smooth & Flowing Motion",
                            interactive=True
                        )
                    with gr.Row():
                        camera_movement = gr.Dropdown(
                            label="Camera Direction",
                            choices=[
                                "Auto Best Fit",
                                "Slow Push-In Tracking Shot",
                                "Dramatic 360 Orbit Pan",
                                "Low-Angle Hero Shot",
                                "Static Frame with Internal Motion",
                            ],
                            value="Auto Best Fit",
                            interactive=True
                        )

            with gr.Row():
                refine_btn = gr.Button("✨ Refine Prompt with Hermes 3", variant="primary", scale=2)
                hermes_temp = gr.Slider(label="Creativity (Temperature)", minimum=0.2, maximum=1.0, value=0.7, step=0.05, scale=1)

            with gr.Row():
                with gr.Column(scale=4):
                    hermes_refined_output = gr.Textbox(
                        label="Hermes Refined Cinematic Prompt",
                        placeholder="Click '✨ Refine Prompt with Hermes 3' above to generate an enhanced Wan prompt...",
                        lines=4,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    apply_prompt_btn = gr.Button("⬇️ Apply to Video Studio", variant="secondary")
                    hermes_status = gr.Markdown("")

        # ── Section 2: Wan 2.2 Video Generator ──
        with gr.Accordion("🎬 Tool 2 — Wan 2.2 Video Studio (A100 GPU)", open=True, elem_classes=["wan-card"]):
            with gr.Row():
                # Left column: Inputs
                with gr.Column(scale=5):
                    with gr.Row():
                        first_frame = gr.Image(label="First Frame (Input Image)", type="pil", height=280)
                        last_frame = gr.Image(label="Last Frame (Optional Transition)", type="pil", height=280)

                    video_prompt = gr.Textbox(
                        label="Video Generation Prompt",
                        placeholder="Prompt describing visual action, motion, and camera dynamics...",
                        lines=4,
                        value=DEFAULT_WAN_PROMPT
                    )

                    with gr.Accordion("Negative Prompt & Safety", open=False):
                        video_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=3,
                            value=DEFAULT_NEGATIVE_PROMPT
                        )

                    with gr.Row():
                        duration_slider = gr.Slider(label="Duration (seconds)", minimum=1.0, maximum=8.0, value=3.5, step=0.5)
                        steps_slider = gr.Slider(label="Inference Steps", minimum=4, maximum=30, value=6, step=1)
                        rife_multiplier = gr.Dropdown(
                            label="Motion Multiplier (RIFE)",
                            choices=[("Native (16 FPS)", 16), ("2x Smooth (32 FPS)", 32), ("3x Ultra (48 FPS)", 48)],
                            value=32,
                            interactive=True
                        )

                    with gr.Accordion("Advanced Scheduler & Guidance", open=False):
                        with gr.Row():
                            guidance_scale = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=10.0, value=5.0, step=0.5)
                            guidance_scale_2 = gr.Slider(label="Image Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.5)
                            flow_shift = gr.Slider(label="Flow Shift", minimum=1.0, maximum=10.0, value=3.0, step=0.5)
                        with gr.Row():
                            scheduler_dropdown = gr.Dropdown(
                                label="Scheduler",
                                choices=["UniPCMultistep", "FlowMatchEulerDiscrete", "DEISMultistep", "DPMSolverMultistep"],
                                value="UniPCMultistep"
                            )
                            seed_input = gr.Number(label="Seed", value=42, precision=0)
                            randomize_seed = gr.Checkbox(label="🎲 Randomize Seed", value=True)

                    generate_btn = gr.Button("🚀 Generate Video with Wan 2.2 (A100 GPU)", variant="primary", size="lg")

                # Right column: Video Player & Output
                with gr.Column(scale=5):
                    output_video = gr.Video(label="Generated Video", interactive=False, autoplay=True, loop=True, height=450)
                    download_video = gr.File(label="Download MP4", interactive=False)
                    video_info = gr.Markdown("")

                    with gr.Accordion("🎞️ Extract Frame from Output Video", open=False):
                        with gr.Row():
                            timestamp_slider = gr.Slider(label="Timestamp (seconds)", minimum=0.0, maximum=8.0, value=0.0, step=0.1)
                            extract_btn = gr.Button("📸 Extract Frame", size="sm")
                        extracted_preview = gr.Image(label="Extracted Frame Preview", type="pil", height=180)
                        with gr.Row():
                            send_to_first = gr.Button("⬅️ Use as First Frame", size="sm")
                            send_to_last = gr.Button("➡️ Use as Last Frame", size="sm")

        # ── Section 3: Outputs Gallery ──
        with gr.Accordion("📁 Smart Studio Outputs Gallery", open=False):
            gallery_list = gr.File(label="Output Videos", value=list_output_videos(), interactive=False)

        # ── Event Bindings ──

        # Hermes Prompt Refinement
        refine_btn.click(
            fn=on_refine_prompt_click,
            inputs=[hermes_raw_prompt, style_preset, motion_dynamics, camera_movement, hermes_temp],
            outputs=[hermes_refined_output, video_negative_prompt, hermes_status]
        )

        # Apply Refined Prompt to Wan
        apply_prompt_btn.click(
            fn=lambda p: p,
            inputs=[hermes_refined_output],
            outputs=[video_prompt]
        )

        # Wan Video Generation
        generate_btn.click(
            fn=on_generate_video_click,
            inputs=[
                first_frame,
                last_frame,
                video_prompt,
                video_negative_prompt,
                duration_slider,
                steps_slider,
                guidance_scale,
                guidance_scale_2,
                seed_input,
                randomize_seed,
                scheduler_dropdown,
                flow_shift,
                rife_multiplier
            ],
            outputs=[output_video, download_video, seed_input, video_info, gallery_list]
        )

        # Frame Extraction
        extract_btn.click(
            fn=extract_frame_from_video,
            inputs=[output_video, timestamp_slider],
            outputs=[extracted_preview]
        )
        send_to_first.click(
            fn=lambda img: img,
            inputs=[extracted_preview],
            outputs=[first_frame]
        )
        send_to_last.click(
            fn=lambda img: img,
            inputs=[extracted_preview],
            outputs=[last_frame]
        )

        # Header buttons
        hb_btn.click(
            fn=on_toggle_heartbeat,
            inputs=[heartbeat_state],
            outputs=[heartbeat_state, hb_btn]
        )
        clear_btn.click(
            fn=on_clear_outputs,
            inputs=[],
            outputs=[output_video, gallery_list, video_info]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
