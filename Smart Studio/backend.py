import io
import os
import time
import pathlib
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Smart Studio Local Backend RPC Layer
# Communicates with Modal app: smart-wan-backend
# ─────────────────────────────────────────────────────────────────────────────

_wan_model_instance = None
_hermes_model_instance = None

smart_heartbeat_enabled = False
last_smart_activity_time = 0.0
last_frontend_heartbeat_time = 0.0

KEEPALIVE_IDLE_WINDOW = 180.0
TAB_ACTIVITY_WINDOW = 45.0


def reset_cached_models():
    """Resets cached model instances for testing and fresh reconnections."""
    global _wan_model_instance, _hermes_model_instance
    _wan_model_instance = None
    _hermes_model_instance = None


def record_smart_activity():
    global last_smart_activity_time
    last_smart_activity_time = time.time()


def record_frontend_heartbeat():
    global last_frontend_heartbeat_time
    last_frontend_heartbeat_time = time.time()


def set_smart_heartbeat_state(enabled: bool):
    global smart_heartbeat_enabled
    smart_heartbeat_enabled = bool(enabled)
    if smart_heartbeat_enabled:
        record_smart_activity()
    print(f"[Smart Studio] Heartbeat state toggled: {smart_heartbeat_enabled}")


def get_wan_model(force_refresh: bool = False):
    global _wan_model_instance
    if _wan_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("smart-wan-backend", "SmartWanModel")
        _wan_model_instance = model_cls()
    return _wan_model_instance


def get_hermes_model(force_refresh: bool = False):
    global _hermes_model_instance
    if _hermes_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("smart-wan-backend", "HermesPromptRefiner")
        _hermes_model_instance = model_cls()
    return _hermes_model_instance


# ─── Hermes Smart Prompt Generator ──────────────────────────────────────────

def build_local_fallback_prompt(raw_prompt: str, style_preset: str, motion_dynamics: str, camera_movement: str) -> dict:
    """Intelligent fallback enhancement if Modal is offline or starting up."""
    style_keywords = {
        "Cinematic Film (35mm)": "shot on 35mm anamorphic lens, cinematic film grain, cinematic depth of field, warm theatrical lighting, photorealistic color grade",
        "Hyperrealistic 4K": "hyperrealistic 8k masterwork, volumetric lighting, subsurface scattering, ultra-high resolution, extreme detail",
        "Cyberpunk & Sci-Fi": "neo-Tokyo cyberpunk aesthetics, glowing neon reflections, holographic chromatic aberration, atmospheric drizzle, dark moody palette",
        "Anime / Studio Ghibli": "exquisite Makoto Shinkai and Ghibli inspired anime aesthetic, lush vibrant colors, hand-painted background details, whimsical volumetric light",
        "Dark Noir & Atmospheric": "high-contrast chiaroscuro lighting, heavy shadows, moody atmospheric smoke, vintage film noir aesthetic",
        "Fantasy & Mythical": "ethereal fantasy painting, glowing magical luminescence, enchanted environment, epic cinematic atmosphere",
        "Drone Aerial Nature": "sweeping high-altitude aerial drone shot, majestic natural vista, golden hour panoramic lighting, crisp horizon",
    }
    style_desc = style_keywords.get(style_preset, "cinematic masterwork, photorealistic lighting, highly detailed")
    motion_desc = f"with {motion_dynamics.lower()} and {camera_movement.lower()}"
    refined = f"{raw_prompt.strip().rstrip('.')}, {motion_desc}, {style_desc}, 4k resolution, smooth animation, preserve face details."
    
    return {
        "refined_prompt": refined,
        "negative_prompt": "blurry, low quality, worst quality, deformed, bad anatomy, jittery motion, erratic camera shake, morphing, watermark",
        "style_applied": style_preset,
        "elapsed_seconds": 0.01,
        "fallback": True
    }


def _safe_progress(progress, val, desc=None):
    if progress is not None:
        try:
            if desc is not None:
                progress(val, desc=desc)
            else:
                progress(val)
        except Exception:
            pass


def refine_prompt_hermes(
    raw_prompt: str,
    style_preset: str = "Cinematic Film (35mm)",
    motion_dynamics: str = "Smooth & Flowing Motion",
    camera_movement: str = "Auto Best Fit",
    temperature: float = 0.7,
    progress=None
) -> dict:
    """
    Sends raw prompt and visual directives to Hermes 3 on Modal.
    Returns enhanced cinematic prompt and negative prompt.
    """
    if not raw_prompt or not raw_prompt.strip():
        raise ValueError("Please provide a prompt to refine.")

    record_smart_activity()
    _safe_progress(progress, 0.2, desc="Connecting to Hermes 3 on Modal...")

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            hermes = get_hermes_model(force_refresh=(attempt > 0))
            _safe_progress(progress, 0.5, desc="Hermes 3 analyzing & expanding scene composition...")

            res = hermes.refine_prompt.remote(
                raw_prompt=raw_prompt,
                style_preset=style_preset,
                motion_dynamics=motion_dynamics,
                camera_movement=camera_movement,
                temperature=temperature
            )
            _safe_progress(progress, 1.0, desc="Prompt refined successfully!")
            return res
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            print(f"[Smart Studio] Hermes remote call warning: {e}. Falling back to rule-based director engine.")
            return build_local_fallback_prompt(raw_prompt, style_preset, motion_dynamics, camera_movement)


# ─── Wan 2.2 Remote Inference ────────────────────────────────────────────────

def run_remote_inference(
    resized_image,
    processed_last_image,
    prompt: str,
    steps: int,
    negative_prompt: str,
    num_frames: int,
    guidance_scale: float,
    guidance_scale_2: float,
    current_seed: int,
    scheduler_name: str,
    flow_shift: float,
    frame_multiplier: int,
    quality: int,
    duration_seconds: float,
    progress=None
) -> str:
    """
    Forwards video generation request to Modal SmartWanModel.
    Streams generation progress and saves output MP4.
    """
    record_smart_activity()

    # 1. Convert input images to bytes
    img_bytes_io = io.BytesIO()
    resized_image.save(img_bytes_io, format="PNG")
    image_bytes = img_bytes_io.getvalue()

    last_image_bytes = None
    if processed_last_image:
        last_bytes_io = io.BytesIO()
        processed_last_image.save(last_bytes_io, format="PNG")
        last_image_bytes = last_bytes_io.getvalue()

    # 2. Connect to Modal SmartWanModel
    max_retries = 2
    model_instance = None
    for attempt in range(max_retries + 1):
        try:
            model_instance = get_wan_model(force_refresh=(attempt > 0))
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"Could not connect to Modal backend 'smart-wan-backend'. Did you deploy it?\n"
                f"Run Option 2 in start.py first. Error: {e}"
            )

    print(f"[Smart Studio] Initiating remote Wan 2.2 execution for seed {current_seed} on Modal...")
    _safe_progress(progress, 0, desc="Initializing remote A100 GPU container...")

    video_path = None

    # Stream progress from remote generator
    for update in model_instance.generate.remote_gen(
        image_bytes=image_bytes,
        last_image_bytes=last_image_bytes,
        prompt=prompt,
        steps=int(steps),
        negative_prompt=negative_prompt,
        num_frames=int(num_frames),
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        seed=int(current_seed),
        scheduler_name=scheduler_name,
        flow_shift=float(flow_shift),
        frame_multiplier=int(frame_multiplier),
        quality=int(quality),
    ):
        if update["type"] == "progress":
            step = update["step"]
            total = update["total_steps"]
            fraction = (step + 1) / total
            _safe_progress(progress, fraction, desc=f"Denoising step {step + 1}/{total} on A100 GPU...")
        elif update["type"] == "status":
            _safe_progress(progress, None, desc=update["message"])
        elif update["type"] == "output":
            outputs_dir = pathlib.Path(__file__).parent / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            video_filename = f"smart_wan_{int(time.time())}_{current_seed}.mp4"
            target_path = outputs_dir / video_filename
            print(f"[Smart Studio] Writing output video: {target_path} ({len(update['video_bytes'])} bytes)")
            with open(target_path, "wb") as f:
                f.write(update["video_bytes"])
            video_path = str(target_path)
            break
        elif update["type"] == "error":
            raise RuntimeError(f"Remote generation error: {update['message']}\n{update.get('traceback', '')}")

    if not video_path:
        raise RuntimeError("Remote generation failed to return video output.")

    return video_path


def clear_all_outputs() -> str:
    """Safely clears all video and image files in Smart Studio outputs directory."""
    outputs_dir = pathlib.Path(__file__).parent / "outputs"
    if not outputs_dir.exists():
        return "Outputs folder is already empty."

    deleted_count = 0
    for item in list(outputs_dir.iterdir()):
        try:
            if item.is_file():
                item.unlink()
                deleted_count += 1
        except Exception as e:
            print(f"[Smart Studio] Could not delete {item}: {e}")

    return f"Cleared {deleted_count} item(s) from Smart Studio outputs folder."
