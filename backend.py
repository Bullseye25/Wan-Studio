import io
import tempfile
import pathlib
import modal
import time


# Global cached model references to avoid redundant gRPC metadata lookups
_wan_model_instance = None
_flux_model_instance = None


def reset_cached_models():
    """Resets cached model instances (useful for testing and fresh reconnects)."""
    global _wan_model_instance, _flux_model_instance
    _wan_model_instance = None
    _flux_model_instance = None


def get_wan_model(force_refresh: bool = False):
    global _wan_model_instance
    if _wan_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("wan-backend", "WanModel")
        _wan_model_instance = model_cls()
    return _wan_model_instance


def get_flux_model(force_refresh: bool = False):
    global _flux_model_instance
    if _flux_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("flux-klein-backend", "FluxKleinModel")
        _flux_model_instance = model_cls()
    return _flux_model_instance


def run_remote_inference(
    resized_image,
    processed_last_image,
    prompt,
    steps,
    negative_prompt,
    num_frames,
    guidance_scale,
    guidance_scale_2,
    current_seed,
    scheduler_name,
    flow_shift,
    frame_multiplier,
    quality,
    duration_seconds,
    safe_mode=False,
    progress=None
):
    """
    Forwards the video generation request to the Modal backend.
    Converts inputs to bytes, streams progress updates, and saves the final video file.
    """
    # 1. Convert input images to bytes
    img_bytes_io = io.BytesIO()
    resized_image.save(img_bytes_io, format="PNG")
    image_bytes = img_bytes_io.getvalue()

    last_image_bytes = None
    if processed_last_image:
        last_bytes_io = io.BytesIO()
        processed_last_image.save(last_bytes_io, format="PNG")
        last_image_bytes = last_bytes_io.getvalue()

    # 2. Connect to the Modal App with automatic retry for transient network/DNS glitches
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
                f"Could not connect to Modal backend 'wan-backend'. Did you deploy it? "
                f"Run option 2 in start.py first. Error: {e}"
            )

    print(f"[local-backend] Initiating remote execution for seed {current_seed} on Modal...")
    
    if progress:
        progress(0, desc="Initializing remote GPU container...")

    video_path = None
    for attempt in range(max_retries + 1):
        try:
            # Refresh model handle on retry
            if attempt > 0:
                model_instance = get_wan_model(force_refresh=True)

            # Call the remote generator method
            generator = model_instance.generate.remote_gen(
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
                quality=int(quality)
            )

            for event in generator:
                etype = event.get("type")
                if etype == "progress":
                    step = event["step"]
                    total = event["total_steps"]
                    pct = float(step) / float(total)
                    if progress:
                        progress(pct, desc=f"Generating frames (Step {step}/{total})...")
                elif etype == "status":
                    if progress:
                        progress(0.95, desc=event["message"])
                elif etype == "error":
                    raise RuntimeError(f"Remote execution failed: {event['message']}\n{event['traceback']}")
                elif etype == "output":
                    # Create local output folder in current workspace directory and write result
                    local_dir = pathlib.Path(__file__).parent / "outputs"
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_path = local_dir / event["filename"]
                    
                    print(f"[local-backend] Writing output video: {local_path} ({len(event['video_bytes'])} bytes)")
                    with open(local_path, "wb") as f_out:
                        f_out.write(event["video_bytes"])
                    
                    video_path = str(local_path)
            break

        except Exception as e:
            # If transient connection / DNS error on first attempt, retry once
            is_conn_error = "getaddrinfo failed" in str(e) or "ConnectionError" in type(e).__name__ or "timeout" in str(e).lower()
            if is_conn_error and attempt < max_retries:
                print(f"[local-backend] Transient connection glitch ({e}), retrying in 1.5s...")
                time.sleep(1.5)
                continue
            print(f"[local-backend] Connection/Execution error: {e}")
            raise e

    if not video_path:
        raise RuntimeError("Modal backend finished execution but did not return any output video bytes.")

    record_wan_activity()

    return video_path, f"task_{current_seed}"


def flux_klein_remote(
    base_image: "PIL.Image.Image",  # type: ignore
    ref_images: list["PIL.Image.Image"],  # type: ignore
    prompt: str,
    negative_prompt: str = "",
    seed: int = -1,
    steps: int = 25,
    guidance_scale: float = 3.5,
    canvas_mode: str = "Scale base image",
    canvas_fit_mode: str = "Cover",
    pad_color: str = "#000000",
    custom_width: int = 1024,
    custom_height: int = 1024,
) -> bytes:
    """
    Forwards an image-generation request to the Modal FluxKleinModel.generate remote method.
    Saves the output image to the local outputs/ folder and returns the JPEG bytes.
    """
    import io
    import time
    import random

    # Convert base image to bytes
    base_bytes_io = io.BytesIO()
    base_image.save(base_bytes_io, format="PNG")
    base_image_bytes = base_bytes_io.getvalue()

    # Convert reference images to bytes
    ref_images_bytes = []
    for ref_img in ref_images:
        ref_bytes_io = io.BytesIO()
        ref_img.save(ref_bytes_io, format="PNG")
        ref_images_bytes.append(ref_bytes_io.getvalue())

    max_retries = 2
    model_instance = None
    for attempt in range(max_retries + 1):
        try:
            model_instance = get_flux_model(force_refresh=(attempt > 0))
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"Could not connect to Modal backend 'flux-klein-backend'. "
                f"Did you deploy it? Run option 3 in start.py first. Error: {e}"
            )

    print(f"[local-backend] Sending request to FLUX.2 Klein model on Modal...")
    result_bytes = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                model_instance = get_flux_model(force_refresh=True)

            result_bytes = model_instance.generate.remote(
                base_image_bytes=base_image_bytes,
                ref_images_bytes=ref_images_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                canvas_mode=canvas_mode,
                canvas_fit_mode=canvas_fit_mode,
                pad_color=pad_color,
                custom_width=int(custom_width),
                custom_height=int(custom_height),
            )
            break
        except Exception as e:
            is_conn_error = "getaddrinfo failed" in str(e) or "ConnectionError" in type(e).__name__ or "timeout" in str(e).lower()
            if is_conn_error and attempt < max_retries:
                print(f"[local-backend] Transient connection glitch on FLUX ({e}), retrying in 1.5s...")
                time.sleep(1.5)
                continue
            raise e

    # Save output image locally in outputs/ folder
    local_dir = pathlib.Path(__file__).parent / "outputs"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    file_seed = seed if seed >= 0 else random.randint(100000, 999999)
    filename = f"flux_{int(time.time())}_{file_seed}.jpg"
    local_path = local_dir / filename

    print(f"[local-backend] Writing output image: {local_path} ({len(result_bytes)} bytes)")
    with open(local_path, "wb") as f_out:
        f_out.write(result_bytes)

    # Record FLUX activity
    record_flux_activity()

    return result_bytes


# ─── Heartbeat & Cost-Effective Keep-Alive State ─────────────────────────────

last_frontend_heartbeat_time = time.time()
last_wan_activity_time = 0.0
last_flux_activity_time = 0.0

# Heartbeats default to False (Paused) to prevent unintended Modal GPU credit burn.
# Users can toggle keepalive on for active multi-generation sessions.
wan_heartbeat_enabled = False
flux_heartbeat_enabled = False

# Maximum idle duration (3 minutes) to keep a warm container alive after generation
KEEPALIVE_IDLE_WINDOW = 180.0


def record_frontend_heartbeat():
    global last_frontend_heartbeat_time
    last_frontend_heartbeat_time = time.time()


def record_wan_activity():
    global last_wan_activity_time
    last_wan_activity_time = time.time()


def record_flux_activity():
    global last_flux_activity_time
    last_flux_activity_time = time.time()


def set_wan_heartbeat_state(enabled: bool):
    global wan_heartbeat_enabled
    wan_heartbeat_enabled = enabled
    print(f"[backend] Wan heartbeat state toggled: {enabled}")


def set_flux_heartbeat_state(enabled: bool):
    global flux_heartbeat_enabled
    flux_heartbeat_enabled = enabled
    print(f"[backend] FLUX.2 Klein heartbeat state toggled: {enabled}")


def start_keepalive_thread():
    """
    Spawns a background thread that pings active backends every 25 seconds ONLY when:
      1. The frontend tab is actively open (heartbeat received within 45s).
      2. The user explicitly enabled keepalive for that model.
      3. The user generated with that specific model within the last 3 minutes.
    This strictly avoids waking up unneeded GPU containers or draining credits when idle.
    """
    import threading
    import time

    def keepalive_worker():
        time.sleep(5)
        print("[keepalive] Cost-optimized keep-alive worker started.")
        while True:
            now = time.time()
            is_frontend_open = (now - last_frontend_heartbeat_time) < 45.0

            if is_frontend_open:
                # 1. Ping Wan backend ONLY if enabled and recently used
                if wan_heartbeat_enabled and last_wan_activity_time > 0:
                    if (now - last_wan_activity_time) < KEEPALIVE_IDLE_WINDOW:
                        try:
                            m = get_wan_model()
                            m.heartbeat.remote()
                        except Exception:
                            pass

                # 2. Ping FLUX.2 Klein backend ONLY if enabled and recently used
                if flux_heartbeat_enabled and last_flux_activity_time > 0:
                    if (now - last_flux_activity_time) < KEEPALIVE_IDLE_WINDOW:
                        try:
                            m = get_flux_model()
                            m.heartbeat.remote()
                        except Exception:
                            pass
            
            time.sleep(25)

    t = threading.Thread(target=keepalive_worker, name="ModalKeepAliveThread")
    t.daemon = True
    t.start()






