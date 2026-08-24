import io
import tempfile
import pathlib
import modal
import time


# Global cached model references to avoid redundant gRPC metadata lookups
_wan_model_instance = None
_qwen_model_instance = None


def reset_cached_models():
    """Resets cached model instances (useful for testing and fresh reconnects)."""
    global _wan_model_instance, _qwen_model_instance
    _wan_model_instance = None
    _qwen_model_instance = None


def get_wan_model(force_refresh: bool = False):
    global _wan_model_instance
    if _wan_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("wan-backend", "WanModel")
        _wan_model_instance = model_cls()
    return _wan_model_instance


def get_qwen_model(force_refresh: bool = False):
    global _qwen_model_instance
    if _qwen_model_instance is None or force_refresh:
        model_cls = modal.Cls.from_name("qwen-image-backend", "QwenImageEditModel")
        _qwen_model_instance = model_cls()
    return _qwen_model_instance


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


def qwen_image_edit_remote(
    base_image: "PIL.Image.Image",  # type: ignore
    ref_images: list["PIL.Image.Image"] = None,  # type: ignore
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    steps: int = 20,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    num_images: int = 1,
    custom_width: int = 1024,
    custom_height: int = 1024,
) -> list[bytes]:
    """
    Forwards an image-generation/editing request to the Modal QwenImageEditModel.generate remote method.
    Saves the output images to the local outputs/ folder and returns a list of JPEG bytes.
    """
    import io
    import time
    import random

    if ref_images is None:
        ref_images = []

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
            model_instance = get_qwen_model(force_refresh=(attempt > 0))
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"Could not connect to Modal backend 'qwen-image-backend'. "
                f"Did you deploy it? Run option 3 in start.py first. Error: {e}"
            )

    print(f"[local-backend] Sending request to Qwen Image Edit model on Modal ({num_images} image(s))...")
    result_bytes_list = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                model_instance = get_qwen_model(force_refresh=True)

            raw_result = model_instance.generate.remote(
                base_image_bytes=base_image_bytes,
                ref_images_bytes=ref_images_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                steps=int(steps),
                true_cfg_scale=float(true_cfg_scale),
                guidance_scale=float(guidance_scale),
                num_images=int(num_images),
                custom_width=int(custom_width),
                custom_height=int(custom_height),
            )
            result_bytes_list = raw_result if isinstance(raw_result, list) else [raw_result]
            break
        except Exception as e:
            is_conn_error = "getaddrinfo failed" in str(e) or "ConnectionError" in type(e).__name__ or "timeout" in str(e).lower()
            if is_conn_error and attempt < max_retries:
                print(f"[local-backend] Transient connection glitch on Qwen ({e}), retrying in 1.5s...")
                time.sleep(1.5)
                continue
            raise e

    # Save output images locally in outputs/ folder
    local_dir = pathlib.Path(__file__).parent / "outputs"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    base_file_seed = seed if seed >= 0 else random.randint(100000, 999999)
    for idx, img_bytes in enumerate(result_bytes_list):
        current_file_seed = base_file_seed + idx
        filename = f"qwen_{timestamp}_{current_file_seed}_{idx+1}.jpg" if len(result_bytes_list) > 1 else f"qwen_{timestamp}_{current_file_seed}.jpg"
        local_path = local_dir / filename

        print(f"[local-backend] Writing output image {idx+1}/{len(result_bytes_list)}: {local_path} ({len(img_bytes)} bytes)")
        with open(local_path, "wb") as f_out:
            f_out.write(img_bytes)

    # Record Qwen activity
    record_qwen_activity()

    return result_bytes_list


# ─── Heartbeat & Cost-Effective Keep-Alive State ─────────────────────────────

last_frontend_heartbeat_time = time.time()
last_wan_activity_time = 0.0
last_qwen_activity_time = 0.0

# Heartbeats default to False (Paused) to prevent unintended Modal GPU credit burn.
# Users can toggle keepalive on for active multi-generation sessions.
wan_heartbeat_enabled = False
qwen_heartbeat_enabled = False

# Maximum idle duration (3 minutes) to keep a warm container alive after generation
KEEPALIVE_IDLE_WINDOW = 180.0


def record_frontend_heartbeat():
    global last_frontend_heartbeat_time
    last_frontend_heartbeat_time = time.time()


def record_wan_activity():
    global last_wan_activity_time
    last_wan_activity_time = time.time()


def record_qwen_activity():
    global last_qwen_activity_time
    last_qwen_activity_time = time.time()


def set_wan_heartbeat_state(enabled: bool):
    global wan_heartbeat_enabled
    wan_heartbeat_enabled = enabled
    print(f"[backend] Wan heartbeat state toggled: {enabled}")


def set_qwen_heartbeat_state(enabled: bool):
    global qwen_heartbeat_enabled
    qwen_heartbeat_enabled = enabled
    print(f"[backend] Qwen Image Edit heartbeat state toggled: {enabled}")


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

                # 2. Ping Qwen backend ONLY if enabled and recently used
                if qwen_heartbeat_enabled and last_qwen_activity_time > 0:
                    if (now - last_qwen_activity_time) < KEEPALIVE_IDLE_WINDOW:
                        try:
                            m = get_qwen_model()
                            m.heartbeat.remote()
                        except Exception:
                            pass
            
            time.sleep(25)

    t = threading.Thread(target=keepalive_worker, name="ModalKeepAliveThread")
    t.daemon = True
    t.start()






