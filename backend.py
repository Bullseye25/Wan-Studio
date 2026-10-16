import io
import tempfile
import pathlib
import modal
import time


# Global cached model references to avoid redundant gRPC metadata lookups
_wan_model_instance = None
_qwen_model_instance = None
_z_image_model_instance = None


def reset_cached_models():
    """Resets cached model instances (useful for testing and fresh reconnects)."""
    global _wan_model_instance, _qwen_model_instance, _z_image_model_instance
    _wan_model_instance = None
    _qwen_model_instance = None
    _z_image_model_instance = None


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


def get_z_image_model(force_refresh: bool = False):
    global _z_image_model_instance
    if _z_image_model_instance is None or force_refresh:
        for app_name, cls_name in [("adaab-z-image-turbo", "ZImageTurboModel"), ("soda-z-image-turbo", "ZImageTurboModel")]:
            try:
                model_cls = modal.Cls.from_name(app_name, cls_name)
                _z_image_model_instance = model_cls()
                break
            except Exception:
                continue
        if _z_image_model_instance is None:
            # Fallback direct lookup
            model_cls = modal.Cls.from_name("adaab-z-image-turbo", "ZImageTurboModel")
            _z_image_model_instance = model_cls()
    return _z_image_model_instance


def _image_to_optimized_bytes(pil_image):
    """Encodes PIL image to high-quality compressed WebP bytes (saving ~85% network bandwidth over PNG)."""
    buf = io.BytesIO()
    try:
        pil_image.save(buf, format="WEBP", quality=95)
    except Exception:
        buf.seek(0)
        buf.truncate()
        pil_image.save(buf, format="PNG")
    return buf.getvalue()


def run_remote_inference(
    resized_image=None,
    processed_last_image=None,
    prompt="",
    steps=8,
    negative_prompt="",
    num_frames=81,
    guidance_scale=2.0,
    guidance_scale_2=1.5,
    current_seed=42,
    scheduler_name="UniPCMultistep",
    flow_shift=3.0,
    frame_multiplier=16,
    quality=6,
    duration_seconds=3.5,
    safe_mode=False,
    lora_repo_or_path="",
    lora_scale=0.8,
    deflicker_frames=True,
    progress=None,
    image=None,
    last_frame_image=None,
    width=None,
    height=None,
):
    """
    Forwards the video generation request to the Modal backend.
    Converts inputs to bytes, streams progress updates, and saves the final video file.
    """
    if resized_image is None and image is not None:
        resized_image = image
    if processed_last_image is None and last_frame_image is not None:
        processed_last_image = last_frame_image

    # 1. Convert input images to optimized bytes
    image_bytes = _image_to_optimized_bytes(resized_image)
    last_image_bytes = _image_to_optimized_bytes(processed_last_image) if processed_last_image else None

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

            gen_kwargs = {
                "image_bytes": image_bytes,
                "last_image_bytes": last_image_bytes,
                "prompt": prompt,
                "steps": int(steps),
                "negative_prompt": negative_prompt,
                "num_frames": int(num_frames),
                "guidance_scale": float(guidance_scale),
                "guidance_scale_2": float(guidance_scale_2),
                "seed": int(current_seed),
                "scheduler_name": scheduler_name,
                "flow_shift": float(flow_shift),
                "frame_multiplier": int(frame_multiplier),
                "quality": int(quality),
                "lora_repo_or_path": str(lora_repo_or_path or ""),
                "lora_scale": float(lora_scale),
                "deflicker_frames": bool(deflicker_frames)
            }

            def consume_stream(kwargs_to_use):
                nonlocal video_path
                generator = model_instance.generate.remote_gen(**kwargs_to_use)
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
                        local_dir = pathlib.Path(__file__).parent / "outputs"
                        local_dir.mkdir(parents=True, exist_ok=True)
                        local_path = local_dir / event["filename"]
                        print(f"[local-backend] Writing output video: {local_path} ({len(event['video_bytes'])} bytes)")
                        with open(local_path, "wb") as f_out:
                            f_out.write(event["video_bytes"])
                        video_path = str(local_path)

            try:
                with _wan_ping_lock:
                    consume_stream(gen_kwargs)
            except Exception as stream_err:
                if "unexpected keyword" in str(stream_err) and ("lora_repo_or_path" in str(stream_err) or "deflicker_frames" in str(stream_err)):
                    print("[local-backend] Remote container uses earlier signature without lora/deflicker. Falling back to base signature...")
                    gen_kwargs.pop("lora_repo_or_path", None)
                    gen_kwargs.pop("lora_scale", None)
                    gen_kwargs.pop("deflicker_frames", None)
                    with _wan_ping_lock:
                        consume_stream(gen_kwargs)
                else:
                    raise stream_err
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


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, worst quality, jpeg artifacts, compression artifacts, "
    "pixelated, out of focus, grainy, noisy, oversaturated, undersaturated, bad anatomy, deformed, "
    "disfigured, extra limbs, missing limbs, extra fingers, missing fingers, fused fingers, mutated hands, "
    "poorly drawn hands, poorly drawn face, malformed limbs, long neck, distorted proportions, "
    "asymmetrical eyes, cross-eyed, watermark, signature, text, logo, cropped, cut off, frame, "
    "border, duplicate, cloned face, mutation, ugly, tiling"
)


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

    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if ref_images is None:
        ref_images = []

    # Convert base image to optimized bytes
    base_image_bytes = _image_to_optimized_bytes(base_image)

    # Convert reference images to optimized bytes
    ref_images_bytes = [_image_to_optimized_bytes(ref_img) for ref_img in ref_images]

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

            with _qwen_ping_lock:
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


def generate_z_image_turbo_remote(
    prompt: str,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    save_dir: str = None,
    filename_prefix: str = "char"
):
    """
    Invokes Z-Image Turbo on Modal (adaab-z-image-turbo / soda-z-image-turbo) to synthesize
    consistent, high-fidelity character reference portraits with locked seed control.
    Returns (PIL.Image.Image, local_file_path).
    """
    import io
    import time
    import base64
    from PIL import Image

    max_retries = 2
    model_instance = None
    for attempt in range(max_retries + 1):
        try:
            model_instance = get_z_image_model(force_refresh=(attempt > 0))
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            raise RuntimeError(
                f"Could not connect to Z-Image Turbo backend on Modal. Error: {e}"
            )

    print(f"[local-backend] Calling Z-Image Turbo (seed={seed}, steps={steps}, prompt='{prompt[:60]}...')...")
    res = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                model_instance = get_z_image_model(force_refresh=True)
            with _z_image_ping_lock:
                res = model_instance.generate.remote(
                    prompt=prompt,
                    height=int(height),
                    width=int(width),
                    num_inference_steps=int(steps),
                    seed=int(seed)
                )
            break
        except Exception as e:
            if attempt < max_retries:
                print(f"[local-backend] Transient error on Z-Image Turbo ({e}), retrying...")
                time.sleep(1.5)
                continue
            raise e

    # Extract image bytes
    if isinstance(res, dict) and "image_base64" in res:
        b64_data = res["image_base64"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
    elif isinstance(res, (bytes, bytearray)):
        img_bytes = bytes(res)
    elif hasattr(res, "save"):
        buf = io.BytesIO()
        res.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    else:
        raise ValueError(f"Unexpected response type from Z-Image Turbo: {type(res)}")

    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Determine local destination
    if save_dir is None:
        target_dir = pathlib.Path(__file__).parent / "outputs" / "studio" / "characters"
    else:
        target_dir = pathlib.Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    file_path = target_dir / f"{filename_prefix}_{seed}_{timestamp}.png"
    pil_img.save(file_path, format="PNG")
    print(f"[local-backend] Saved Z-Image Turbo portrait to: {file_path}")

    record_z_image_activity()

    return pil_img, str(file_path)


# ─── Cost-Optimized Heartbeat & On-Demand Keep-Warm State ────────────────────
import threading

last_frontend_heartbeat_time = time.time()
last_wan_activity_time = time.time()
last_qwen_activity_time = time.time()
last_z_image_activity_time = time.time()

# Non-blocking locks prevent a background heartbeat ping from EVER overlapping
# with an active generation or another in-flight cold-start ping (eliminates
# duplicate container spin-ups on Modal).
_wan_ping_lock = threading.Lock()
_qwen_ping_lock = threading.Lock()
_z_image_ping_lock = threading.Lock()

# COST SAVER: All three GPU heartbeats default to FALSE (On-Demand $0 Idle).
# Containers spin up only when a phase actually runs and scale down to $0
# within 15 seconds of finishing.
wan_heartbeat_enabled = False
qwen_heartbeat_enabled = False
z_image_heartbeat_enabled = False

FRONTEND_HEARTBEAT_TIMEOUT = 50.0
# Auto-shutoff if a user manually toggles keep-warm ON and leaves it idle for >3 min
KEEPALIVE_IDLE_TIMEOUT = 180.0


def record_frontend_heartbeat():
    global last_frontend_heartbeat_time
    last_frontend_heartbeat_time = time.time()


def record_wan_activity():
    global last_wan_activity_time
    last_wan_activity_time = time.time()


def record_qwen_activity():
    global last_qwen_activity_time
    last_qwen_activity_time = time.time()


def record_z_image_activity():
    global last_z_image_activity_time
    last_z_image_activity_time = time.time()


def set_wan_heartbeat_state(enabled: bool):
    global wan_heartbeat_enabled
    wan_heartbeat_enabled = bool(enabled)
    if wan_heartbeat_enabled:
        record_wan_activity()
    print(f"[backend] Wan heartbeat state toggled: {wan_heartbeat_enabled}")


def set_qwen_heartbeat_state(enabled: bool):
    global qwen_heartbeat_enabled
    qwen_heartbeat_enabled = bool(enabled)
    if qwen_heartbeat_enabled:
        record_qwen_activity()
    print(f"[backend] Qwen Image Edit heartbeat state toggled: {qwen_heartbeat_enabled}")


def set_z_image_heartbeat_state(enabled: bool):
    global z_image_heartbeat_enabled
    z_image_heartbeat_enabled = bool(enabled)
    if z_image_heartbeat_enabled:
        record_z_image_activity()
    print(f"[backend] Z-Image Turbo heartbeat state toggled: {z_image_heartbeat_enabled}")


def activate_studio_phase(active_phase: int, trigger_warmup: bool = True):
    """
    Phase-Driven Single-GPU Lifecycle:
      - Phase 1 (Story):      All 3 GPUs OFF ($0.00/hr — uses Ollama Cloud only)
      - Phase 2 (Characters): ONLY Z-Image Turbo (L4) ON; Qwen & Wan OFF
      - Phase 3 (Scenes):     ONLY Qwen Image Edit (L40S) ON; Z-Image & Wan OFF
      - Phase 4 (Videos):     ONLY Wan 2.2 (A100) ON; Z-Image & Qwen OFF
    """
    set_z_image_heartbeat_state(active_phase == 2)
    set_qwen_heartbeat_state(active_phase == 3)
    set_wan_heartbeat_state(active_phase == 4)

    if trigger_warmup and active_phase in (2, 3, 4):
        ping_all_active_backends()


def release_other_phase_heartbeats(active_phase: int, trigger_warmup: bool = False):
    """
    Activates only the GPU needed for `active_phase` and releases
    heartbeats for all other phases so idle containers scale down to $0 within 15s.
    """
    activate_studio_phase(active_phase, trigger_warmup=trigger_warmup)


def ping_all_active_backends():
    """
    Pings only explicitly enabled Modal backends using non-blocking locks so
    concurrent heartbeats NEVER queue up duplicate containers during cold start
    or generation.
    """
    def _ping(getter, label, lock):
        if not lock.acquire(blocking=False):
            # Another ping or generation is already in flight on this model; skip!
            return
        try:
            m = getter()
            m.heartbeat.remote()
        except Exception as e:
            print(f"[keepalive] {label} ping note: {e}")
        finally:
            lock.release()

    threads = []
    if z_image_heartbeat_enabled:
        threads.append(threading.Thread(target=_ping, args=(get_z_image_model, "Z-Image Turbo", _z_image_ping_lock), daemon=True))
    if qwen_heartbeat_enabled:
        threads.append(threading.Thread(target=_ping, args=(get_qwen_model, "Qwen Image Edit", _qwen_ping_lock), daemon=True))
    if wan_heartbeat_enabled:
        threads.append(threading.Thread(target=_ping, args=(get_wan_model, "Wan 2.2", _wan_ping_lock), daemon=True))

    for t in threads:
        t.start()
    return len(threads)


def start_keepalive_thread():
    """
    Spawns a background daemon thread that only pings models whose keep-warm
    toggle is explicitly enabled AND within the 3-minute activity window.
    """
    def keepalive_worker():
        time.sleep(5)
        print("[keepalive] Cost-Saver Keep-Alive worker started (On-Demand mode: $0 idle cost by default).")
        while True:
            now = time.time()
            is_frontend_open = (now - last_frontend_heartbeat_time) < FRONTEND_HEARTBEAT_TIMEOUT

            # Auto-disable any heartbeat that has been idle for > KEEPALIVE_IDLE_TIMEOUT (3 min)
            if z_image_heartbeat_enabled and (now - last_z_image_activity_time) > KEEPALIVE_IDLE_TIMEOUT:
                print("[keepalive] Auto-pausing Z-Image Turbo heartbeat after 3m idle to save GPU cost.")
                set_z_image_heartbeat_state(False)
            if qwen_heartbeat_enabled and (now - last_qwen_activity_time) > KEEPALIVE_IDLE_TIMEOUT:
                print("[keepalive] Auto-pausing Qwen heartbeat after 3m idle to save GPU cost.")
                set_qwen_heartbeat_state(False)
            if wan_heartbeat_enabled and (now - last_wan_activity_time) > KEEPALIVE_IDLE_TIMEOUT:
                print("[keepalive] Auto-pausing Wan 2.2 heartbeat after 3m idle to save GPU cost.")
                set_wan_heartbeat_state(False)

            if is_frontend_open and (z_image_heartbeat_enabled or qwen_heartbeat_enabled or wan_heartbeat_enabled):
                ping_all_active_backends()

            time.sleep(25)

    t = threading.Thread(target=keepalive_worker, name="ModalKeepAliveThread")
    t.daemon = True
    t.start()

