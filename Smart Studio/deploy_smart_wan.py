import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: smart-wan-backend
# Dedicated Wan 2.2 Video Studio + Hermes 3 Smart Prompt Refiner
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("smart-wan-backend")

# Shared persistent volume for all cached Hugging Face model weights
weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)

# ─── Secrets & Credentials ───────────────────────────────────────────────────
def get_credentials():
    """Reads HF_TOKEN from environment or local credentials.txt."""
    creds = {"HF_TOKEN": os.environ.get("HF_TOKEN")}
    for candidate in [
        os.path.join(os.path.dirname(__file__), "credentials.txt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.txt"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            creds["HF_TOKEN"] = line.split("=", 1)[1]
                        elif line.startswith("hf_"):
                            creds["HF_TOKEN"] = line
            except Exception:
                pass
            break
    return creds

secrets_dict = get_credentials()
secrets = []
if secrets_dict.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": secrets_dict["HF_TOKEN"]}))


# ─── Modal Images ────────────────────────────────────────────────────────────

# Image for Wan 2.2 Image-To-Video + RIFE
wan_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "build-essential", "ninja-build",
        "libjpeg-dev", "wget", "tar", "xz-utils", "ffmpeg", "unzip"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "git clone https://huggingface.co/spaces/kulkas2pintu/wan555 /app"
    )
    .run_commands(
        "cd /app && wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip -O RIFEv4.26_0921.zip && unzip -o RIFEv4.26_0921.zip && rm RIFEv4.26_0921.zip"
    )
    .pip_install(
        "diffusers==0.38.0", "transformers==4.57.6", "accelerate==1.13.0", "safetensors",
        "sentencepiece", "peft==0.19.1", "ftfy", "imageio", "imageio-ffmpeg",
        "opencv-python-headless", "torchao==0.15.0", "numpy==1.23.5", "tqdm", "gradio>=5.0,<6.0"
    )
)

# Image for Hermes 3 Prompt Refiner
hermes_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install(
        "git", "build-essential", "ninja-build"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "transformers>=4.45.0", "accelerate>=1.0.0", "safetensors",
        "sentencepiece", "huggingface_hub", "bitsandbytes>=0.43.0"
    )
)


# ─── Pre-Cache Weight Functions ──────────────────────────────────────────────

@app.function(
    image=wan_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_wan_weights():
    """Pre-caches Wan 2.2 model weights into the persistent volume."""
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"HF login skipped: {e}")

    print("Pre-caching Wan 2.2 model weights to persistent volume `/root/.cache/huggingface`...")
    snapshot_download(
        repo_id="TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
        cache_dir="/root/.cache/huggingface"
    )
    weights_vol.commit()
    print("✓ Wan 2.2 model weights cached successfully!")


@app.function(
    image=hermes_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_hermes_weights():
    """Pre-caches NousResearch/Hermes-3-Llama-3.1-8B model weights into persistent volume."""
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"HF login skipped: {e}")

    print("Pre-caching Hermes 3 (NousResearch/Hermes-3-Llama-3.1-8B) weights...")
    snapshot_download(
        repo_id="NousResearch/Hermes-3-Llama-3.1-8B",
        cache_dir="/root/.cache/huggingface"
    )
    weights_vol.commit()
    print("✓ Hermes 3 model weights cached successfully!")


@app.function(
    image=wan_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_all_smart_weights():
    """Pre-caches both Wan 2.2 and Hermes 3 model weights."""
    print("=== Step 1/2: Pre-caching Wan 2.2 ===")
    download_wan_weights.local()
    print("\n=== Step 2/2: Pre-caching Hermes 3 ===")
    download_hermes_weights.local()
    print("\n✓ ALL Smart Studio model weights cached successfully in Modal Volume!")


# ─── Hermes 3 Prompt Refiner ────────────────────────────────────────────────

HERMES_SYSTEM_PROMPT = """You are Hermes, a world-class cinematic video director and master prompt engineer specialized in the Wan 2.1 and Wan 2.2 video diffusion models.
Your objective is to transform raw user concepts into richly detailed, physically grounded, cinematic video generation prompts that maximize visual realism, coherent camera movement, and fluid motion dynamics.

When expanding the prompt:
1. Core Subject & Actions: Describe characters, attire, expressions, anatomical fidelity, textures, and specific natural physical actions.
2. Motion & Physics: Detail natural motion dynamics (cloth fluttering in the wind, hair flowing, water ripples, subtle breathing, organic momentum).
3. Cinematography & Lens: Specify exact camera movements (e.g. slow push-in tracking shot, sweeping low-angle orbit, cinematic shallow depth of field, 35mm anamorphic lens, bokeh).
4. Lighting & Atmosphere: Detail lighting schemes (e.g. golden hour volumetric rim lighting, soft diffuse shadows, moody chiaroscuro, atmospheric dust motes, specular reflections).
5. Visual Rendering: 4k resolution, cinematic color grade, photorealistic masterwork.

Output format:
Return ONLY the expanded cinematic prompt as a cohesive, vivid paragraph (60-120 words). Do not include conversational filler, preamble, or markdown bullet points."""

HERMES_NEGATIVE_DEFAULT = (
    "blurry, low quality, worst quality, deformed, disfigured, bad anatomy, "
    "jittery motion, erratic camera shake, frame skipping, morphing, text, watermark, signature"
)


@app.cls(
    image=hermes_image,
    gpu="L4",  # L4 or A10G runs 8B model with 4-bit / bfloat16 fast and cost-effectively
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=600,
    min_containers=0,
    scaledown_window=60,
)
class HermesPromptRefiner:
    @modal.method()
    def heartbeat(self):
        return "alive"

    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN")
        model_id = "NousResearch/Hermes-3-Llama-3.1-8B"
        print(f"[HermesPromptRefiner] Loading {model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir="/root/.cache/huggingface"
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
            cache_dir="/root/.cache/huggingface"
        )
        print("[HermesPromptRefiner] Hermes 3 loaded successfully on GPU!")

    @modal.method()
    def refine_prompt(
        self,
        raw_prompt: str,
        style_preset: str = "Cinematic Film (35mm)",
        motion_dynamics: str = "Smooth & Flowing Motion",
        camera_movement: str = "Auto Best Fit",
        temperature: float = 0.7,
        max_new_tokens: int = 250,
    ) -> dict:
        import time
        import torch

        start_time = time.time()
        print(f"[HermesPromptRefiner] Refining prompt: '{raw_prompt[:60]}...'")

        user_instruction = (
            f"User Prompt: {raw_prompt}\n"
            f"Desired Style: {style_preset}\n"
            f"Motion Dynamics: {motion_dynamics}\n"
            f"Camera Direction: {camera_movement}\n\n"
            f"Generate the ultimate Wan 2.2 cinematic video generation prompt."
        )

        messages = [
            {"role": "system", "content": HERMES_SYSTEM_PROMPT},
            {"role": "user", "content": user_instruction}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        refined = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean up any surrounding quotes or markdown
        refined = refined.strip('"\'')
        elapsed = round(time.time() - start_time, 2)
        print(f"[HermesPromptRefiner] Refined in {elapsed}s:\n{refined}")

        return {
            "refined_prompt": refined,
            "negative_prompt": HERMES_NEGATIVE_DEFAULT,
            "style_applied": style_preset,
            "elapsed_seconds": elapsed,
        }


# ─── Wan 2.2 Video Model ────────────────────────────────────────────────────

@app.cls(
    image=wan_image,
    gpu="A100-80GB",
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=1800,
    min_containers=0,
    scaledown_window=60,
)
class SmartWanModel:
    @modal.method()
    def heartbeat(self):
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import sys
        sys.path.append("/app")
        import torch
        from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline

        # Load RIFE model
        print("[SmartWanModel] Loading RIFE motion interpolation model...")
        from train_log.RIFE_HDv3 import Model as RIFEModel
        self.rife_model = RIFEModel()
        self.rife_model.load_model("/app/train_log", -1)
        self.rife_model.eval()

        # Load Wan 2.2 Pipeline
        print("[SmartWanModel] Loading Wan 2.2 Image-To-Video pipeline onto A100-80GB...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
            torch_dtype=torch.bfloat16,
            cache_dir="/root/.cache/huggingface"
        ).to("cuda")

        self.original_scheduler = self.pipe.scheduler
        print("[SmartWanModel] Wan 2.2 Pipeline ready on GPU!")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        last_image_bytes: bytes | None,
        prompt: str,
        steps: int,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        guidance_scale_2: float,
        seed: int,
        scheduler_name: str,
        flow_shift: float,
        frame_multiplier: int,
        quality: int
    ):
        import io
        import copy
        import queue
        import threading
        import gc
        import torch
        from PIL import Image
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            SASolverScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepInverseScheduler,
            UniPCMultistepScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
        )

        SCHEDULER_MAP = {
            "FlowMatchEulerDiscrete": FlowMatchEulerDiscreteScheduler,
            "SASolver": SASolverScheduler,
            "DEISMultistep": DEISMultistepScheduler,
            "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
            "UniPCMultistep": UniPCMultistepScheduler,
            "DPMSolverMultistep": DPMSolverMultistepScheduler,
            "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
        }

        # 1. Process inputs
        resized_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_last_image = None
        if last_image_bytes:
            processed_last_image = Image.open(io.BytesIO(last_image_bytes)).convert("RGB")

        # 2. Setup Scheduler
        scheduler_class = SCHEDULER_MAP.get(scheduler_name, UniPCMultistepScheduler)
        if scheduler_class.__name__ != self.pipe.scheduler.config._class_name or flow_shift != self.pipe.scheduler.config.get("flow_shift", "shift"):
            config = copy.deepcopy(self.original_scheduler.config)
            if scheduler_class == FlowMatchEulerDiscreteScheduler:
                config['shift'] = flow_shift
            else:
                config['flow_shift'] = flow_shift
            self.pipe.scheduler = scheduler_class.from_config(config)

        # 3. Setup Async pipeline run with queue
        gc.collect()
        torch.cuda.empty_cache()

        q = queue.Queue()

        def step_callback(pipe_obj, step_idx, timestep, callback_kwargs):
            q.put({"type": "progress", "step": step_idx, "total_steps": steps})
            return callback_kwargs

        def run_inference_thread():
            try:
                print(f"[SmartWanModel] Remote inference started: seed={seed}, steps={steps}...")
                generator = torch.Generator(device="cuda").manual_seed(seed)
                result = self.pipe(
                    image=resized_image,
                    last_image=processed_last_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=resized_image.height,
                    width=resized_image.width,
                    num_frames=num_frames,
                    guidance_scale=float(guidance_scale),
                    guidance_scale_2=float(guidance_scale_2),
                    num_inference_steps=int(steps),
                    generator=generator,
                    output_type="np",
                    callback_on_step_end=step_callback
                )
                raw_frames = result.frames[0]
                q.put({"type": "success", "frames": raw_frames})
            except Exception as e:
                import traceback
                q.put({"type": "error", "message": str(e), "traceback": traceback.format_exc()})

        thread = threading.Thread(target=run_inference_thread)
        thread.start()

        # Yield progress from the thread
        raw_frames_np = None
        while thread.is_alive() or not q.empty():
            try:
                event = q.get(timeout=0.2)
                if event["type"] == "progress":
                    yield {"type": "progress", "step": event["step"], "total_steps": event["total_steps"]}
                elif event["type"] == "error":
                    yield {"type": "error", "message": event["message"], "traceback": event["traceback"]}
                    return
                elif event["type"] == "success":
                    raw_frames_np = event["frames"]
            except queue.Empty:
                continue

        thread.join(timeout=5.0)
        self.pipe.scheduler = self.original_scheduler

        if raw_frames_np is None:
            yield {"type": "error", "message": "Pipeline did not produce frames", "traceback": ""}
            return

        # 4. Interpolate using RIFE
        frame_factor = frame_multiplier // 16
        if frame_factor > 1:
            yield {"type": "status", "message": f"Processing frames with RIFE ({frame_factor}x smooth interpolation)..."}
            final_frames = self.interpolate_bits_internal(raw_frames_np, multiplier=int(frame_factor))
        else:
            final_frames = list(raw_frames_np)

        # 5. Export to video
        import tempfile
        from diffusers.utils.export_utils import export_to_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            video_path = tmpfile.name

        final_fps = 16 * int(frame_factor)
        export_to_video(final_frames, video_path, fps=final_fps, quality=quality)

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        del raw_frames_np, final_frames
        gc.collect()
        torch.cuda.empty_cache()

        yield {"type": "output", "video_bytes": video_bytes, "filename": f"smart_wan_{seed}.mp4"}

    def interpolate_bits_internal(self, frames_np, multiplier=2, scale=1.0):
        import torch
        from torch.nn import functional as F

        device = torch.device("cuda")
        T, H, W, C = frames_np.shape
        n_interp = multiplier - 1

        tmp = max(128, int(128 / scale))
        ph = ((H - 1) // tmp + 1) * tmp
        pw = ((W - 1) // tmp + 1) * tmp
        padding = (0, pw - W, 0, ph - H)

        def to_tensor(frame_np):
            t = torch.from_numpy(frame_np).to(device)
            t = t.permute(2, 0, 1).unsqueeze(0)
            return F.pad(t, padding).half()

        def from_tensor(tensor):
            t = tensor[0, :, :H, :W]
            t = t.permute(1, 2, 0)
            return t.float().cpu().numpy()

        def make_inference(I0, I1, n):
            if self.rife_model.version >= 3.9:
                res = []
                for i in range(n):
                    res.append(self.rife_model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
                return res
            else:
                middle = self.rife_model.inference(I0, I1, scale)
                if n == 1:
                    return [middle]
                first_half = make_inference(I0, middle, n=n//2)
                second_half = make_inference(middle, I1, n=n//2)
                if n % 2:
                    return [*first_half, middle, *second_half]
                else:
                    return [*first_half, *second_half]

        output_frames = []
        I1 = to_tensor(frames_np[0])

        for i in range(T - 1):
            I0 = I1
            output_frames.append(from_tensor(I0))
            I1 = to_tensor(frames_np[i+1])
            mid_tensors = make_inference(I0, I1, n_interp)
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))

        output_frames.append(from_tensor(I1))

        del I0, I1
        torch.cuda.empty_cache()
        return output_frames
