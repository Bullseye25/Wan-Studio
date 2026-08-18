import os
import modal

# Setup Modal app
app = modal.App("wan-backend")

# Define Modal Volumes for caching HF models and saving outputs
weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("wan-outputs", create_if_missing=True)

# Build Modal Image with all required system packages, CUDA-enabled PyTorch, and RIFE + dependencies
wan_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "build-essential", "ninja-build", 
        "libjpeg-dev", "wget", "tar", "xz-utils", "ffmpeg", "unzip"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        # Clone the Hugging Face Space repository into /app (for layout, RIFE code, etc.)
        "git clone https://huggingface.co/spaces/kulkas2pintu/wan555 /app"
    )
    .run_commands(
        # Download and extract the RIFE model weights inside the container
        "cd /app && wget -q https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip -O RIFEv4.26_0921.zip && unzip -o RIFEv4.26_0921.zip && rm RIFEv4.26_0921.zip"
    )
    .pip_install(
        # Install exact versions from requirements.txt to match Space
        "diffusers==0.38.0", "transformers==4.57.6", "accelerate==1.13.0", "safetensors",
        "sentencepiece", "peft==0.19.1", "ftfy", "imageio", "imageio-ffmpeg",
        "opencv-python-headless", "torchao==0.15.0", "numpy==1.23.5", "tqdm", "gradio>=5.0,<6.0"
    )
)

# Fetch secrets from environment or credentials.txt
def get_credentials():
    creds = {"HF_TOKEN": os.environ.get("HF_TOKEN")}
    try:
        with open("credentials.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    creds["HF_TOKEN"] = line.split("=", 1)[1]
                elif line.startswith("hf_"):
                    creds["HF_TOKEN"] = line
    except Exception:
        pass
    return creds

secrets_dict = get_credentials()
secrets = []
if secrets_dict.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": secrets_dict["HF_TOKEN"]}))

# Setup download weights function to pre-cache key model weights
@app.function(
    image=wan_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200
)
def download_all_weights():
    import os
    from huggingface_hub import snapshot_download

    # Log in if token is available (optional for public models)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"[WanModel] Note: Hugging Face token authentication skipped: {e}")

    print("Pre-downloading Wan 2.2 model weights to persistent volume `/root/.cache/huggingface`...")
    try:
        snapshot_download(
            repo_id="TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
            cache_dir="/root/.cache/huggingface"
        )
        print("Cached WAMU_v2_WAN2.2_I2V_LIGHTNING successfully.")
    except Exception as e:
        print(f"ERROR: Could not download model: {e}")


    print("\nPre-download task complete!")

# Main generation execution class running on A10G GPU
@app.cls(
    image=wan_image,
    gpu="A10G",  # A10G (24GB VRAM) is cost-effective with Wan 2.2 model CPU offload
    volumes={"/root/.cache/huggingface": weights_vol, "/app/outputs": outputs_vol},
    secrets=secrets,
    timeout=1800,
    min_containers=0,
    scaledown_window=60,
)
class WanModel:
    @modal.method()
    def heartbeat(self):
        print("[WanModel] Heartbeat received.")
        return "alive"


    @modal.enter()
    def load_pipeline(self):
        import os
        import sys
        sys.path.append("/app")
        import torch
        from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, Int8WeightOnlyConfig
        import torch._dynamo

        # Log in if token is available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token)
            except Exception as e:
                print(f"[WanModel] Note: Hugging Face token login skipped: {e}")

        # 1. Load RIFE model
        print("Loading RIFE model...")
        from train_log.RIFE_HDv3 import Model as RIFEModel
        self.rife_model = RIFEModel()
        self.rife_model.load_model("/app/train_log", -1)
        self.rife_model.eval()

        # 2. Load Wan 2.2 Pipeline on CPU
        print("Loading Wan 2.2 Image-To-Video pipeline on CPU...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING",
            torch_dtype=torch.bfloat16,
            cache_dir="/root/.cache/huggingface"
        )

        # 3. Enable Model CPU Offload
        print("Enabling Model CPU Offload...")
        self.pipe.enable_model_cpu_offload()
            
        self.original_scheduler = self.pipe.scheduler


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
        import time
        import numpy as np
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
        q = queue.Queue()

        def step_callback(pipe_obj, step_idx, timestep, callback_kwargs):
            q.put({"type": "progress", "step": step_idx, "total_steps": steps})
            return callback_kwargs

        def run_inference_thread():
            try:
                print(f"[Remote] Starting generation for seed {seed}...")
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

        # Restore original scheduler
        self.pipe.scheduler = self.original_scheduler

        if raw_frames_np is None:
            yield {"type": "error", "message": "Pipeline did not produce frames", "traceback": ""}
            return

        # 4. Interpolate using RIFE
        frame_factor = frame_multiplier // 16 # FIXED_FPS is 16
        if frame_factor > 1:
            yield {"type": "status", "message": f"Processing frames (RIFE Multiplier: {frame_factor}x)..."}
            final_frames = self.interpolate_bits_internal(raw_frames_np, multiplier=int(frame_factor))
        else:
            final_frames = list(raw_frames_np)

        # 5. Export to video
        import tempfile
        import shutil
        from diffusers.utils.export_utils import export_to_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            video_path = tmpfile.name

        final_fps = 16 * int(frame_factor)
        export_to_video(final_frames, video_path, fps=final_fps, quality=quality)

        # Copy to Modal persistent outputs volume
        persistent_dir = "/app/outputs"
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_path = os.path.join(persistent_dir, f"output_{seed}.mp4")
        try:
            print(f"[Remote] Copying video to persistent volume: {persistent_path}...")
            shutil.copy2(video_path, persistent_path)
            # Commit changes if the volume class has a commit method
            if hasattr(outputs_vol, "commit"):
                outputs_vol.commit()
        except Exception as e:
            print(f"ERROR: Could not copy video to persistent volume: {e}")

        # 6. Read and yield output bytes
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        yield {"type": "output", "video_bytes": video_bytes, "filename": f"output_{seed}.mp4"}

    def interpolate_bits_internal(self, frames_np, multiplier=2, scale=1.0):
        import torch
        import numpy as np
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


