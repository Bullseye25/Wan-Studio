import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: qwen-image-backend
# Provides GPU-accelerated image editing and generation via Qwen Image Edit Rapid AIO
# Cloned from: https://huggingface.co/spaces/davidharryv/qwen-image-edit-rapid-aio-nsfw-static
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("qwen-image-backend")

weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("wan-outputs", create_if_missing=True)

# Build image with Python 3.12, CUDA 12.4
qwen_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install(
        "git", "git-lfs", "libgl1", "libglib2.0-0",
        "build-essential", "ninja-build", "ffmpeg"
    )
    .run_commands(
        "git lfs install",
        "git clone https://huggingface.co/spaces/davidharryv/qwen-image-edit-rapid-aio-nsfw-static /qwen"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "pip install 'git+https://github.com/huggingface/diffusers.git' "
        "'git+https://github.com/huggingface/peft.git'"
    )
    .pip_install(
        "transformers", "accelerate", "safetensors", "sentencepiece",
        "huggingface_hub", "av", "numpy", "pillow", "gradio>=5.0,<7.0",
        "pillow-heif", "controlnet-aux"
    )
)

# ─── Credentials ──────────────────────────────────────────────────────────────
def get_credentials():
    """Read HF_TOKEN from credentials.txt (local file next to this script)."""
    token = os.environ.get("HF_TOKEN")
    try:
        creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.txt")
        with open(creds_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("hf_"):
                    token = line.strip()
    except Exception:
        pass
    return token

hf_token_local = get_credentials()

# Build the secrets list from credentials.txt token if present.
secrets = []
if hf_token_local:
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token_local}))


# ─── Weight Caching ───────────────────────────────────────────────────────────
@app.function(
    image=qwen_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_qwen_weights():
    import os
    import huggingface_hub
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            huggingface_hub.login(token=hf_token)
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HF_HUB_TOKEN"] = hf_token
        except Exception as e:
            print(f"[deploy_qwen] Note: Hugging Face login skipped: {e}")

    print("Pre-downloading Qwen Image Edit model weights...")
    for model_id in ["Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit"]:
        try:
            print(f"Downloading {model_id}...")
            snapshot_download(
                repo_id=model_id,
                token=hf_token,
                cache_dir="/root/.cache/huggingface"
            )
            print(f"Cached {model_id} successfully!")
            break
        except Exception as e:
            print(f"Attempt for {model_id} failed: {e}. Trying fallback...")

    print("\nQwen Image Edit weights cached successfully!")


# ─── Class Definition ─────────────────────────────────────────────────────────

@app.cls(
    image=qwen_image,
    gpu="A100-80GB",  # 20B parameters (~40GB in bf16) runs comfortably and rapidly on 80GB VRAM
    volumes={
        "/root/.cache/huggingface": weights_vol,
        "/outputs": outputs_vol,
    },
    secrets=secrets,
    timeout=1800,
    min_containers=0,
    scaledown_window=60,
)
class QwenImageEditModel:
    @modal.method()
    def heartbeat(self):
        print("[QwenImageEditModel] Heartbeat received.")
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import sys
        import os
        import torch
        import huggingface_hub

        # Add cloned Space to Python path
        sys.path.insert(0, "/qwen")

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                huggingface_hub.login(token=hf_token)
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
                os.environ["HF_HUB_TOKEN"] = hf_token
            except Exception as e:
                print(f"[QwenImageEditModel] Note: Hugging Face login skipped: {e}")

        from diffusers import QwenImageEditPlusPipeline, DiffusionPipeline

        dtype = torch.bfloat16

        print("[QwenImageEditModel] Loading Qwen Image Edit pipeline...")
        pipe = None
        for model_id in ["Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit"]:
            try:
                print(f"Trying to load {model_id}...")
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    token=hf_token,
                    cache_dir="/root/.cache/huggingface",
                )
                print(f"Loaded {model_id} successfully!")
                break
            except Exception as e:
                print(f"Failed to load {model_id} with QwenImageEditPlusPipeline: {e}")
                try:
                    pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        token=hf_token,
                        cache_dir="/root/.cache/huggingface",
                    )
                    break
                except Exception as e2:
                    print(f"Failed fallback load {model_id}: {e2}")

        if pipe is None:
            raise RuntimeError("Could not load any Qwen Image Edit pipeline model.")

        self.pipe = pipe
        self.pipe.enable_model_cpu_offload()

        print("[QwenImageEditModel] Pipeline loaded successfully with CPU offloading.")

    @modal.method()
    def generate(
        self,
        base_image_bytes: bytes,
        ref_images_bytes: list[bytes],
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        steps: int = 20,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        num_images: int = 1,
        custom_width: int = 1024,
        custom_height: int = 1024,
    ) -> list[bytes]:
        import io
        import gc
        import random
        import pathlib
        import time
        import inspect
        import torch
        from PIL import Image

        # Warm-up clean
        gc.collect()
        torch.cuda.empty_cache()

        # Load input images
        base_pil = Image.open(io.BytesIO(base_image_bytes)).convert("RGB")
        ref_pils = [Image.open(io.BytesIO(b)).convert("RGB") for b in ref_images_bytes]
        images_input = [base_pil] + ref_pils if ref_pils else base_pil

        count = max(1, min(6, int(num_images)))
        base_seed = random.randint(0, 2**31 - 1 - count) if seed < 0 else seed
        
        output_bytes_list = []
        sig = inspect.signature(self.pipe.__call__)

        print(f"[QwenImageEditModel] Generating {count} image(s) (base_seed={base_seed}, steps={steps}, prompt='{prompt[:50]}...')...")

        for i in range(count):
            current_seed = base_seed + i
            generator = torch.Generator(device="cuda").manual_seed(current_seed)

            pipe_kwargs = {
                "image": images_input,
                "prompt": prompt,
                "num_inference_steps": int(steps),
                "generator": generator,
            }

            if "negative_prompt" in sig.parameters:
                pipe_kwargs["negative_prompt"] = negative_prompt if negative_prompt else ""
            if "true_cfg_scale" in sig.parameters:
                pipe_kwargs["true_cfg_scale"] = float(true_cfg_scale)
            if "guidance_scale" in sig.parameters:
                pipe_kwargs["guidance_scale"] = float(guidance_scale)
            if "width" in sig.parameters and custom_width > 0:
                pipe_kwargs["width"] = int(custom_width)
            if "height" in sig.parameters and custom_height > 0:
                pipe_kwargs["height"] = int(custom_height)

            with torch.inference_mode():
                result = self.pipe(**pipe_kwargs).images[0]

            # Save copy to Modal volume output folder
            try:
                modal_output_dir = pathlib.Path("/outputs/output")
                modal_output_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"qwen_{int(time.time())}_{current_seed}_{i+1}.jpg" if count > 1 else f"qwen_{int(time.time())}_{current_seed}.jpg"
                modal_path = modal_output_dir / filename
                
                result.save(modal_path, format="JPEG", quality=95)
                print(f"[QwenImageEditModel] Saved copy {i+1}/{count} to Modal Volume: {modal_path}")
            except Exception as e:
                print(f"[QwenImageEditModel] Warning: Could not save copy to Modal Volume: {e}")

            out_buf = io.BytesIO()
            result.save(out_buf, format="JPEG", quality=95)
            output_bytes_list.append(out_buf.getvalue())

        try:
            outputs_vol.commit()
        except Exception:
            pass

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return output_bytes_list
