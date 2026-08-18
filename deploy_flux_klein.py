import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: flux-klein-backend
# Provides GPU-accelerated image editing and generation via FLUX.2 Klein 9B
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("flux-klein-backend")

weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("wan-outputs", create_if_missing=True)

# Build image with Python 3.12, CUDA 12.4
# We clone the Hugging Face space inside the container so all local LoRAs are available
flux_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "git-lfs", "libgl1", "libglib2.0-0",
        "build-essential", "ninja-build", "ffmpeg"
    )
    .run_commands(
        "git lfs install",
        "git clone https://huggingface.co/spaces/M3st3rJ4k3l/FLUX.2-Klein-Multi-LoRA /flux"
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

# Build the secrets list from credentials.txt token.
# If no token found, deploy continues but gated model access will fail at runtime.
secrets = []
if hf_token_local:
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token_local}))
else:
    print(
        "[deploy_flux_klein] WARNING: No HF_TOKEN found in credentials.txt.\n"
        "  The FLUX.2 Klein 9B model is gated and requires authentication.\n"
        "  Fix: create a file at /Users/raza/Documents/Wan/credentials.txt containing:\n"
        "       HF_TOKEN=hf_your_token_here\n"
        "  Then redeploy (option 3) and re-cache weights (option 5).\n"
        "  Get your token from: https://huggingface.co/settings/tokens"
    )

# ─── Weight Caching ───────────────────────────────────────────────────────────
@app.function(
    image=flux_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_flux_weights():
    import os
    import huggingface_hub
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Cannot download gated model.\n"
            "Fix: create credentials.txt with HF_TOKEN=hf_... next to deploy_flux_klein.py, OR run:\n"
            "     modal secret create huggingface-secret HF_TOKEN=hf_your_token_here"
        )

    # Login AND set the env var so every huggingface_hub call picks it up
    try:
        huggingface_hub.login(token=hf_token)
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face authentication failed with the token in credentials.txt ({e}).\n"
            f"Please update credentials.txt with a valid HF Token from https://huggingface.co/settings/tokens\n"
            f"and make sure you accepted the model agreement at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
        ) from e

    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HUB_TOKEN"] = hf_token

    print("Pre-downloading FLUX.2-klein-9B base model...")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        token=hf_token,
        cache_dir="/root/.cache/huggingface"
    )
    print("\nFLUX.2 Klein weights cached successfully!")


# ─── Class Definition ─────────────────────────────────────────────────────────

@app.cls(
    image=flux_image,
    gpu="A10G",  # 9B parameters (~18GB) fits in 24GB VRAM at ~1/4 the cost of A100
    volumes={
        "/root/.cache/huggingface": weights_vol,
        "/outputs": outputs_vol,
    },
    secrets=secrets,
    timeout=1800,
    min_containers=0,
    scaledown_window=60,
)
class FluxKleinModel:
    @modal.method()
    def heartbeat(self):
        print("[FluxKleinModel] Heartbeat received.")
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import sys
        import os
        import torch
        import huggingface_hub

        # Add cloned Space to Python path
        sys.path.insert(0, "/flux")

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set in the Modal container. "
                "Cannot load gated model black-forest-labs/FLUX.2-klein-9B.\n"
                "Fix: create credentials.txt with HF_TOKEN=hf_... next to deploy_flux_klein.py, OR run:\n"
                "     modal secret create huggingface-secret HF_TOKEN=hf_your_token_here"
            )

        # Login AND expose token via all env vars HF libraries check
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            raise RuntimeError(
                f"Hugging Face authentication failed in FluxKleinModel: {e}.\n"
                f"Please update credentials.txt with a valid HF Token from https://huggingface.co/settings/tokens\n"
                f"and make sure you accepted the model agreement at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
            ) from e

        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_HUB_TOKEN"] = hf_token

        from diffusers import Flux2KleinPipeline

        dtype = torch.bfloat16
        device = torch.device("cuda")

        print("[FluxKleinModel] Loading FLUX.2 Klein 9B pipeline...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=dtype,
            token=hf_token,
            cache_dir="/root/.cache/huggingface",
        ).to(device)

        print("[FluxKleinModel] Pipeline loaded successfully.")

    @modal.method()
    def generate(
        self,
        base_image_bytes: bytes,
        ref_images_bytes: list[bytes],
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
        import sys
        import os
        import io
        import gc
        import random
        import torch
        from PIL import Image

        # Import helper utilities from the cloned space
        sys.path.insert(0, "/flux")
        from image_utils import fit_to_canvas, compute_canvas_dimensions

        # Warm-up clean
        gc.collect()
        torch.cuda.empty_cache()

        # Load input images
        base_pil = Image.open(io.BytesIO(base_image_bytes)).convert("RGB")
        ref_pils = [Image.open(io.BytesIO(b)).convert("RGB") for b in ref_images_bytes]
        pil_images = [base_pil] + ref_pils

        # Compute canvas size
        width, height = compute_canvas_dimensions(pil_images[0], canvas_mode, custom_width, custom_height)
        processed = [fit_to_canvas(img, width, height, canvas_fit_mode, pad_color) for img in pil_images]
        image_input = processed if len(processed) > 1 else processed[0]

        actual_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)

        print(f"[FluxKleinModel] Running inference (seed={actual_seed}, steps={steps}, size={width}x{height})...")
        result = self.pipe(
            image=image_input,
            prompt=prompt,
            negative_prompt_embeds=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Save copy to Modal volume output folder
        import pathlib
        import time
        try:
            modal_output_dir = pathlib.Path("/outputs/output")
            modal_output_dir.mkdir(parents=True, exist_ok=True)
            
            file_seed = seed if seed >= 0 else actual_seed
            filename = f"flux_{int(time.time())}_{file_seed}.jpg"
            modal_path = modal_output_dir / filename
            
            result.save(modal_path, format="JPEG", quality=95)
            print(f"[FluxKleinModel] Saved copy to Modal Volume: {modal_path}")
            
            # Commit the written file to persistent storage
            outputs_vol.commit()
        except Exception as e:
            print(f"[FluxKleinModel] Warning: Could not save copy to Modal Volume: {e}")

        out_buf = io.BytesIO()
        result.save(out_buf, format="JPEG", quality=95)
        return out_buf.getvalue()
