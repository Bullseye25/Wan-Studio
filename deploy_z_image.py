import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: adaab-z-image-turbo
# Provides GPU-accelerated seed-locked character reference sheet generation
# via Tongyi-MAI/Z-Image-Turbo for the Wan 2.2 AI Studio Pipeline.
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("adaab-z-image-turbo")

weights_vol = modal.Volume.from_name("adaab-hf-cache", create_if_missing=True)

z_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install(
        "git", "git-lfs", "libgl1", "libglib2.0-0",
        "build-essential", "ninja-build"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "pip install 'git+https://github.com/huggingface/diffusers.git'"
    )
    .pip_install(
        "transformers", "accelerate", "safetensors", "sentencepiece",
        "huggingface_hub", "numpy", "pillow"
    )
)


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
secrets = []
if hf_token_local:
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token_local}))


@app.function(
    image=z_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=3600,
)
def download_z_image_weights():
    import os
    import huggingface_hub
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"[deploy_z_image] Note: Hugging Face login skipped: {e}")

    print("Pre-downloading Tongyi-MAI/Z-Image-Turbo model weights...")
    snapshot_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        token=hf_token,
        cache_dir="/root/.cache/huggingface",
    )
    weights_vol.commit()
    print("\nZ-Image Turbo weights cached successfully in Modal volume!")


@app.cls(
    image=z_image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=900,
    min_containers=0,
    scaledown_window=60,
)
class ZImageTurboModel:
    @modal.method()
    def heartbeat(self):
        print("[ZImageTurboModel] Heartbeat received.")
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import os
        import torch
        import huggingface_hub
        from diffusers import ZImagePipeline, DiffusionPipeline

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                huggingface_hub.login(token=hf_token)
            except Exception as e:
                print(f"[ZImageTurboModel] Note: Hugging Face login skipped: {e}")

        print("[ZImageTurboModel] Loading Tongyi-MAI/Z-Image-Turbo pipeline...")
        try:
            self.pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                cache_dir="/root/.cache/huggingface",
            )
        except Exception:
            self.pipe = DiffusionPipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                cache_dir="/root/.cache/huggingface",
            )
        self.pipe.to("cuda")
        print("[ZImageTurboModel] Pipeline loaded successfully on GPU.")

    @modal.method()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        seed: int = 42,
    ) -> dict:
        import io
        import base64
        import torch

        generator = torch.Generator("cuda").manual_seed(int(seed))
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64_str, "seed": int(seed)}
