import os
import io
import time
import modal

# Setup Modal app
app = modal.App("minimax-backend")

# Define Modal Volumes for caching HF models
weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)

# Build Modal Image with CUDA-enabled PyTorch, diffusers (MiniMax-H3 commit), av, and dependencies
minimax_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "H3_MODEL_REPO": "multimodalart/MiniMax-H3-Pruned",
        "H3_WIDTH": "bf16-pruned",
        "H3_PLACEMENT": "pack",
        "H3_ATTENTION": "_native_cudnn",
        "H3_CONDITIONER": "multimodalart/qwen3vl-conditioner",
    })
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "build-essential", "ninja-build",
        "libjpeg-dev", "wget", "ffmpeg"
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git@665f578278365ea4a3318cb8c9b66ce6c01204b9",
        "transformers==4.49.0", "accelerate>=1.1.0", "safetensors>=0.4.0",
        "av", "pillow", "numpy", "requests", "gradio_client", "huggingface_hub"
    )
    .add_local_file("h3_split_blocks.py", remote_path="/root/h3_split_blocks.py")
)

secrets = []
if "HF_TOKEN" in os.environ:
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))


CANVASES = {
    # 16:9
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    # 9:16
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    # 1:1
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    "1024x1024 · 1:1 max": (1024, 1024),
    # 4:3 / 3:4
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    # 21:9
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5


def snap_frames(seconds: float) -> int:
    """The frame count MiniMax-H3's video VAE can decode: the next `17 * n + 5` at 24 fps."""
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


# ─── Pre-cache Function ───────────────────────────────────────────────────────

@app.function(
    image=minimax_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=3600,
)
def download_minimax_weights():
    """Downloads and caches MiniMax-H3-Pruned components into the persistent Modal volume."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import huggingface_hub
    from diffusers import ComponentsManager
    from h3_split_blocks import MiniMaxH3GeneratorBlocks

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)

    model_repo = os.environ.get("H3_MODEL_REPO", "multimodalart/MiniMax-H3-Pruned")
    print(f"Pre-caching MiniMax-H3 model weights from {model_repo}...")

    manager = ComponentsManager()
    blocks = MiniMaxH3GeneratorBlocks()
    pipe = blocks.init_pipeline(model_repo, components_manager=manager, collection="h3")
    pipe.load_components(dtype=torch.bfloat16, trust_remote_code=True)

    weights_vol.commit()
    print("\nMiniMax-H3 model weights cached successfully!")


# ─── Class Definition ─────────────────────────────────────────────────────────

@app.cls(
    image=minimax_image,
    gpu="A100-80GB",  # 33B parameters (unquantized bfloat16 + float32 VAEs) runs on 80GB VRAM
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=1800,
    min_containers=0,
    scaledown_window=60,
)
class MiniMaxH3Model:
    @modal.method()
    def heartbeat(self):
        print("[MiniMaxH3Model] Heartbeat received.")
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import sys
        sys.path.insert(0, "/root")
        import torch
        import huggingface_hub
        from diffusers import ComponentsManager
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                huggingface_hub.login(token=hf_token)
            except Exception as e:
                print(f"[MiniMaxH3Model] Note: Hugging Face login skipped: {e}")

        # Lower duration floor to 2s
        MiniMaxH3ModularPipeline.min_duration = property(lambda self: 2.0)

        model_repo = os.environ.get("H3_MODEL_REPO", "multimodalart/MiniMax-H3-Pruned")
        print(f"[MiniMaxH3Model] Loading MiniMax-H3 components from {model_repo}...")

        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        pipe = blocks.init_pipeline(model_repo, components_manager=manager, collection="h3")
        pipe.load_components(dtype=torch.bfloat16, trust_remote_code=True)

        # Attention backend
        attention_backend = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
        try:
            pipe.transformer.set_attention_backend(attention_backend)
        except Exception as e:
            print(f"[MiniMaxH3Model] Attention backend set exception (falling back to native): {e}")

        # Place transformer and VAEs on GPU
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.audio_vae.to("cuda")

        self.pipe = pipe
        self.manager = manager
        print("[MiniMaxH3Model] Pipeline loaded successfully on A100 GPU!")

    @modal.method()
    def generate(
        self,
        prompt: str,
        first_image_bytes: bytes = None,
        last_image_bytes: bytes = None,
        canvas: str = DEFAULT_CANVAS,
        duration: int = 5,
        steps: int = 28,
        seed: int = 42,
        upsample: bool = False,
    ) -> dict:
        import sys
        sys.path.insert(0, "/root")
        import gc
        import tempfile
        import time
        import torch
        from PIL import Image, ImageOps
        from gradio_client import Client, handle_file
        from safetensors import safe_open
        from diffusers.utils import encode_video

        # Clean VRAM before generation
        gc.collect()
        torch.cuda.empty_cache()

        conditioner_space = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
        num_frames = snap_frames(duration)

        print(f"[MiniMaxH3Model] Starting generation: prompt='{prompt[:50]}...', canvas={canvas}, duration={duration}s ({num_frames} frames), steps={steps}, seed={seed}")

        # Write input images to temp files if provided for the conditioner
        temp_first_path = None
        temp_last_path = None
        first_pil = None
        last_pil = None

        if first_image_bytes:
            first_pil = Image.open(io.BytesIO(first_image_bytes)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_first_path = f.name
                first_pil.save(temp_first_path, format="PNG")

        if last_image_bytes:
            last_pil = Image.open(io.BytesIO(last_image_bytes)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_last_path = f.name
                last_pil.save(temp_last_path, format="PNG")

        # 1. Conditioning over Gradio Client
        print(f"[MiniMaxH3Model] Requesting conditioning from {conditioner_space}...")
        client = Client(conditioner_space)
        encoded_path, plan = client.predict(
            prompt=prompt,
            image_path=handle_file(temp_first_path) if temp_first_path else None,
            last_image_path=handle_file(temp_last_path) if temp_last_path else None,
            canvas=canvas,
            num_frames=num_frames,
            rewrite_prompt=bool(upsample),
            api_name="/encode",
        )

        with safe_open(encoded_path, framework="pt") as handle:
            metadata = handle.metadata()
            prompt_embeds = handle.get_tensor("prompt_embeds")
            text_token_tags = handle.get_tensor("text_token_tags")

        height = int(metadata.get("height", CANVASES.get(canvas, (544, 960))[0]))
        width = int(metadata.get("width", CANVASES.get(canvas, (544, 960))[1]))
        resolved_num_frames = int(metadata.get("num_frames", num_frames))
        refined_prompt = plan.get("refined_prompt") or ""

        # Prepare keyframes
        keyframe_first = ImageOps.exif_transpose(first_pil).convert("RGB") if first_pil else None
        keyframe_last = ImageOps.exif_transpose(last_pil).convert("RGB") if last_pil else None

        # 2. Denoise loop on A100 GPU
        print(f"[MiniMaxH3Model] Denoising {steps} steps at {width}x{height}, {resolved_num_frames} frames...")
        start_gen = time.time()
        
        state = self.pipe(
            prompt_embeds=prompt_embeds.to("cuda"),
            text_token_tags=text_token_tags,
            image=keyframe_first,
            last_image=keyframe_last,
            height=height,
            width=width,
            num_frames=resolved_num_frames,
            num_inference_steps=int(steps),
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        )

        frames = state.get("videos")[0]
        audio = state.get("audio")[0].cpu()
        sampling_rate = state.get("sampling_rate")
        gen_time = time.time() - start_gen

        # 3. Mux video and synchronized soundtrack into MP4
        output_temp_path = os.path.join(tempfile.gettempdir(), f"minimax_{int(time.time() * 1000)}_{seed}.mp4")
        encode_video(frames, fps=FPS, output_path=output_temp_path, audio=audio, audio_sample_rate=sampling_rate)

        with open(output_temp_path, "rb") as f:
            video_bytes = f.read()

        # Clean up temp files
        for p in [temp_first_path, temp_last_path, output_temp_path, encoded_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        # Cleanup GPU memory
        del state, frames, audio
        gc.collect()
        torch.cuda.empty_cache()

        report = (
            f"`{width}x{height}`, {resolved_num_frames} frames ({resolved_num_frames / FPS:.2f}s), {int(steps)} steps · "
            f"Denoise + Audio Decode: {gen_time:.1f}s ({gen_time / max(1, int(steps)):.1f}s/step) · Seed: {int(seed)}"
        )
        print(f"[MiniMaxH3Model] Complete! {report}")

        return {
            "video_bytes": video_bytes,
            "report": report,
            "refined_prompt": refined_prompt,
            "filename": f"minimax_{int(time.time())}_{seed}.mp4",
        }
