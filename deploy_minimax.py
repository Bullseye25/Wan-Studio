import os
import io
import time
import modal

# Setup Modal app
app = modal.App("minimax-backend")

# Define Modal Volumes for caching HF models (all weights live in user's persistent volume)
weights_vol = modal.Volume.from_name("wan-cache", create_if_missing=True)

# Build self-contained Modal Image with CUDA-enabled PyTorch, diffusers (MiniMax-H3 commit), av, and dependencies
minimax_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "H3_MODEL_REPO": "multimodalart/MiniMax-H3-Pruned",
        "H3_CONDITIONER_REPO": "MiniMaxAI/MiniMax-H3",
        "H3_WIDTH": "bf16-pruned",
        "H3_PLACEMENT": "pack",
        "H3_ATTENTION": "_native_cudnn",
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
        "huggingface-hub==1.24.0",
        "transformers==5.8.0",
        "accelerate==1.14.0",
        "safetensors>=0.8.0",
        "av", "pillow", "numpy", "requests", "gradio_client",
        "git+https://github.com/huggingface/diffusers.git@665f578278365ea4a3318cb8c9b66ce6c01204b9",
    )
    .add_local_file("h3_split_blocks.py", remote_path="/root/h3_split_blocks.py")
    .add_local_file("prompt_rewrite.py", remote_path="/root/prompt_rewrite.py")
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


# ─── Self-Contained Pre-cache Function ───────────────────────────────────────

@app.function(
    image=minimax_image,
    volumes={"/root/.cache/huggingface": weights_vol},
    secrets=secrets,
    timeout=7200,
)
def download_minimax_weights():
    """Downloads and caches ALL MiniMax-H3 components (conditioner + generator) into the persistent Modal volume."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import huggingface_hub
    from diffusers import ComponentsManager
    from h3_split_blocks import MiniMaxH3GeneratorBlocks, MiniMaxH3ConditionerBlocks

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            print(f"HF login skipped: {e}")

    cond_repo = os.environ.get("H3_CONDITIONER_REPO", "MiniMaxAI/MiniMax-H3")
    gen_repo = os.environ.get("H3_MODEL_REPO", "multimodalart/MiniMax-H3-Pruned")

    print(f"1. Pre-caching MiniMax-H3 Conditioner (Text Encoder) from {cond_repo}...")
    cond_blocks = MiniMaxH3ConditionerBlocks()
    cond_pipe = cond_blocks.init_pipeline(cond_repo)
    cond_pipe.load_components(dtype=torch.bfloat16)

    print(f"2. Pre-caching MiniMax-H3 Generator (DiT Transformer + VAEs) from {gen_repo}...")
    manager = ComponentsManager()
    gen_blocks = MiniMaxH3GeneratorBlocks()
    gen_pipe = gen_blocks.init_pipeline(gen_repo, components_manager=manager, collection="h3")
    gen_pipe.load_components(dtype=torch.bfloat16, trust_remote_code=True)

    weights_vol.commit()
    print("\n✓ Full MiniMax-H3 model suite (Conditioner + Generator) cached successfully in Modal storage volume!")


# ─── Class Definition (100% Self-Contained on A100 GPU) ──────────────────────

@app.cls(
    image=minimax_image,
    gpu="A100-80GB",
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
        from h3_split_blocks import MiniMaxH3GeneratorBlocks, MiniMaxH3ConditionerBlocks

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                huggingface_hub.login(token=hf_token)
            except Exception as e:
                print(f"[MiniMaxH3Model] Note: Hugging Face login skipped: {e}")

        # Lower duration floor to 2s
        MiniMaxH3ModularPipeline.min_duration = property(lambda self: 2.0)

        cond_repo = os.environ.get("H3_CONDITIONER_REPO", "MiniMaxAI/MiniMax-H3")
        gen_repo = os.environ.get("H3_MODEL_REPO", "multimodalart/MiniMax-H3-Pruned")

        print(f"[MiniMaxH3Model] Loading local Conditioner (Text Encoder) from {cond_repo}...")
        cond_blocks = MiniMaxH3ConditionerBlocks()
        cond_pipe = cond_blocks.init_pipeline(cond_repo)
        cond_pipe.load_components(dtype=torch.bfloat16)

        print(f"[MiniMaxH3Model] Loading local Generator (Transformer + VAEs) from {gen_repo}...")
        manager = ComponentsManager()
        gen_blocks = MiniMaxH3GeneratorBlocks()
        gen_pipe = gen_blocks.init_pipeline(gen_repo, components_manager=manager, collection="h3")
        gen_pipe.load_components(dtype=torch.bfloat16, trust_remote_code=True)

        # Set attention backend
        attention_backend = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
        try:
            gen_pipe.transformer.set_attention_backend(attention_backend)
        except Exception as e:
            print(f"[MiniMaxH3Model] Attention backend set exception (falling back to native): {e}")

        self.cond_pipe = cond_pipe
        self.gen_pipe = gen_pipe
        self.manager = manager
        print("[MiniMaxH3Model] Full self-contained MiniMax-H3 pipeline loaded on Modal!")

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
        from diffusers.utils import encode_video

        # Clean VRAM before generation
        gc.collect()
        torch.cuda.empty_cache()

        height, width = CANVASES.get(canvas, (544, 960))
        num_frames = snap_frames(duration)

        print(f"[MiniMaxH3Model] Starting generation: prompt='{prompt[:50]}...', canvas={canvas}, duration={duration}s ({num_frames} frames), steps={steps}, seed={seed}")

        first_pil = Image.open(io.BytesIO(first_image_bytes)).convert("RGB") if first_image_bytes else None
        last_pil = Image.open(io.BytesIO(last_image_bytes)).convert("RGB") if last_image_bytes else None

        keyframe_first = ImageOps.exif_transpose(first_pil).convert("RGB") if first_pil else None
        keyframe_last = ImageOps.exif_transpose(last_pil).convert("RGB") if last_pil else None

        # 1. Local prompt upsampling & conditioning on Modal A100 GPU
        refined_prompt = ""
        final_prompt = prompt

        print("[MiniMaxH3Model] Moving text encoder to A100 GPU for conditioning...")
        self.cond_pipe.to("cuda")

        if upsample:
            try:
                import prompt_rewrite
                print("[MiniMaxH3Model] Rewriting/enhancing prompt locally with Qwen3-VL...")
                refined_prompt = prompt_rewrite.refine_keyframe_prompt(
                    self.cond_pipe, prompt, keyframe_first, keyframe_last, num_frames, 1000
                )
                if refined_prompt:
                    final_prompt = refined_prompt
                    print(f"[MiniMaxH3Model] Enhanced prompt generated locally:\n{final_prompt}")
            except Exception as e:
                print(f"[MiniMaxH3Model] Prompt enhancement warning: {e}, continuing with raw prompt")

        print("[MiniMaxH3Model] Encoding prompt & keyframes on A100...")
        cond_state = self.cond_pipe(
            prompt=final_prompt,
            image=keyframe_first,
            last_image=keyframe_last,
            height=int(height),
            width=int(width),
        )
        prompt_embeds = cond_state.get("prompt_embeds")
        text_token_tags = cond_state.get("text_token_tags")

        # Offload text encoder from GPU to CPU to give 100% of 80GB VRAM to the 33B DiT transformer
        self.cond_pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Local Denoising on A100 GPU
        print(f"[MiniMaxH3Model] Loading DiT transformer & VAEs to A100 GPU...")
        self.gen_pipe.transformer.to("cuda")
        self.gen_pipe.vae.to("cuda")
        self.gen_pipe.audio_vae.to("cuda")

        print(f"[MiniMaxH3Model] Denoising {steps} steps at {width}x{height}, {num_frames} frames...")
        start_gen = time.time()

        state = self.gen_pipe(
            prompt_embeds=prompt_embeds.to("cuda"),
            text_token_tags=text_token_tags,
            image=keyframe_first,
            last_image=keyframe_last,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        )

        frames = state.get("videos")[0]
        audio = state.get("audio")[0].cpu()
        sampling_rate = state.get("sampling_rate")
        gen_time = time.time() - start_gen

        # Offload transformer and VAEs back to CPU to conserve GPU VRAM
        self.gen_pipe.transformer.to("cpu")
        self.gen_pipe.vae.to("cpu")
        self.gen_pipe.audio_vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Mux video and synchronized soundtrack into MP4
        output_temp_path = os.path.join(tempfile.gettempdir(), f"minimax_{int(time.time() * 1000)}_{seed}.mp4")
        encode_video(frames, fps=FPS, output_path=output_temp_path, audio=audio, audio_sample_rate=sampling_rate)

        with open(output_temp_path, "rb") as f:
            video_bytes = f.read()

        if os.path.exists(output_temp_path):
            try:
                os.remove(output_temp_path)
            except Exception:
                pass

        # Cleanup variables
        del state, frames, audio
        gc.collect()
        torch.cuda.empty_cache()

        report = (
            f"`{width}x{height}`, {num_frames} frames ({num_frames / FPS:.2f}s), {int(steps)} steps · "
            f"Denoise + Audio Decode: {gen_time:.1f}s ({gen_time / max(1, int(steps)):.1f}s/step) · Seed: {int(seed)}"
        )
        print(f"[MiniMaxH3Model] Complete! {report}")

        return {
            "video_bytes": video_bytes,
            "report": report,
            "refined_prompt": refined_prompt,
            "filename": f"minimax_{int(time.time())}_{seed}.mp4",
        }
