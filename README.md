# 🎨 Wan 2.2 Studio (with FLUX.2 Klein 9B)

A generative AI creative studio combining **FLUX.2 Klein 9B** image editing with **Wan 2.2** Image-to-Video animation on serverless NVIDIA A10G GPUs via [Modal](https://modal.com).

---

## 🌟 Workflow Overview

```
 📷 User Image / Text Prompt
            │
            ▼
 🖼️ Step 1: FLUX.2 Klein 9B Image Studio
    ├── Text-to-Image Generation
    ├── Reference-based Editing (Face-Swap, Pose Alignment)
    └── Canvas Resizing (Cover, Contain, Pad)
            │
            ▼ (1-Click Route)
 🎬 Step 2: Wan 2.2 Image-to-Video Animator
    ├── Diffusion Motion Guidance
    ├── Multi-scheduler support (UniPC, FlowMatch, SASolver, etc.)
    └── RIFE Optical Flow Frame Interpolation (16, 32, 64 FPS)
            │
            ▼
 🎥 Final Rendered Video (.mp4 in ./outputs/)
```

---

## 🚀 Key Features

- **2-Step Seamless Studio**: Generate or edit images in Step 1, then immediately route the output to Step 2 for video animation.
- **Serverless Cloud GPUs**: Backends deploy to Modal on NVIDIA GPUs (A100-80GB for Wan 2.2, A10G 24GB VRAM for FLUX.2 Klein).
- **Cost-Optimized Keepalive**: Targeted model keepalive prevents unneeded GPU containers from running when idle, protecting cloud credits.
- **RIFE Fluidity Interpolation**: Real-time optical flow frame interpolation supporting up to 64 FPS video exports.
- **Gradio 6 Interactive Interface**: Real-time progress tracking, video timestamp scrubbing, and direct frame extraction.

---

## 📂 Project Structure

```
.
├── app.py                  # Gradio 6 Web Application
├── backend.py              # Local Modal RPC client & keepalive manager
├── deploy_wan.py           # Modal backend for Wan 2.2 I2V + RIFE (A100-80GB GPU)
├── deploy_flux_klein.py    # Modal backend for FLUX.2 Klein 9B (A10G GPU)
├── start.py                # Local runner & management CLI
├── tests/                  # Comprehensive unit & E2E test suite
│   ├── test_suite.py
│   ├── test_backend_contract.py
│   └── test_e2e_pipeline.py
├── test_bus.png            # Sample test asset
└── README.md
```

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- A [Modal](https://modal.com) account
- A [Hugging Face](https://huggingface.co) account with an active User Access Token

### 2. Setup Credentials
Create a `credentials.txt` file in the root directory:
```text
HF_TOKEN=hf_YourHuggingFaceTokenHere
```
*(Ensure you have accepted the model license agreement for [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) on Hugging Face).*

### 3. Launch Runner
```bash
python start.py
```

The interactive menu will let you:
1. **Option 1**: Start Wan 2.2 Studio frontend locally (`http://localhost:7860`).
2. **Option 2**: Deploy Wan 2.2 backend to Modal.
3. **Option 3**: Deploy FLUX.2 Klein backend to Modal.
4. **Option 4**: Pre-cache Wan 2.2 model weights on Modal.
5. **Option 5**: Pre-cache FLUX.2 Klein weights on Modal.
6. **Option 6**: Start on Local Network for mobile/tablet access.
7. **Option 7**: Setup / authenticate Modal CLI (`modal setup`).

---

## 🧪 Testing

Run the automated test suite:
```bash
python -m pytest tests/ -v
```

All 21 tests validate image aspect ratio clamping, frame calculations, LoRA-free API contracts, and the end-to-end Picture $\rightarrow$ Modify $\rightarrow$ Animate pipeline.

---

## 📄 License
Apache-2.0 / MIT.
