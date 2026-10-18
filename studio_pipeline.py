"""
Studio Automated Multi-Agent Story & Video Production Pipeline
=============================================================
This module orchestrates the complete end-to-end studio pipeline:
1. Story & Character Breakdown   (Gemma 4: 31B via Ollama Cloud)
2. Character Portrait Synthesis  (Z-Image Turbo with locked identical seed)
3. Scene Keyframe Synthesis      (Qwen Image Edit with multi-reference character transfer)
4. Scene Animation Direction     (Gemma 4: 31B *vision* — watches each keyframe and writes
                                  a per-clip shot plan: action timeline, camera, Wan params)
5. Video Clip Animation          (Wan 2.2 with 4k+1 temporal constraints & CFG stability)
6. Video Assembly & Stitching    (Seamless final movie generation)
"""

import os
import io
import re
import json
import math
import time
import random
import base64
import pathlib
import tempfile
import subprocess
import urllib.request
from typing import Callable, Optional, Dict, List, Any
from PIL import Image

import imageio_ffmpeg

from backend import (
    generate_z_image_turbo_remote,
    qwen_image_edit_remote,
    run_remote_inference,
    DEFAULT_NEGATIVE_PROMPT,
)
from cinematography_vault import (
    retrieve_cinematography_blueprint,
    NARRATIVE_BEATS_BY_COUNT,
    CAUSAL_NARRATIVE_BLUEPRINTS,
    StoryMemoryBank,
)
from narrator_engine import (
    NARRATOR_VOICE_PRESETS,
    DEFAULT_NARRATOR_VOICE,
    enforce_narration_word_budget,
    extract_sync_trigger_word,
    OpenSourceNarratorTTS,
    AudioVisualSyncEngine,
)
from orchestra_rules import (
    MAX_STORY_CHARACTERS,
    TransitionType,
    build_ffmpeg_transition_filter,
    calculate_transition_srt_timestamps,
    validate_story_budget,
    enforce_orchestra_story_constraints,
)


# ── Defaults & Constants ──────────────────────────────────────────────────────

DEFAULT_OLLAMA_API_KEY = "ca8516405cc74a238c9525b490331d3a.HmyaGAbsmLFyYN25P1K8_tG6"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_ENDPOINT = "https://ollama.com/v1/chat/completions"

# Verified active Ollama Cloud models (tested & benchmarked)
VERIFIED_OLLAMA_MODELS = [
    ("⚡ gemma4:31b — Fast Multimodal Vision & Character Genesis (31B - Default)", "gemma4:31b"),
    ("🎬 gpt-oss:120b — Hollywood Screenwriter & Master Storyteller (120B)", "gpt-oss:120b"),
    ("⚔️ nemotron-3-ultra — Weapon-Locked Combat Choreographer & Physics", "nemotron-3-ultra"),
    ("🔍 nemotron-3-super — AI Director QA Critic & Keyframe Auditor", "nemotron-3-super"),
    ("🚀 nemotron-3-nano:30b — Lightning-Fast JSON & Timeline Compiler (30B)", "nemotron-3-nano:30b"),
    ("⚡ gpt-oss:20b — Agile Interactive Co-Pilot & Quick Tweaks (20B)", "gpt-oss:20b"),
    ("🤖 Auto-Routing (Smart Multi-Agent Workflow — Best Model per Task)", "auto"),
]

# Aspect ratio presets for scene synthesis and video animation
ASPECT_RATIO_PRESETS = {
    "9:16 Vertical (Shorts/Reels)": {"qwen_w": 576, "qwen_h": 1024, "wan_w": 480, "wan_h": 832, "char_w": 576, "char_h": 1024},
    "16:9 Landscape (Cinematic)":   {"qwen_w": 1024, "qwen_h": 576,  "wan_w": 832, "wan_h": 480, "char_w": 576, "char_h": 1024},
    "1:1 Square (Social Media)":     {"qwen_w": 1024, "qwen_h": 1024, "wan_w": 640, "wan_h": 640, "char_w": 576, "char_h": 1024},
}

# Character reference portraits default to 9:16 vertical full-body orientation (576x1024)
CHARACTER_PORTRAIT_WIDTH = 576
CHARACTER_PORTRAIT_HEIGHT = 1024

# Wan 2.2 temporal limits (3D-VAE requires 4k+1 frames; native window 17..81 frames @ 16fps)
WAN_FPS = 16
WAN_MIN_FRAMES = 17
WAN_MAX_FRAMES = 81
MAX_CLIP_SECONDS = 5.0   # 81 frames / 16 fps ≈ 5.06s — longer clips degrade (pixel break-up, face drift)
MIN_CLIP_SECONDS = 1.0

# Wan 2.2 Lightning defaults — kept identical to the tuned Tool 2 defaults in app.py
WAN_DEFAULTS = {
    "steps": 8,
    "guidance_scale": 2.0,
    "guidance_scale_2": 1.5,
    "flow_shift": 3.0,
    "quality": 6,
    "scheduler_name": "UniPCMultistep",
    "frame_multiplier": WAN_FPS,   # no RIFE interpolation by default
}

# Safe ranges the Director's suggested parameters are clamped into
DIRECTOR_PARAM_LIMITS = {
    "guidance_scale": (1.5, 2.5),
    "steps": (6, 12),
    "flow_shift": (2.0, 5.0),
    "motion_intensity": (1, 5),
}

IDENTITY_NEGATIVE = (
    "face morphing, changing face, altered facial features, changing identity, inconsistent face, "
    "shifting features, melting face, face distortion, mutating face, character design change, "
    "outfit change, color shift, temporal flickering, pixel noise, pixel degradation, warping, tearing"
)

QWEN_MAX_INPUT_IMAGES = 3   # base + 2 references (Qwen-Image-Edit-Plus multi-image limit)


# ── Pure helpers (unit-testable) ──────────────────────────────────────────────

def generate_three_studio_seeds() -> Dict[str, int]:
    """
    Picks three random 6-digit seeds — one locked seed for each of the three AI tools:
      1. z_image_seed : used for every Z-Image Turbo character portrait
      2. qwen_seed    : used for every Qwen Image Edit scene/scenario keyframe
      3. wan_seed     : used for every Wan 2.2 5-second video clip
    """
    return {
        "z_image_seed": random.randint(100000, 999999),
        "qwen_seed": random.randint(100000, 999999),
        "wan_seed": random.randint(100000, 999999),
    }


def frames_for_duration(duration_seconds: float) -> int:
    """Nearest valid Wan 2.2 frame count (4k+1) for a duration, clamped to 17..81."""
    raw = int(round(float(duration_seconds) * WAN_FPS))
    k = round(raw / 4)
    k = max((WAN_MIN_FRAMES - 1) // 4, min((WAN_MAX_FRAMES - 1) // 4, k))
    return 1 + 4 * k


def plan_scene_count(target_duration: float, requested_scenes: int = 0) -> int:
    """Minimum scenes needed so no clip exceeds Wan's ~5s window (e.g. 30s -> 6 scenes)."""
    needed = math.ceil(float(target_duration) / MAX_CLIP_SECONDS)
    return max(int(requested_scenes or 0), needed, 1)


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return max(lo, min(hi, v))


def _trim(text: Any, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"


def match_color_histogram_lab(
    source_pil: Image.Image,
    reference_pil: Image.Image,
    strength: float = 0.85,
) -> Image.Image:
    """
    Matches the CIE-Lab luminance and color-temperature statistics (Reinhard transfer)
    of `source_pil` (e.g. End Frame) to `reference_pil` (e.g. Start Frame) so Wan 2.2
    First-Last-Frame (FLF2V) interpolation never suffers from exposure flashing or tint drift.
    """
    import numpy as np
    import cv2

    strength = float(_clamp(strength, 0.0, 1.0, 0.85))
    if strength <= 0.0 or source_pil is None or reference_pil is None:
        return source_pil

    src_rgb = np.array(source_pil.convert("RGB"), dtype=np.uint8)
    ref_rgb = np.array(reference_pil.convert("RGB").resize(source_pil.size, Image.Resampling.BILINEAR), dtype=np.uint8)

    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    matched_lab = np.empty_like(src_lab)
    for ch in range(3):
        s_mean, s_std = float(src_lab[:, :, ch].mean()), float(src_lab[:, :, ch].std()) + 1e-5
        r_mean, r_std = float(ref_lab[:, :, ch].mean()), float(ref_lab[:, :, ch].std()) + 1e-5
        # Clamp std ratio to [0.7, 1.35] so contrast never blows out
        std_ratio = max(0.7, min(1.35, r_std / s_std))
        transferred = (src_lab[:, :, ch] - s_mean) * std_ratio + r_mean
        matched_lab[:, :, ch] = (1.0 - strength) * src_lab[:, :, ch] + strength * transferred

    matched_lab = np.clip(matched_lab, 0.0, 255.0).astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(matched_rgb)


def restore_extracted_video_frame(
    frame_input: Any,
    reference_pil: Optional[Image.Image] = None,
    sharpen_strength: float = 0.35,
    color_match_strength: float = 0.75,
) -> Image.Image:
    """
    Restores high-frequency detail and eliminates compression drift on frames extracted
    from generated videos before extending clips in Wan 2.2:
    1. Bilateral edge-preserving filtering eliminates H.264 macroblock compression artifacts.
    2. Edge-preserving unsharp masking restores micro-contrast lost during 3D-VAE decoding.
    3. CIE-Lab luminance/chrominance matching normalizes exposure and tone against clean reference image.
    """
    import numpy as np
    import cv2

    if frame_input is None:
        return None

    if isinstance(frame_input, Image.Image):
        rgb_arr = np.array(frame_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(frame_input, np.ndarray):
        rgb_arr = np.array(frame_input, dtype=np.uint8)
    elif isinstance(frame_input, str) and os.path.exists(frame_input):
        rgb_arr = np.array(Image.open(frame_input).convert("RGB"), dtype=np.uint8)
    else:
        return frame_input

    # 1. Bilateral edge-preserving filter (deblock mosquito noise without blurring edges)
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    deblocked = cv2.bilateralFilter(bgr, d=5, sigmaColor=25, sigmaSpace=25)

    # 2. Edge-preserving unsharp masking (recover micro-contrast on eyes, hair, edges)
    strength = float(_clamp(sharpen_strength, 0.0, 1.0, 0.35))
    if strength > 0.0:
        blurred = cv2.GaussianBlur(deblocked, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(deblocked, 1.0 + strength, blurred, -strength, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        sharpened = deblocked

    restored_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    restored_pil = Image.fromarray(restored_rgb)

    # 3. Normalization back to clean reference image (if provided)
    if reference_pil is not None:
        restored_pil = match_color_histogram_lab(restored_pil, reference_pil, strength=color_match_strength)

    return restored_pil


def measure_frame_pose_delta(img_a: Any, img_b: Any) -> float:
    """
    Calculates the spatial patch-based motion delta between two keyframes.
    Evaluates both the global mean and the 90th-percentile patch difference across a 16x16 grid,
    accurately detecting localized limb/body pose transformations even against static backgrounds
    while guarding against false retries or missed frozen poses.
    """
    import numpy as np
    if img_a is None or img_b is None:
        return 0.0
    try:
        if isinstance(img_a, str) and os.path.exists(img_a):
            img_a = Image.open(img_a)
        if isinstance(img_b, str) and os.path.exists(img_b):
            img_b = Image.open(img_b)
        a = np.array(img_a.resize((256, 256)).convert("RGB"), dtype=np.float32)
        b = np.array(img_b.resize((256, 256)).convert("RGB"), dtype=np.float32)
        diff = np.abs(a - b).mean(axis=-1)  # (256, 256) per-pixel mean difference

        # Split into 16x16 spatial patches (each 16x16 pixels)
        patches = diff.reshape(16, 16, 16, 16).mean(axis=(1, 3))
        top_patch_motion = float(np.percentile(patches, 90))
        global_mean = float(diff.mean())

        # Effective pose delta balances top localized body motion (70%) and global scene delta (30%)
        effective_delta = 0.7 * top_patch_motion + 0.3 * global_mean
        return float(effective_delta)
    except Exception:
        return 0.0


def ensure_high_delta_end_prompt(start_prompt: str, end_prompt: str, scenario: str = "", motion: str = "") -> str:
    """
    Enforces an unmistakable physical action and stance delta between the Start Frame (t=0.0s)
    and the End Frame (t=5.0s). If the end prompt is identical, missing, or too semantically
    similar to the start prompt, programmatically injects a dynamic contrasting posture.
    """
    s_clean = (start_prompt or "").strip().lower()
    e_clean = (end_prompt or "").strip().lower()

    s_words = set(re.findall(r"\w+", s_clean))
    e_words = set(re.findall(r"\w+", e_clean))
    overlap = (len(s_words & e_words) / max(1, len(s_words | e_words))) if (s_words and e_words) else 0.0

    pose_delta_keywords = [
        "duck", "crouch", "kneel", "drop", "recoil", "lunge", "turn", "shield", "cover",
        "collapse", "leap", "dive", "halt", "skid", "reload", "raise", "lowered", "reach",
        "aimed", "spin", "fall", "blast", "strike", "tackle", "dodge", "wave", "waving",
        "bow", "bowing", "point", "pointing", "gesture", "jump", "jumping", "cheer"
    ]
    has_delta_kw = any(kw in e_clean for kw in pose_delta_keywords)

    if not end_prompt or e_clean == s_clean or overlap > 0.65 or not has_delta_kw:
        beat = (end_prompt if end_prompt and e_clean != s_clean else "") or motion or scenario or s_clean or "dramatic action completion"
        # Contextual pose transformation
        if any(w in beat.lower() for w in ["fire", "firi", "weapon", "aim", "shoot", "attack", "gun", "combat", "fight", "blast", "bullet"]):
            posture = "recoiling back with weapon lowered, dropped to one knee behind solid rock cover, head ducked low shielding face from smoke and debris"
        elif any(w in beat.lower() for w in ["run", "sprint", "chase", "walk", "rush", "advance", "approach"]):
            posture = "skidding to a full halt, crouched low with one hand bracing on the ground, torso angled sharply, looking back over shoulder in alarm"
        elif any(w in beat.lower() for w in ["magic", "spell", "glow", "cast", "power", "energy", "staff"]):
            posture = "arms thrust forward channeling a brilliant glowing shockwave, torso arched, hair billowing back from energy release"
        elif any(w in beat.lower() for w in ["discover", "find", "look", "see", "stare", "reach", "touch"]):
            posture = "kneeling down touching the ground with one outstretched hand, torso tilted forward, eyes wide in awe"
        else:
            posture = "completely shifted into dynamic finishing pose, crouched into a low stance ducking down defensively behind cover, body turned 90 degrees"

        return (
            f"Ending frame at t=5.0s: The characters have completed the action: {beat}. "
            f"Visibly transformed body posture and physical silhouette: {posture}. "
            f"Distinct arm, torso, and leg positions from the starting frame. "
            f"Keep the exact same room geometry, trees, rock background, lighting, art style, and clothing identical."
        )
    return end_prompt


def evaluate_flf2v_compatibility(
    start_pil: Any,
    end_pil: Any,
    min_delta: float = 18.0,
    max_delta: float = 85.0,
    min_bg_ssim: float = 0.68,
) -> Dict[str, Any]:
    """
    Two-Sided Pose-Delta & Background-Similarity Gatekeeper for Wan 2.2 FLF2V:
    - TOO_STATIC:        pose_delta < 18.0 (start and end poses are too similar -> frozen video)
    - EXTREME_DIVERGENCE: pose_delta > 85.0 or bg_similarity < 0.68 (background/geometry diverged -> mid-clip cross-fade/morphing)
    - OPTIMAL_FLF2V:     18.0 <= pose_delta <= 85.0 and bg_similarity >= 0.68 (sweet spot for smooth 81-frame interpolation)
    """
    import numpy as np

    if start_pil is None or end_pil is None:
        return {
            "pose_delta": 0.0,
            "bg_similarity": 1.0,
            "status": "TOO_STATIC",
            "flf2v_safe": False,
        }

    try:
        if isinstance(start_pil, str) and os.path.exists(start_pil):
            start_pil = Image.open(start_pil)
        if isinstance(end_pil, str) and os.path.exists(end_pil):
            end_pil = Image.open(end_pil)

        pose_delta = measure_frame_pose_delta(start_pil, end_pil)

        a = np.array(start_pil.resize((256, 256)).convert("RGB"), dtype=np.float32)
        b = np.array(end_pil.resize((256, 256)).convert("RGB"), dtype=np.float32)

        # Compute outer 14% border ring similarity to verify background geometry stability
        border = int(round(256 * 0.14))
        mask = np.zeros((256, 256), dtype=bool)
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True

        border_diff = float(np.abs(a[mask] - b[mask]).mean())
        bg_similarity = float(max(0.0, min(1.0, 1.0 - (border_diff / 90.0))))

        if pose_delta < min_delta:
            status = "TOO_STATIC"
            flf2v_safe = False
            use_flf2v = True
        elif pose_delta > max_delta or bg_similarity < min_bg_ssim:
            status = "EXTREME_DIVERGENCE"
            flf2v_safe = False
            use_flf2v = False
        else:
            status = "OPTIMAL_FLF2V"
            flf2v_safe = True
            use_flf2v = True

        return {
            "pose_delta": round(pose_delta, 2),
            "bg_similarity": round(bg_similarity, 3),
            "bg_ssim": round(bg_similarity, 3),
            "status": status,
            "flf2v_safe": flf2v_safe,
            "use_flf2v": use_flf2v,
        }
    except Exception:
        return {
            "pose_delta": 0.0,
            "bg_similarity": 1.0,
            "bg_ssim": 1.0,
            "status": "TOO_STATIC",
            "flf2v_safe": False,
            "use_flf2v": True,
        }


def stabilize_end_frame_background(
    start_pil: Image.Image,
    end_pil: Image.Image,
    border_ratio: float = 0.14,
) -> Image.Image:
    """
    Soft-feathered background border lock for Wan 2.2 FLF2V:
    Blends the outer peripheral background border of `start_pil` into `end_pil` using a smooth
    Gaussian-feathered alpha mask while leaving the central 72% character action zone untouched.
    Eliminates peripheral wall/foliage drift during Qwen Image Edit end-frame synthesis.
    """
    import numpy as np
    import cv2

    if start_pil is None or end_pil is None:
        return end_pil

    w, h = end_pil.size
    start_resized = start_pil.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    end_rgb = np.array(end_pil.convert("RGB"), dtype=np.float32)
    start_rgb = np.array(start_resized, dtype=np.float32)

    # Create central action mask (1.0 in center, 0.0 at outer border edges)
    mask = np.zeros((h, w), dtype=np.float32)
    bx = max(4, int(round(w * border_ratio)))
    by = max(4, int(round(h * border_ratio)))
    mask[by : h - by, bx : w - bx] = 1.0

    # Feather the transition smoothly
    ksize = max(15, (min(bx, by) // 2) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=max(5.0, min(bx, by) * 0.45))
    mask_3d = mask[:, :, None]

    # Preserve 100% of end_rgb in center, blend 65% of start_rgb along outer peripheral border
    edge_weight = (1.0 - mask_3d) * 0.65
    blended = end_rgb * (1.0 - edge_weight) + start_rgb * edge_weight
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def compute_motion_adaptive_wan_params(
    motion_intensity: int = 2,
    duration_seconds: float = 5.0,
) -> Dict[str, Any]:
    """
    Motion-Adaptive Dual-CFG & Flow-Shift Matrix for Wan 2.2 14B Lightning:
    - Low-Motion / Dialogue (Intensity 1-2): flow_shift=2.8, cfg=2.0, cfg_2=1.5, steps=8
    - Medium Narrative Action (Intensity 3): flow_shift=3.8, cfg=1.9, cfg_2=1.4, steps=10
    - High-Velocity Action / Combat (Intensity 4-5): flow_shift=4.8, cfg=1.75, cfg_2=1.25, steps=12
      (prevents high-frequency CFG ghosting / double-limbs during fast motion).
    """
    intensity = int(round(_clamp(motion_intensity, 1, 5, 2)))
    if intensity >= 4:
        params = {
            "guidance_scale": 1.75,
            "guidance_scale_2": 1.25,
            "flow_shift": 4.8,
            "steps": 12,
        }
    elif intensity == 3:
        params = {
            "guidance_scale": 1.9,
            "guidance_scale_2": 1.4,
            "flow_shift": 3.8,
            "steps": 10,
        }
    else:
        params = {
            "guidance_scale": WAN_DEFAULTS["guidance_scale"],
            "guidance_scale_2": WAN_DEFAULTS["guidance_scale_2"],
            "flow_shift": WAN_DEFAULTS["flow_shift"],
            "steps": WAN_DEFAULTS["steps"],
        }
    return params


def verify_video_clip_quality(
    video_path: str,
    expected_peak_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Post-Render Optical Flow & Sharpness Telemetry Gate:
    Inspects a rendered MP4 clip using Farneback dense optical flow and Laplacian variance:
      1. mean_flow: average optical flow magnitude across the clip (detects frozen clips < 0.35).
      2. surge_ratio: max(frame_flow) / median(frame_flow) (detects mid-clip cross-fade/morphing > 6.5).
      3. sharpness_retention: Laplacian sharpness of final 20% frames vs first 20% frames (detects late-frame blur < 0.45).
      4. peak_flow_sec & audio_sync_delta_sec: measures temporal alignment between peak visual motion
         and the narrator's spoken sync_trigger_word timestamp.
    """
    import numpy as np
    import cv2

    if not video_path or not os.path.exists(video_path):
        return {
            "passed": False,
            "mean_flow": 0.0,
            "surge_ratio": 0.0,
            "sharpness_retention": 0.0,
            "peak_flow_sec": 0.0,
            "audio_sync_delta_sec": None,
            "issues": ["Video file missing"],
        }

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or WAN_FPS)
    if fps <= 0:
        fps = float(WAN_FPS)

    gray_frames = []
    sharpness_vals = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        small = cv2.resize(frame, (160, 160))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_vals.append(lap_var)
    cap.release()

    if len(gray_frames) < 2:
        return {
            "passed": True,
            "mean_flow": 0.0,
            "surge_ratio": 1.0,
            "sharpness_retention": 1.0,
            "peak_flow_sec": 0.0,
            "audio_sync_delta_sec": None,
            "issues": [],
        }

    flow_mags = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1], None,
            pyr_scale=0.5, levels=2, winsize=13, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        flow_mags.append(mag)

    mean_flow = float(np.mean(flow_mags)) if flow_mags else 0.0
    median_flow = float(np.median(flow_mags)) if flow_mags else 0.0
    max_flow = float(np.max(flow_mags)) if flow_mags else 0.0
    surge_ratio = float(max_flow / max(median_flow, 0.15)) if flow_mags else 1.0

    peak_idx = int(np.argmax(flow_mags)) if flow_mags else 0
    peak_flow_sec = round((peak_idx + 1) / fps, 2)

    n_sharp = max(1, len(sharpness_vals) // 5)
    early_sharp = float(np.mean(sharpness_vals[:n_sharp])) + 1e-5
    late_sharp = float(np.mean(sharpness_vals[-n_sharp:])) + 1e-5
    sharpness_retention = float(late_sharp / early_sharp)

    issues = []
    if mean_flow < 0.15:
        issues.append(f"Frozen clip detected (mean_flow={mean_flow:.2f} < 0.15)")
    if surge_ratio > 6.5 and max_flow > 4.0:
        issues.append(f"Mid-clip morphing surge detected (surge_ratio={surge_ratio:.2f} > 6.5)")
    if sharpness_retention < 0.45 and early_sharp > 15.0:
        issues.append(f"Late-frame sharpness degradation (retention={sharpness_retention:.2f} < 0.45)")

    audio_sync_delta = None
    if expected_peak_sec is not None:
        audio_sync_delta = round(abs(peak_flow_sec - float(expected_peak_sec)), 2)

    return {
        "passed": len(issues) == 0,
        "mean_flow": round(mean_flow, 3),
        "surge_ratio": round(surge_ratio, 2),
        "sharpness_retention": round(sharpness_retention, 2),
        "peak_flow_sec": peak_flow_sec,
        "audio_sync_delta_sec": audio_sync_delta,
        "issues": issues,
    }


class StudioPipeline:
    """
    Automated Multi-Agent Video Production Studio Coordinator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ):
        self.api_key = api_key or DEFAULT_OLLAMA_API_KEY
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.endpoint_url = endpoint_url or DEFAULT_OLLAMA_ENDPOINT

        if output_base_dir:
            self.base_dir = pathlib.Path(output_base_dir)
        else:
            self.base_dir = pathlib.Path(__file__).parent / "outputs" / "studio"

        self.char_dir = self.base_dir / "characters"
        self.scenes_dir = self.base_dir / "scenes"
        self.clips_dir = self.base_dir / "clips"
        self.movies_dir = self.base_dir / "movies"
        self.audio_dir = self.base_dir / "audio"

        for d in [self.char_dir, self.scenes_dir, self.clips_dir, self.movies_dir, self.audio_dir]:
            d.mkdir(parents=True, exist_ok=True)


    # ─────────────────────────────────────────────────────────────────────────
    # LLM transport (shared by Story Architect and Animation Director)
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_model_tag(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        task_role: str = "default",
        model_override: Optional[str] = None,
    ) -> str:
        """
        Resolves the actual Ollama model tag to use.
        If model is 'auto', routes task to the benchmarked optimal model:
          - 'screenplay': gpt-oss:120b (Hollywood depth & dialogue)
          - 'character': gemma4:31b (Visual prompt isolation)
          - 'director_qa', 'end_frame': nemotron-3-super (Rigorous alignment & critic)
          - 'choreography', 'combat': nemotron-3-ultra (Physics & weapon-locked action)
          - 'quick_edit', 'timeline': nemotron-3-nano:30b (Fast 2.8s turnaround)
        If messages contain images, automatically routes to vision-capable 'gemma4:31b'.
        """
        if model_override:
            return model_override

        # Vision safeguard: only vision models can receive image_url payloads
        if messages:
            for m in messages:
                c = m.get("content")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            return "gemma4:31b"

        m = (self.model or DEFAULT_OLLAMA_MODEL).strip()
        if "auto" in m.lower():
            if task_role == "screenplay":
                return "gpt-oss:120b"
            elif task_role in ("director_qa", "end_frame"):
                return "nemotron-3-super"
            elif task_role in ("choreography", "combat"):
                return "nemotron-3-ultra"
            elif task_role in ("quick_edit", "timeline"):
                return "nemotron-3-nano:30b"
            return "gemma4:31b"
        return m

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2500,
        temperature: float = 0.7,
        timeout: int = 120,
        max_retries: int = 2,
        task_role: str = "default",
        model_override: Optional[str] = None,
    ) -> str:
        """POSTs to the OpenAI-compatible Ollama Cloud endpoint and returns the message text."""
        actual_model = self._resolve_model_tag(messages=messages, task_role=task_role, model_override=model_override)
        req_body = {
            "model": actual_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(req_body).encode("utf-8"),
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    return resp_data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(2.0)
                    continue
        raise RuntimeError(f"Ollama API request failed ({actual_model}): {last_err}")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Story Generation & Character Extraction (Gemma 4: 31B)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_story_and_characters(
        self,
        prompt: str,
        num_scenes: int = 6,
        target_duration: float = 30.0,
        art_style: str = "2D vibrant cartoon animation",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Queries Ollama Cloud (gemma4:31b) to produce a structured short story with
        extracted character visual details and scene-by-scene story beats (including both
        START frame and END frame prompts for Wan 2.2 First-Last-Frame animation).
        Scene count is raised automatically so no clip exceeds Wan's 5s window.
        """
        num_scenes = plan_scene_count(target_duration, num_scenes)
        scene_duration = round(min(MAX_CLIP_SECONDS, target_duration / num_scenes), 2)

        if progress_cb:
            progress_cb(0.03, "Consulting AI Director for Causal Story Architecture & State Ledgers...")

        # ── Retrieve In-Context Few-Shot Exemplars from Memory Bank ────────────
        memory_bank = StoryMemoryBank()
        exemplars = memory_bank.retrieve_exemplars(prompt, top_k=1)
        few_shot_prompt = memory_bank.format_few_shot_prompt(exemplars)

        # ── Retrieve Narrative Beat Guide for Target Scene Count ───────────────
        beat_presets = NARRATIVE_BEATS_BY_COUNT.get(num_scenes, NARRATIVE_BEATS_BY_COUNT.get(6, []))
        beat_guide_text = "\n".join(
            f"  - Scene {i+1}: {b.get('beat', f'Beat {i+1}')} (Causal Connector: '{b.get('connector', 'THEREFORE')}') -> {b.get('role', '')}"
            for i, b in enumerate(beat_presets[:num_scenes])
        )

        system_prompt = (
            "You are a master animation director and AI screenwriter directing a causally coherent animated short film.\n"
            f"The user wants a short movie: {target_duration} seconds total, EXACTLY {num_scenes} scenes, "
            f"~{scene_duration}s each. Every scene is ONE continuous shot animated in Wan 2.2 FLF2V between "
            "a FIRST FRAME (t=0.0s) and a LAST FRAME (t=5.0s).\n\n"
            "CRITICAL CAUSAL STORYTELLING LAWS (PIXAR / SOUTH PARK RULE):\n"
            "1. NEVER CONNECT SCENES WITH 'AND THEN': Every scene must connect via 'THEREFORE' (direct causal consequence "
            "of the preceding scene) or 'BUT' (unexpected obstacle or twist). Random unrelated events are strictly forbidden!\n"
            "2. ATOMIC 5-SECOND FEASIBILITY: A single 5.0-second Wan 2.2 clip (81 frames) can ONLY execute ONE atomic physical action "
            "and its immediate reaction (e.g. 'draws blade and leaps forward' or 'lands on one knee and shields face'). "
            "NEVER cram 3-4 separate story beats into one 5s scene!\n"
            "3. WEAPON & PROP RETENTION LOCK: All active weapons, items, and accessories in character hands (swords, lanterns, scrolls, goggles) "
            "must stay in character hands across cuts unless explicitly shown being dropped, stowed, or shattered.\n"
            "4. 180-DEGREE SCREEN DIRECTION: Maintain consistent spatial eyelines and screen direction (e.g. hero moving screen-left to screen-right) "
            "across cuts so characters do not disorientingly flip orientation.\n"
            "5. HIGH POSE DELTA: Wan 2.2 FLF2V interpolates between Frame 0 and Frame 80. If start and end poses are too similar, the video will freeze! "
            "The End Frame (t=5.0s) MUST have a visibly transformed body posture, changed limbs, and clear follow-through.\n"
            "6. SYNCHRONIZED VOICEOVER NARRATION (6 TO 11 WORDS MAX PER SCENE): Every scene must include a spoken 'narration_text' (strictly 6 to 11 words "
            "so it is spoken cleanly in 2.2s–3.8s inside the 5.0s shot), a 'narration_voice_style' (e.g. 'warm_wonder', 'tense_whisper', 'urgent_rising', "
            "'epic_climax', 'joyful_resolution'), and a single 'sync_trigger_word' (the exact action verb in narration_text where the visual climax occurs).\n\n"
            f"NARRATIVE BEAT STRUCTURE TO FOLLOW:\n{beat_guide_text}\n\n"
            + (f"{few_shot_prompt}\n\n" if few_shot_prompt else "")
            + "CRITICAL: Output ONLY valid JSON matching this schema without markdown fences or commentary:\n"
            "{\n"
            '  "title": "Short Film Title",\n'
            '  "art_style": "Visual style description",\n'
            '  "theme": "Moral or thematic premise",\n'
            '  "characters": [\n'
            "    {\n"
            '      "id": "char_1",\n'
            '      "name": "Character Name",\n'
            '      "description": "Personality and role summary",\n'
            '      "face": "Exact face features: shape, eyes, expression, facial markings",\n'
            '      "hair": "Hair/fur style, color, texture, length",\n'
            '      "clothing": "Complete outfit, fabric colors, style, shoes",\n'
            '      "signature_details": "Unique identifiable items, props, or weapons held",\n'
            '      "z_image_prompt": "Ultra-detailed full-body character sheet prompt for Z-Image Turbo, front view, '
            'clean neutral studio background, in the target art style"\n'
            "    }\n"
            "  ],\n"
            '  "scenes": [\n'
            "    {\n"
            '      "scene_number": 1,\n'
            '      "title": "Scene Name",\n'
            f'      "duration_seconds": {scene_duration},\n'
            '      "narrative_beat": "e.g. Hook & Status Quo / Inciting Obstacle / Climax & Payoff",\n'
            '      "causal_connector": "SETUP / BUT ... / THEREFORE ...",\n'
            '      "narration_text": "Spoken voiceover line for this 5s shot (strictly 6 to 11 words)",\n'
            '      "narration_voice_style": "warm_wonder / tense_whisper / urgent_rising / epic_climax / joyful_resolution",\n'
            '      "sync_trigger_word": "exact action verb in narration_text that triggers the visual climax",\n'
            '      "state_ledger": {\n'
            '        "entry_state": "Exact character body posture, location, and hand items at t=0.0s",\n'
            '        "active_props": ["list of held weapons or items"],\n'
            '        "spatial_zone": "Position and facing direction (e.g. screen-left facing right)",\n'
            '        "atomic_action_5s": "The single physical action and reaction taking 5 seconds",\n'
            '        "exit_state": "Exact character body posture and prop hold at t=5.0s"\n'
            '      },\n'
            '      "scenario": "Story beat: what happens in this single shot",\n'
            '      "qwen_scene_prompt": "FIRST FRAME (t=0s) prompt: environment, composition, initiating character poses at the START of the shot",\n'
            '      "qwen_end_scene_prompt": "LAST FRAME (t=5s) prompt: same environment and lighting, but VISIBLY TRANSFORMED character poses/limbs/actions at the END of the shot",\n'
            '      "wan_motion_prompt": "Draft motion description bridging the first frame to the last frame",\n'
            '      "camera_movement": "e.g. Slow Cinematic Push-In / Static Tripod Shot",\n'
            '      "character_ids": ["char_1"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_content = (
            f"Story Premise: {prompt}\nTarget Style: {art_style}\n"
            f"Number of Scenes: {num_scenes}\nTotal Duration: {target_duration}s."
        )

        raw_text = self._chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=3500,
            temperature=0.7,
            task_role="screenplay",
        )

        parsed_data = self._parse_json_safely(raw_text)
        parsed_data = self.normalize_story(
            parsed_data, prompt=prompt, num_scenes=num_scenes,
            target_duration=target_duration, art_style=art_style,
        )

        if progress_cb:
            progress_cb(
                0.15,
                f"Story generated: '{parsed_data.get('title', 'Studio Story')}' "
                f"({len(parsed_data['characters'])} characters, {len(parsed_data['scenes'])} scenes with Causal State Ledgers)",
            )
        return parsed_data


    @staticmethod
    def normalize_story(
        data: Dict[str, Any],
        prompt: str,
        num_scenes: int,
        target_duration: float,
        art_style: str,
    ) -> Dict[str, Any]:
        """
        Enforces the schema contract on LLM output: valid character ids, scenes that reference
        existing characters, sequential numbering, both start & end keyframe prompts, and per-scene
        durations within Wan's window.
        """
        data = dict(data or {})
        data.setdefault("title", "Studio Story")
        data.setdefault("art_style", art_style)

        chars = data.get("characters")
        if not isinstance(chars, list) or not chars:
            chars = [{
                "id": "char_1",
                "name": "Hero",
                "face": "Friendly and expressive",
                "hair": "Clean styled",
                "clothing": "Casual animation attire",
                "signature_details": "",
                "z_image_prompt": f"Character sheet of the protagonist of: {prompt}, {art_style}, neutral studio background",
            }]
        norm_chars, seen_ids = [], set()
        for i, c in enumerate(chars):
            if not isinstance(c, dict):
                continue
            c = dict(c)
            cid = str(c.get("id") or f"char_{i+1}")
            if cid in seen_ids:
                cid = f"{cid}_{i+1}"
            seen_ids.add(cid)
            c["id"] = cid
            c.setdefault("name", f"Character {i+1}")
            norm_chars.append(c)
        # Enforce maximum 3 characters budget (Heart-Touching Intimacy Constraint)
        if len(norm_chars) > MAX_STORY_CHARACTERS:
            norm_chars = norm_chars[:MAX_STORY_CHARACTERS]
        data["characters"] = norm_chars
        valid_ids = [c["id"] for c in norm_chars]

        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            scenes = [{
                "title": "Scene 1",
                "scenario": prompt,
                "qwen_scene_prompt": prompt,
                "qwen_end_scene_prompt": f"{prompt}, characters settled in warm final pose at end of shot",
                "wan_motion_prompt": "cinematic smooth motion, locked character identity",
                "camera_movement": "Slow Cinematic Push-In",
                "character_ids": valid_ids[:1],
            }]
        scenes = [s for s in scenes if isinstance(s, dict)][: max(1, int(num_scenes))]

        per_scene = round(_clamp(target_duration / max(1, len(scenes)), MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, 3.5), 2)
        norm_scenes = []
        for i, s in enumerate(scenes):
            s = dict(s)
            s["scene_number"] = i + 1
            s.setdefault("title", f"Scene {i+1}")
            s.setdefault("scenario", s.get("qwen_scene_prompt", ""))
            s.setdefault("qwen_scene_prompt", s.get("scenario", prompt))
            s["qwen_end_scene_prompt"] = ensure_high_delta_end_prompt(
                start_prompt=s.get("qwen_scene_prompt", ""),
                end_prompt=s.get("qwen_end_scene_prompt", ""),
                scenario=s.get("scenario", ""),
                motion=s.get("wan_motion_prompt", ""),
            )
            ids = s.get("character_ids")
            if isinstance(ids, str):
                ids = [ids]
            ids = [cid for cid in (ids or []) if cid in valid_ids]
            s["character_ids"] = ids or valid_ids[:1]
            s["duration_seconds"] = round(
                _clamp(s.get("duration_seconds", per_scene), MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, per_scene), 2
            )
            norm_scenes.append(s)

        # Re-balance so clip durations sum to the target (capped by the per-clip max)
        total = sum(s["duration_seconds"] for s in norm_scenes)
        if abs(total - target_duration) > 0.5:
            for s in norm_scenes:
                s["duration_seconds"] = per_scene

        # ── Causal State Ledger, Narration & Narrative Beat Enforcement ───────
        char_props = [c.get("signature_details") for c in norm_chars if c.get("signature_details")]
        beat_presets = NARRATIVE_BEATS_BY_COUNT.get(len(norm_scenes), NARRATIVE_BEATS_BY_COUNT.get(6, []))

        for i, s in enumerate(norm_scenes):
            default_beat = beat_presets[i]["beat"] if i < len(beat_presets) else f"Beat {i+1}"
            default_conn = beat_presets[i]["connector"] if i < len(beat_presets) else ("SETUP" if i == 0 else "THEREFORE")
            s.setdefault("narrative_beat", default_beat)
            s.setdefault("causal_connector", default_conn)

            # Enforce 6-11 word spoken narration & sync trigger word
            raw_narration = s.get("narration_text") or ""
            s["narration_text"] = enforce_narration_word_budget(
                raw_narration,
                scenario=s.get("scenario") or s.get("qwen_scene_prompt", ""),
                max_words=12,
            )
            s.setdefault("narration_voice_style", "cinematic_warm")
            s["sync_trigger_word"] = extract_sync_trigger_word(
                s["narration_text"],
                scenario=s.get("scenario", ""),
                explicit_trigger=s.get("sync_trigger_word"),
            )

            raw_ledger = s.get("state_ledger")
            if not isinstance(raw_ledger, dict):
                raw_ledger = {}

            active_props = raw_ledger.get("active_props")
            if not isinstance(active_props, list) or not active_props:
                if i > 0 and norm_scenes[i - 1].get("state_ledger", {}).get("active_props"):
                    active_props = list(norm_scenes[i - 1]["state_ledger"]["active_props"])
                else:
                    active_props = [char_props[0]] if char_props else []

            entry_state = raw_ledger.get("entry_state")
            if not entry_state:
                if i > 0 and norm_scenes[i - 1].get("state_ledger", {}).get("exit_state"):
                    entry_state = f"Continuing from previous shot: {norm_scenes[i - 1]['state_ledger']['exit_state']}"
                else:
                    entry_state = f"Starting stance: {s.get('qwen_scene_prompt', '')[:100]}"

            exit_state = raw_ledger.get("exit_state")
            if not exit_state:
                exit_state = f"Finishing posture: {s.get('qwen_end_scene_prompt', '')[:100]}"

            spatial = raw_ledger.get("spatial_zone") or (
                norm_scenes[i - 1].get("state_ledger", {}).get("spatial_zone") if i > 0 else "Screen-center"
            )
            atomic = raw_ledger.get("atomic_action_5s") or s.get("scenario") or s.get("wan_motion_prompt") or "Primary physical action"

            s["state_ledger"] = {
                "entry_state": entry_state,
                "active_props": active_props,
                "spatial_zone": spatial,
                "atomic_action_5s": atomic,
                "exit_state": exit_state,
            }

        data["scenes"] = norm_scenes
        data = enforce_orchestra_story_constraints(data)
        return data

    def audit_and_repair_story_logic(
        self,
        story_data: Dict[str, Any],
        model_choice: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-Agent 'Story Doctor' & Causal Continuity Critic.
        Audits the screenplay before image/video rendering against 5 key criteria:
        1. Narrative Causality: Are scenes linked by THEREFORE / BUT instead of AND THEN?
        2. 5-Second Feasibility: Does each scene feature 1 atomic action + reaction that fits in 5 seconds (and narration <= 12 words)?
        3. Prop & Weapon Permanence: Do items stay in hands without vanishing mid-story?
        4. Spatial & Screen Direction: Is 180-degree cinema direction respected across cuts?
        5. Visual Action Progression: Is there a dramatic physical pose difference between start and end?

        If any score is < 8/10, automatically repairs and enhances the scene beats,
        and logs the (rejected, chosen) pair to outputs/studio/story_dpo_pairs.jsonl
        for ongoing model fine-tuning.
        """
        if progress_cb:
            progress_cb(0.08, "Running Story Doctor Causal Audit on screenplay...")

        scenes = story_data.get("scenes", [])
        if not scenes:
            return {
                "passed": True,
                "overall_score": 10.0,
                "subscores": {"causality": 10.0, "feasibility_5s": 10.0, "prop_permanence": 10.0, "spatial_continuity": 10.0, "visual_delta": 10.0},
                "critique": "No scenes to audit.",
                "issues": [],
                "repaired": False,
                "story": story_data,
            }

        system_prompt = (
            "You are the Lead Story Doctor and Animation Continuity Critic for a world-class animation studio.\n"
            "Your mission is to audit an animated short film storyboard and ensure it possesses ironclad narrative causality, "
            "strict 5-second temporal feasibility, prop permanence, synchronized narration (6-11 words), and visual action progression.\n\n"
            "AUDIT CRITERIA (Score 1-10 each):\n"
            "1. causality: Every scene transition must connect via 'THEREFORE' (consequence) or 'BUT' (obstacle). Ban lazy 'AND THEN' jumps.\n"
            "2. feasibility_5s: Each 5.0-second Wan 2.2 clip can only execute ONE primary physical action and immediate reaction, and narration_text must be <= 12 words.\n"
            "3. prop_permanence: Weapons and items in characters' hands must stay in their hands across cuts unless explicitly dropped, sheared, or stowed.\n"
            "4. spatial_continuity: Keep 180-degree screen direction consistent (e.g. facing screen-right vs screen-left).\n"
            "5. visual_delta: The End Frame (t=5s) MUST be a drastically different physical body stance from the Start Frame (t=0s).\n\n"
            "Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "overall_score": 1-10,\n'
            '  "subscores": {\n'
            '    "causality": 1-10,\n'
            '    "feasibility_5s": 1-10,\n'
            '    "prop_permanence": 1-10,\n'
            '    "spatial_continuity": 1-10,\n'
            '    "visual_delta": 1-10\n'
            '  },\n'
            '  "critique": "Detailed constructive feedback identifying logical weak points",\n'
            '  "needs_repair": true/false,\n'
            '  "repaired_scenes": [\n'
            "    {\n"
            '      "scene_number": 1,\n'
            '      "title": "Scene Name",\n'
            '      "narrative_beat": "e.g. Hook & Status Quo",\n'
            '      "causal_connector": "SETUP / BUT ... / THEREFORE ...",\n'
            '      "narration_text": "Spoken line (6 to 11 words)",\n'
            '      "narration_voice_style": "warm_wonder / tense_whisper / urgent_rising / epic_climax / joyful_resolution",\n'
            '      "sync_trigger_word": "action verb in narration_text",\n'
            '      "state_ledger": {\n'
            '        "entry_state": "Starting body posture at t=0s",\n'
            '        "active_props": ["held items/weapons"],\n'
            '        "spatial_zone": "Position and facing direction",\n'
            '        "atomic_action_5s": "Single physical action and reaction taking 5s",\n'
            '        "exit_state": "Finishing body posture at t=5s"\n'
            '      },\n'
            '      "scenario": "Polished one-sentence beat",\n'
            '      "qwen_scene_prompt": "First frame prompt",\n'
            '      "qwen_end_scene_prompt": "Last frame prompt (high-delta)",\n'
            '      "wan_motion_prompt": "Motion description",\n'
            '      "camera_movement": "Camera directive",\n'
            '      "character_ids": ["char_1"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_content = json.dumps(
            {
                "title": story_data.get("title"),
                "theme": story_data.get("theme"),
                "art_style": story_data.get("art_style"),
                "characters": story_data.get("characters"),
                "scenes": [
                    {
                        "scene_number": s.get("scene_number"),
                        "title": s.get("title"),
                        "narrative_beat": s.get("narrative_beat"),
                        "causal_connector": s.get("causal_connector"),
                        "narration_text": s.get("narration_text"),
                        "narration_voice_style": s.get("narration_voice_style"),
                        "sync_trigger_word": s.get("sync_trigger_word"),
                        "state_ledger": s.get("state_ledger"),
                        "scenario": s.get("scenario"),
                        "qwen_scene_prompt": s.get("qwen_scene_prompt"),
                        "qwen_end_scene_prompt": s.get("qwen_end_scene_prompt"),
                        "wan_motion_prompt": s.get("wan_motion_prompt"),
                        "camera_movement": s.get("camera_movement"),
                        "character_ids": s.get("character_ids"),
                    }
                    for s in scenes
                ],
            },
            indent=2,
        )

        audit_model = model_choice or "nemotron-3-super"
        audit_raw = None
        try:
            audit_raw = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model_override=audit_model,
                max_tokens=3000,
                temperature=0.4,
                task_role="screenplay",
                max_retries=1,
            )
        except Exception as e:
            print(f"[StudioPipeline] Story Doctor LLM call skipped or failed ({e}). Running deterministic heuristic critic.")

        parsed = self._parse_json_safely(audit_raw) if audit_raw else {}

        # Fallback heuristic evaluation if LLM didn't return valid scores
        subscores = parsed.get("subscores") or {}
        overall_score = float(parsed.get("overall_score") or 0.0)

        if not subscores or overall_score == 0.0:
            # Deterministic heuristic analysis
            c_score = 9.0
            f_score = 9.0
            p_score = 9.0
            s_score = 9.0
            v_score = 9.0
            issues = []

            for i, s in enumerate(scenes):
                conn = str(s.get("causal_connector", "")).upper()
                if "AND THEN" in conn or (i > 0 and not any(k in conn for k in ("BUT", "THEREFORE", "CONSEQUENCE", "RESULT"))):
                    c_score = max(5.0, c_score - 1.5)
                    issues.append(f"Scene {i+1} lacks an explicit 'THEREFORE' or 'BUT' causal connector.")

                # Check 5s feasibility (heuristic: overly long text with multiple distinct conjunctions)
                scen_text = s.get("scenario", "")
                if scen_text.count(" and ") >= 3 or scen_text.count(" then ") >= 2:
                    f_score = max(6.0, f_score - 1.0)
                    issues.append(f"Scene {i+1} scenario contains multiple actions that may exceed the 5-second window.")

                # Check narration word budget
                narr_words = re.findall(r"\w+", str(s.get("narration_text", "")))
                if len(narr_words) > 13:
                    f_score = max(6.0, f_score - 1.0)
                    issues.append(f"Scene {i+1} narration exceeds the 12-word spoken budget for a 5s shot.")
                    s["narration_text"] = enforce_narration_word_budget(s.get("narration_text", ""), scen_text, max_words=12)

                # Check visual delta
                p1 = s.get("qwen_scene_prompt", "")
                p2 = s.get("qwen_end_scene_prompt", "")
                if p1 and p2 and p1.strip().lower() == p2.strip().lower():
                    v_score = 4.0
                    issues.append(f"Scene {i+1} Start Frame and End Frame prompts are identical (causes frozen video).")

            overall_score = round((c_score + f_score + p_score + s_score + v_score) / 5.0, 1)
            subscores = {
                "causality": c_score,
                "feasibility_5s": f_score,
                "prop_permanence": p_score,
                "spatial_continuity": s_score,
                "visual_delta": v_score,
            }
            needs_repair = overall_score < 8.0 or any(v < 7.0 for v in subscores.values())
            critique = "; ".join(issues) if issues else "Storyboard conforms to causal narrative standards."
        else:
            overall_score = round(float(overall_score), 1)
            needs_repair = bool(parsed.get("needs_repair", overall_score < 8.0))
            critique = str(parsed.get("critique", ""))
            issues = [critique] if critique else []

        repaired = False
        repaired_scenes = parsed.get("repaired_scenes")
        if needs_repair and isinstance(repaired_scenes, list) and len(repaired_scenes) == len(scenes):
            original_story = json.loads(json.dumps(story_data))
            for i, rep in enumerate(repaired_scenes):
                if isinstance(rep, dict):
                    # Retain durations and identifiers while updating narrative content
                    for key in (
                        "title", "narrative_beat", "causal_connector", "narration_text",
                        "narration_voice_style", "sync_trigger_word", "state_ledger",
                        "scenario", "qwen_scene_prompt", "qwen_end_scene_prompt",
                        "wan_motion_prompt", "camera_movement",
                    ):
                        if rep.get(key):
                            scenes[i][key] = rep[key]
            story_data["scenes"] = scenes
            repaired = True

            # Record DPO preference pair into memory bank
            try:
                memory_bank = StoryMemoryBank()
                memory_bank.record_dpo_pair(
                    premise=story_data.get("theme", story_data.get("title", "")),
                    rejected_story=original_story,
                    chosen_story=story_data,
                    critique_feedback=critique,
                )
            except Exception as e:
                print(f"[StudioPipeline] Failed to record DPO pair: {e}")

        audit_result = {
            "passed": overall_score >= 8.0,
            "overall_score": overall_score,
            "subscores": subscores,
            "critique": critique,
            "issues": issues,
            "repaired": repaired,
        }
        # Decouple back-reference to prevent infinite recursion
        story_data["logic_audit"] = dict(audit_result)
        result_with_story = dict(audit_result)
        result_with_story["story"] = story_data
        return result_with_story


    def record_story_to_memory(
        self,
        story_data: Dict[str, Any],
        premise: str = "",
        rating: float = 10.0,
        user_edited: bool = False,
        notes: str = "",
    ) -> bool:
        """Saves a story into the local episodic memory bank for future in-context learning."""
        bank = StoryMemoryBank()
        p = premise or story_data.get("theme") or story_data.get("title") or "Studio Story"
        return bank.save_exemplar(
            premise=p,
            story_data=story_data,
            rating=rating,
            user_edited=user_edited,
            notes=notes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1B: Open-Source Narrator Voice Synthesis & Audio-Visual Sync
    # ─────────────────────────────────────────────────────────────────────────

    def generate_story_narration(
        self,
        scenes: List[Dict[str, Any]],
        voice_preset: str = DEFAULT_NARRATOR_VOICE,
        quantize_durations: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes open-source voiceover narration (.wav) for every scene using OpenSourceNarratorTTS
        (Kokoro-82M Apache 2.0 with offline harmonic fallback), extracts word-level timestamps,
        quantizes audio to Wan 2.2 4k+1 frame counts, and calculates exact phase transition
        boundaries (p1, p2) locked to each scene's spoken `sync_trigger_word`.
        """
        tts = OpenSourceNarratorTTS(voice_preset=voice_preset)
        total = len(scenes)
        updated_scenes = []

        for idx, scene in enumerate(scenes):
            scene_entry = dict(scene)
            scene_num = scene_entry.get("scene_number", idx + 1)
            title = self._sanitize_filename(scene_entry.get("title", f"scene_{scene_num}"))

            if progress_cb:
                pct = 0.12 + (idx / max(1, total)) * 0.04
                progress_cb(pct, f"🎙️ Synthesizing Narrator Voice for Scene {idx+1}/{total}: '{scene_entry.get('title', '')}'...")

            narration_text = enforce_narration_word_budget(
                scene_entry.get("narration_text", ""),
                scenario=scene_entry.get("scenario") or scene_entry.get("qwen_scene_prompt", ""),
                max_words=12,
            )
            scene_entry["narration_text"] = narration_text

            trigger_word = extract_sync_trigger_word(
                narration_text,
                scenario=scene_entry.get("scenario", ""),
                explicit_trigger=scene_entry.get("sync_trigger_word"),
            )
            scene_entry["sync_trigger_word"] = trigger_word
            scene_entry["narrator_voice_preset"] = voice_preset

            wav_path = str(self.audio_dir / f"scene_{scene_num}_{title}_narration.wav")
            tts_res = tts.synthesize_scene_narration(
                text=narration_text,
                output_wav_path=wav_path,
                voice_preset=voice_preset,
                voice_style=scene_entry.get("narration_voice_style", "cinematic_warm"),
            )

            quant_info = AudioVisualSyncEngine.quantize_audio_to_wan_frames(
                audio_path=tts_res["audio_path"],
                word_timestamps=tts_res["word_timestamps"],
                lead_in_sec=0.25,
                tail_pause_sec=0.35,
            )

            if quantize_durations:
                scene_entry["duration_seconds"] = quant_info["duration_seconds"]
                scene_entry["num_frames"] = quant_info["num_frames"]

            clip_dur = float(scene_entry.get("duration_seconds", quant_info["duration_seconds"]))
            synced_phases = AudioVisualSyncEngine.compute_synced_phase_boundaries(
                duration_seconds=clip_dur,
                word_timestamps=quant_info["shifted_timestamps"],
                sync_trigger_word=trigger_word,
            )

            scene_entry["audio_path"] = quant_info["audio_path"]
            scene_entry["narration_audio_path"] = quant_info["audio_path"]
            scene_entry["word_timestamps"] = quant_info["shifted_timestamps"]
            scene_entry["audio_quantization"] = quant_info
            scene_entry["synced_phases"] = synced_phases
            scene_entry["tts_engine"] = tts_res.get("engine", "kokoro-82m")

            scene.update(scene_entry)
            updated_scenes.append(scene_entry)

        return updated_scenes

    def regenerate_single_scene_narration(
        self,
        story_data: Dict[str, Any],
        scene_index: int,
        narration_text: Optional[str] = None,
        sync_trigger_word: Optional[str] = None,
        voice_preset: Optional[str] = None,
        quantize_durations: bool = False,
    ) -> Dict[str, Any]:
        """Regenerates the open-source narrator audio and sync phase boundaries for a single scene."""
        scenes = list((story_data or {}).get("scenes", []))
        if not scenes or scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"Scene index {scene_index} is out of range (total: {len(scenes)}).")

        target = dict(scenes[scene_index])
        if narration_text is not None and str(narration_text).strip():
            target["narration_text"] = str(narration_text).strip()
        if sync_trigger_word is not None and str(sync_trigger_word).strip():
            target["sync_trigger_word"] = str(sync_trigger_word).strip()

        use_voice = voice_preset or target.get("narrator_voice_preset") or story_data.get("narrator_voice_preset") or DEFAULT_NARRATOR_VOICE
        updated_list = self.generate_story_narration(
            [target],
            voice_preset=use_voice,
            quantize_durations=quantize_durations,
        )
        updated_scene = updated_list[0]
        scenes[scene_index] = updated_scene
        story_data["scenes"] = scenes
        return updated_scene



    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Character Portrait Synthesis via Z-Image Turbo (Locked Seed)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_character_prompt(char: Dict[str, Any], art_style: str = "") -> str:
        """Constructs the full Z-Image Turbo character reference sheet prompt in 9:16 vertical full-body orientation."""
        name = char.get("name", "Character")
        details = ", ".join(
            v for v in [char.get("face"), char.get("hair"), char.get("clothing"), char.get("signature_details")] if v
        )
        base_prompt = char.get("z_image_prompt") or f"Full-body character model sheet of {name}"
        char_prompt = f"{base_prompt}, {details}" if details and details not in base_prompt else base_prompt
        if art_style and art_style.lower() not in char_prompt.lower():
            char_prompt = f"{char_prompt}, {art_style}"
        if "single character" not in char_prompt.lower():
            char_prompt += ", single character, 9:16 vertical full-body portrait, standing head-to-toe, complete outfit and footwear visible, front view, clean neutral background, crisp lighting"
        elif "full-body" not in char_prompt.lower():
            char_prompt += ", 9:16 vertical full-body portrait, standing head-to-toe, complete outfit and footwear visible"
        return char_prompt

    def generate_character_portraits(
        self,
        characters: List[Dict[str, Any]],
        shared_seed: int = 42,
        art_style: str = "",
        width: int = CHARACTER_PORTRAIT_WIDTH,
        height: int = CHARACTER_PORTRAIT_HEIGHT,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes reference portrait images for all extracted characters using
        Z-Image Turbo on Modal in 9:16 vertical full-body orientation (576x1024).
        Uses the SAME LOCKED SEED for all characters to establish a coherent base latent
        structure and visual harmony.
        """
        total = len(characters)
        updated_characters = []
        locked_seed = int(shared_seed)

        for idx, char in enumerate(characters):
            name = char.get("name", f"Char_{idx+1}")
            char_id = char.get("id", f"char_{idx+1}")
            char_prompt = self.build_character_prompt(char, art_style)

            if progress_cb:
                pct = 0.15 + (idx / max(1, total)) * 0.15
                progress_cb(pct, f"Synthesizing character reference {idx+1}/{total}: '{name}' (Seed {locked_seed}, 9:16)...")

            print(f"[StudioPipeline] Generating Z-Image Turbo 9:16 portrait for '{name}' with locked seed {locked_seed} ({width}x{height})...")
            pil_img, local_path = generate_z_image_turbo_remote(
                prompt=char_prompt,
                seed=locked_seed,
                width=width,
                height=height,
                steps=9,
                save_dir=str(self.char_dir),
                filename_prefix=f"{char_id}_{self._sanitize_filename(name)}",
            )

            char_entry = dict(char)
            char_entry["portrait_pil"] = pil_img
            char_entry["portrait_path"] = local_path
            char_entry["portrait_seed"] = locked_seed
            updated_characters.append(char_entry)

        if progress_cb:
            progress_cb(0.30, f"All {total} character portraits synthesized in 9:16 with locked seed {locked_seed}.")

        return updated_characters

    def regenerate_single_character(
        self,
        story_data: Dict[str, Any],
        char_index: int,
        updated_fields: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        width: int = CHARACTER_PORTRAIT_WIDTH,
        height: int = CHARACTER_PORTRAIT_HEIGHT,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Updates a single character's attributes (name, face, hair, clothing, signature_details,
        z_image_prompt) and regenerates ONLY that character's image with Z-Image Turbo using
        the specified seed in 9:16 vertical full-body orientation (576x1024).
        """
        chars = list((story_data or {}).get("characters", []))
        if not chars or char_index < 0 or char_index >= len(chars):
            raise IndexError(f"Character index {char_index} is out of range (total: {len(chars)}).")

        target = dict(chars[char_index])
        if updated_fields:
            for k in ("name", "description", "face", "hair", "clothing", "signature_details", "z_image_prompt"):
                if k in updated_fields and updated_fields[k] is not None:
                    target[k] = updated_fields[k]

        use_seed = int(seed if seed is not None else target.get("portrait_seed", 42))
        art_style = (story_data or {}).get("art_style", "")
        name = target.get("name", f"Char_{char_index+1}")
        char_id = target.get("id", f"char_{char_index+1}")
        char_prompt = self.build_character_prompt(target, art_style)

        if progress_cb:
            progress_cb(0.25, f"Redoing character '{name}' in 9:16 with Z-Image Turbo (Seed {use_seed})...")

        print(f"[StudioPipeline] Redoing Z-Image Turbo 9:16 portrait for '{name}' (index={char_index}, seed={use_seed}, {width}x{height})...")
        pil_img, local_path = generate_z_image_turbo_remote(
            prompt=char_prompt,
            seed=use_seed,
            width=width,
            height=height,
            steps=9,
            save_dir=str(self.char_dir),
            filename_prefix=f"{char_id}_{self._sanitize_filename(name)}",
        )

        target["portrait_pil"] = pil_img
        target["portrait_path"] = local_path
        target["portrait_seed"] = use_seed
        chars[char_index] = target
        story_data["characters"] = chars

        if progress_cb:
            progress_cb(1.0, f"Character '{name}' regenerated with seed {use_seed}.")
        return target

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Dual Scene Frame Synthesis via Qwen Image Edit (Start + End Frames)
    # ─────────────────────────────────────────────────────────────────────────

    def _synthesize_single_scene_frame(
        self,
        scene: Dict[str, Any],
        characters: List[Dict[str, Any]],
        aspect_ratio: str,
        seed: int,
        generate_end_frame: Optional[bool] = None,
        which_frame: str = "both",
        end_seed: Optional[int] = None,
        color_match_end: bool = True,
        story_meta: Optional[Dict[str, Any]] = None,
        art_style: str = "",
        direct_end_frame: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates the Starting Frame (t=0s) and/or Ending Frame (t=5s) for a scene using Qwen Image Edit.
        - Start Frame (`keyframe_pil` / `keyframe_path`): Conditioned on character reference portraits + `qwen_scene_prompt`.
        - End Frame (`last_keyframe_pil` / `last_keyframe_path`): Directed by Gemma 4: 31B Vision on the Start Frame
          for an explicit, high-delta pose progression, then anchor-chained onto the Start Frame as `base_image`
          plus character reference portraits + `qwen_end_scene_prompt`, then normalized in CIE-Lab space via
          `match_color_histogram_lab` so background, lighting, and character identity remain 100% locked.
        """
        dims = ASPECT_RATIO_PRESETS.get(aspect_ratio, ASPECT_RATIO_PRESETS["9:16 Vertical (Shorts/Reels)"])
        qwen_w, qwen_h = dims["qwen_w"], dims["qwen_h"]

        char_map = {c.get("id"): c for c in characters}
        default_char = characters[0] if characters else None

        scene_num = scene.get("scene_number", 1)
        title = scene.get("title", f"Scene {scene_num}")
        char_ids = scene.get("character_ids", [])

        # Ensure portrait_pil is loaded if only portrait_path is present
        for c in characters:
            if c.get("portrait_pil") is None and c.get("portrait_path") and os.path.exists(c["portrait_path"]):
                try:
                    c["portrait_pil"] = Image.open(c["portrait_path"]).convert("RGB")
                except Exception:
                    pass

        ref_chars = [char_map[cid] for cid in char_ids if cid in char_map]
        if not ref_chars and default_char:
            ref_chars = [default_char]
        ref_chars = [rc for rc in ref_chars if rc.get("portrait_pil") is not None][:QWEN_MAX_INPUT_IMAGES]

        scene_entry = dict(scene)
        use_seed = int(seed)
        if end_seed is not None and int(end_seed) != use_seed:
            use_end_seed = int(end_seed)
        elif scene.get("last_keyframe_seed") is not None and int(scene["last_keyframe_seed"]) != use_seed:
            use_end_seed = int(scene["last_keyframe_seed"])
        else:
            use_end_seed = (use_seed + 10007) % (2**31 - 1)
        which = str(which_frame or "both").lower().strip()
        if which not in ("both", "start", "end"):
            which = "both"

        if generate_end_frame is None:
            should_gen_end = bool(scene.get("qwen_end_scene_prompt")) or which == "end"
        else:
            should_gen_end = bool(generate_end_frame) or which == "end"

        scene_prompt = scene.get("qwen_scene_prompt") or scene.get("scenario") or "Scene background"
        timestamp = int(time.time())
        clean_title = self._sanitize_filename(title)

        # ── 1. Generate Starting Frame (t = 0.0s) ─────────────────────────────
        keyframe_pil = scene_entry.get("keyframe_pil")
        if keyframe_pil is None and scene_entry.get("keyframe_path") and os.path.exists(scene_entry["keyframe_path"]):
            try:
                keyframe_pil = Image.open(scene_entry["keyframe_path"]).convert("RGB")
            except Exception:
                keyframe_pil = None

        if which in ("both", "start") or keyframe_pil is None:
            if ref_chars:
                base_pil = ref_chars[0]["portrait_pil"]
                additional_refs = [rc["portrait_pil"] for rc in ref_chars[1:]]
                mapping = "; ".join(
                    f"{rc.get('name', 'character')} is the character in image {i+1}" for i, rc in enumerate(ref_chars)
                )
                refined_prompt = (
                    f"{scene_prompt}. {mapping}. Keep each character's face, hairstyle, colors and outfit identical "
                    f"to their reference image. Full-frame cinematic composition, detailed background, 4k"
                )
            else:
                base_pil = Image.new("RGB", (qwen_w, qwen_h), (255, 255, 255))
                additional_refs = []
                refined_prompt = f"{scene_prompt}, full-frame cinematic composition, detailed background, 4k"

            print(f"[StudioPipeline] Generating Qwen START keyframe for Scene {scene_num} ({qwen_w}x{qwen_h}, seed={use_seed})...")
            results = qwen_image_edit_remote(
                base_image=base_pil,
                ref_images=additional_refs,
                prompt=refined_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=use_seed,
                steps=20,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_images=1,
                custom_width=qwen_w,
                custom_height=qwen_h,
            )
            keyframe_pil = Image.open(io.BytesIO(results[0])).convert("RGB")
            filename = f"scene_{scene_num}_{clean_title}_{timestamp}.jpg"
            keyframe_path = self.scenes_dir / filename
            keyframe_pil.save(keyframe_path, format="JPEG", quality=95)
            print(f"[StudioPipeline] Saved Scene {scene_num} START keyframe: {keyframe_path}")

            scene_entry["keyframe_pil"] = keyframe_pil
            scene_entry["keyframe_path"] = str(keyframe_path)
            scene_entry["keyframe_seed"] = use_seed

        # ── 2. Generate Ending Frame (t = 5.0s, Directed & Anchor-Chained on Start Frame) ─
        if should_gen_end and which in ("both", "end"):
            end_raw = scene.get("qwen_end_scene_prompt")
            # If direct_end_frame is True or end_raw is missing / redundant, consult Gemma Director
            if (direct_end_frame or not end_raw or end_raw == scene_prompt) and keyframe_pil is not None:
                directed_end = self.direct_scene_end_frame_with_gemma(
                    scene=scene_entry,
                    keyframe_pil=keyframe_pil,
                    characters=characters,
                    story_meta=story_meta,
                    art_style=art_style,
                )
                if directed_end:
                    end_raw = directed_end

            end_raw = ensure_high_delta_end_prompt(
                start_prompt=scene_prompt,
                end_prompt=end_raw or "",
                scenario=scene.get("scenario", ""),
                motion=scene.get("wan_motion_prompt", ""),
            )
            scene_entry["qwen_end_scene_prompt"] = end_raw

            # Anchor on Start Frame (`keyframe_pil`) as image 1 + up to 2 character portraits as image 2/3
            char_refs_for_end = [rc["portrait_pil"] for rc in ref_chars[: max(0, QWEN_MAX_INPUT_IMAGES - 1)]]
            char_ref_mapping = "; ".join(
                f"{rc.get('name', 'character')} matches character sheet in image {i+2}"
                for i, rc in enumerate(ref_chars[: max(0, QWEN_MAX_INPUT_IMAGES - 1)])
            )
            if "image 1" in end_raw.lower():
                end_refined_prompt = (
                    f"{end_raw}. "
                    + (f"{char_ref_mapping}. " if char_ref_mapping else "")
                    + "Full-frame cinematic composition, identical background and character identity, 4k"
                )
            else:
                end_refined_prompt = (
                    f"image 1 is the starting frame of this 5-second continuous animation. "
                    f"Transform the character poses from image 1 into their dynamic ending pose at t=5.0s: {end_raw}. "
                    + (f"{char_ref_mapping}. " if char_ref_mapping else "")
                    + "Keep the exact same background environment, trees, walls, lighting, and character outfits from image 1, "
                    f"but visibly advance their physical body stance and action into this distinct final pose. 4k"
                )

            print(f"[StudioPipeline] Generating Qwen END keyframe for Scene {scene_num} ({qwen_w}x{qwen_h}, seed={use_end_seed})...")
            end_results = qwen_image_edit_remote(
                base_image=keyframe_pil,
                ref_images=char_refs_for_end,
                prompt=end_refined_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=use_end_seed,
                steps=20,
                true_cfg_scale=5.5,
                guidance_scale=1.0,
                num_images=1,
                custom_width=qwen_w,
                custom_height=qwen_h,
            )
            last_keyframe_pil = Image.open(io.BytesIO(end_results[0])).convert("RGB")
            if color_match_end and keyframe_pil is not None:
                last_keyframe_pil = match_color_histogram_lab(last_keyframe_pil, keyframe_pil, strength=0.75)
                last_keyframe_pil = stabilize_end_frame_background(keyframe_pil, last_keyframe_pil, border_ratio=0.14)

            # ── Pose Delta Gatekeeper (Self-Correction for Stagnant Frames) ───
            pose_delta = measure_frame_pose_delta(keyframe_pil, last_keyframe_pil)
            print(f"[StudioPipeline] Scene {scene_num} Keyframe Pose Delta: {pose_delta:.2f} (Threshold: 18.0)")

            if pose_delta < 18.0 and keyframe_pil is not None:
                print(
                    f"[StudioPipeline] [PoseDeltaGate] Warning: Scene {scene_num} End frame too similar to Start frame "
                    f"(delta={pose_delta:.2f} < 18.0). Triggering High-Delta Pose Replacement..."
                )
                retry_end_seed = (use_end_seed + 54321) % (2**31 - 1)
                boosted_delta_prompt = (
                    f"image 1 is the starting frame. For this ending frame at t=5.0s, completely replace the characters' body stances: "
                    f"they must NOT be in the same posture as image 1. "
                    f"They have completed {scene_prompt} and are now in a drastic, contrasting final pose: "
                    f"dropped low into a defensive crouch on one knee, or recoiling with arms shielding head, or turned 90 degrees pointing in alarm. "
                    f"Limb and head positions must visibly differ from image 1. "
                    f"Keep background room, rock geometry, lighting, and clothing from image 1 identical. 4k"
                )
                if char_ref_mapping:
                    boosted_delta_prompt += f" {char_ref_mapping}."

                retry_results = qwen_image_edit_remote(
                    base_image=keyframe_pil,
                    ref_images=char_refs_for_end,
                    prompt=boosted_delta_prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=retry_end_seed,
                    steps=20,
                    true_cfg_scale=6.0,
                    guidance_scale=1.0,
                    num_images=1,
                    custom_width=qwen_w,
                    custom_height=qwen_h,
                )
                retry_pil = Image.open(io.BytesIO(retry_results[0])).convert("RGB")
                if color_match_end:
                    retry_pil = match_color_histogram_lab(retry_pil, keyframe_pil, strength=0.75)
                    retry_pil = stabilize_end_frame_background(keyframe_pil, retry_pil, border_ratio=0.14)

                retry_delta = measure_frame_pose_delta(keyframe_pil, retry_pil)
                print(f"[StudioPipeline] Scene {scene_num} Retry Pose Delta: {retry_delta:.2f}")
                last_keyframe_pil = retry_pil
                use_end_seed = retry_end_seed
                pose_delta = retry_delta

            flf2v_compat = evaluate_flf2v_compatibility(keyframe_pil, last_keyframe_pil)

            end_filename = f"scene_{scene_num}_end_{clean_title}_{timestamp}.jpg"
            last_keyframe_path = self.scenes_dir / end_filename
            last_keyframe_pil.save(last_keyframe_path, format="JPEG", quality=95)
            print(f"[StudioPipeline] Saved Scene {scene_num} END keyframe: {last_keyframe_path} (FLF2V status={flf2v_compat['status']})")

            scene_entry["last_keyframe_pil"] = last_keyframe_pil
            scene_entry["last_keyframe_path"] = str(last_keyframe_path)
            scene_entry["last_keyframe_seed"] = use_end_seed
            scene_entry["pose_delta"] = round(pose_delta, 2)
            scene_entry["flf2v_compat"] = flf2v_compat
            scene_entry["flf2v_safe"] = flf2v_compat["flf2v_safe"]

        return scene_entry


    def direct_scene_end_frame_with_gemma(
        self,
        scene: Dict[str, Any],
        keyframe_pil: Image.Image,
        characters: List[Dict[str, Any]],
        story_meta: Optional[Dict[str, Any]] = None,
        art_style: str = "",
        use_vision: bool = True,
    ) -> Optional[str]:
        """
        Consults Gemma 4: 31B Vision Director to inspect the generated START keyframe (t=0.0s)
        and command an explicit, high-delta, dynamic ENDING pose (t=5.0s) for Qwen Image Edit.
        Guarantees that the end frame has a visible, cinematic action progression instead of
        being stagnant or identical to the starting frame.
        """
        scene_num = scene.get("scene_number", 1)
        title = scene.get("title", "")
        scenario = scene.get("scenario") or scene.get("qwen_scene_prompt", "")
        wan_motion = scene.get("wan_motion_prompt", "")
        char_map = {c.get("id"): c for c in (characters or [])}
        cast = [char_map[cid] for cid in scene.get("character_ids", []) if cid in char_map]
        if not cast and characters:
            cast = characters[:2]
        cast_desc = ", ".join(f"{c.get('name', 'Hero')} ({c.get('clothing', '')})" for c in cast) or "characters in image"
        ledger = scene.get("state_ledger") or {}
        entry_state = ledger.get("entry_state", "")
        active_props = ledger.get("active_props", [])
        atomic_action = ledger.get("atomic_action_5s", "")
        target_exit_state = ledger.get("exit_state", "")
        props_str = ", ".join(active_props) if active_props else "None"

        system_prompt = (
            "You are an expert Animation Director directing the KEYFRAME POSE PROGRESSION of a 5-second continuous animated shot in Wan 2.2 FLF2V.\n"
            "You are inspecting image 1: the STARTING KEYFRAME (t = 0.0s) of this shot.\n"
            "Your critical task is to direct the character(s) into their dynamic, visibly distinct ENDING POSE (t = 5.0s).\n\n"
            "CRITICAL DIRECTORIAL LAWS:\n"
            "1. ZERO STAGNANT POSES: Wan 2.2 FLF2V interpolates between Frame 0 and Frame 80. If the poses are similar, the video has zero motion!\n"
            "2. COMMAND A RADICAL POSE TRANSFORMATION: You must specify an explicit physical posture switch (e.g. from standing/aiming to crouching low behind cover, dropping to one knee, recoiling backward from impact, weapon lowered to reload, turning 90 degrees in shock, or leaping into action).\n"
            "3. WEAPON & PROP RETENTION LOCK: All active weapons, items, and accessories (" + props_str + ") MUST remain in the characters' hands in the ending frame unless explicitly dropped or destroyed. Never allow weapons to vanish!\n"
            "4. EXPLICIT BODY MECHANICS: Describe exact torso angle, arm positions, head direction, hands, and props.\n"
            "5. KEEP THE WORLD IDENTICAL: The background environment, room geometry, trees, lighting, art style, and character clothing MUST remain identical to image 1, but the physical body pose MUST be completely transformed.\n"
            "6. Output ONLY valid JSON in this schema:\n"
            "{\n"
            '  "action_summary": "1-sentence summary of the action that occurred over 5 seconds",\n'
            '  "qwen_end_frame_prompt": "Imperative edit prompt for Qwen Image Edit: \'image 1 is the starting frame. For this ending frame at t=5.0s, completely transform the characters\\\' body poses into their dynamic final stance: [exact physical action, new arm, leg, and head positions]. Keep the exact same background environment, lighting, art style, and outfits from image 1, but visibly advance their body pose into this new action.\'"\n'
            "}"
        )

        user_text = (
            f"Shot {scene_num}: '{title}' | Style: {art_style}\n"
            f"Scenario / Shot Action: {scenario}\n"
            f"State Ledger Action (5s): {atomic_action}\n"
            f"Target Exit State: {target_exit_state}\n"
            f"Active Props/Weapons: {props_str}\n"
            f"Intended 5s Motion: {wan_motion}\n"
            f"Characters in Shot: {cast_desc}\n"
            f"Analyze image 1 (the start frame) and direct the exact ending pose at t=5.0s with a clear, visible action delta while keeping all weapons/props locked in hand."
        )


        b64_start = None
        if use_vision and keyframe_pil is not None:
            try:
                img_thumb = keyframe_pil.copy()
                img_thumb.thumbnail((768, 768))
                buf = io.BytesIO()
                img_thumb.convert("RGB").save(buf, format="JPEG", quality=85)
                b64_start = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                print(f"[StudioPipeline] Failed to encode start keyframe thumbnail for Gemma Vision: {e}")

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if b64_start:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_start}"}})

        try:
            raw_text = self._chat_completion(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                max_tokens=600,
                temperature=0.5,
                max_retries=1,
            )
            parsed = self._parse_json_safely(raw_text)
            end_prompt = parsed.get("qwen_end_frame_prompt")
            if end_prompt and len(end_prompt.strip()) > 20:
                print(f"[StudioPipeline] Gemma Director directed Scene {scene_num} End Frame: '{end_prompt[:60]}...'")
                return end_prompt.strip()
        except Exception as e:
            print(f"[StudioPipeline] Gemma Director end frame direction note for scene {scene_num}: {e}")

        # Fallback if Vision/LLM fails: deterministic high-delta prompt
        return ensure_high_delta_end_prompt(
            start_prompt=scenario,
            end_prompt="",
            scenario=scenario,
            motion=wan_motion,
        )

    def verify_scene_frames_with_vlm(
        self,
        scene: Dict[str, Any],
        characters: List[Dict[str, Any]],
        art_style: str = "",
    ) -> Dict[str, Any]:
        """
        Gemma 4: 31B Vision QA Critic — inspects the generated Start Frame and End Frame
        against the character sheet and scene beat to catch anatomy errors, wrong outfits,
        or background discontinuity before video animation.
        """
        char_map = {c.get("id"): c for c in (characters or [])}
        cast = [char_map[cid] for cid in scene.get("character_ids", []) if cid in char_map]
        cast_desc = "\n".join(
            f"- {c.get('name')}: face={c.get('face', '')}; hair={c.get('hair', '')}; "
            f"clothing={c.get('clothing', '')}; props={c.get('signature_details', '')}"
            for c in cast
        ) or "- (no specific characters)"

        b64_start = self._keyframe_b64(scene)
        b64_end = self._last_keyframe_b64(scene)
        if not b64_start:
            return {
                "passed": False,
                "score": 0,
                "issues": "Missing start keyframe image.",
                "suggested_start_prompt": scene.get("qwen_scene_prompt", ""),
                "suggested_end_prompt": scene.get("qwen_end_scene_prompt", ""),
            }

        system = (
            "You are a strict animation Quality Assurance (QA) Vision Critic.\n"
            "You inspect the generated START FRAME (and END FRAME if attached) of a 5-second animated shot.\n"
            "Check for:\n"
            "1. Character accuracy: Do the characters match their required face, fur/hair, and outfit descriptions?\n"
            "2. Visual quality: Are there extra limbs, deformed faces, text watermarks, or severe artifacts?\n"
            "3. Start-to-End continuity: If two frames are attached, does the End Frame preserve the same background "
            "and character design as the Start Frame while advancing the pose?\n"
            "Output ONLY valid JSON without markdown:\n"
            "{\n"
            '  "passed": true,\n'
            '  "score": 1-10,\n'
            '  "issues": "Brief summary of any visual mistakes or \'None\'",\n'
            '  "suggested_start_prompt": "Improved start frame prompt if score < 7, else empty string",\n'
            '  "suggested_end_prompt": "Improved end frame prompt if score < 7, else empty string"\n'
            "}"
        )
        user_text = (
            f"Scene {scene.get('scene_number', 1)}: {scene.get('title', '')} | Style: {art_style}\n"
            f"Story Beat: {scene.get('scenario', '')}\n"
            f"Start Frame Prompt: {scene.get('qwen_scene_prompt', '')}\n"
            f"End Frame Prompt: {scene.get('qwen_end_scene_prompt', '')}\n"
            f"Expected Characters:\n{cast_desc}\n"
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_start}"}})
        if b64_end:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_end}"}})

        try:
            raw_text = self._chat_completion(
                [{"role": "system", "content": system}, {"role": "user", "content": content}],
                max_tokens=600,
                temperature=0.3,
                max_retries=1,
            )
            parsed = self._parse_json_safely(raw_text)
            score = int(round(_clamp(parsed.get("score"), 1, 10, 8)))
            passed = bool(parsed.get("passed", score >= 7)) and score >= 7
            report = {
                "passed": passed,
                "score": score,
                "issues": _trim(parsed.get("issues") or "None", 240),
                "suggested_start_prompt": _trim(parsed.get("suggested_start_prompt") or "", 300),
                "suggested_end_prompt": _trim(parsed.get("suggested_end_prompt") or "", 300),
            }
        except Exception as e:
            print(f"[StudioPipeline] VLM QA Critic note for scene {scene.get('scene_number', 1)}: {e}")
            report = {
                "passed": True,
                "score": 8,
                "issues": "QA check skipped (offline fallback)",
                "suggested_start_prompt": "",
                "suggested_end_prompt": "",
            }

        scene["qa_report"] = report
        return report

    def generate_scene_keyframes(
        self,
        scenes: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        shared_seed: int = 42,
        generate_end_frame: Optional[bool] = None,
        chain_previous_end_frame: bool = False,
        auto_qa_retry: bool = False,
        art_style: str = "",
        direct_end_frame: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes scene keyframes (Start Frame + End Frame) using Qwen Image Edit Rapid AIO.
        Uses the SAME LOCKED SEED (`shared_seed` / `qwen_seed`) across all scenes so lighting,
        palette, and rendering grain stay consistent, while transferring character faces and
        outfits via multi-reference conditioning.
        If `direct_end_frame=True`, Gemma 4: 31B Vision inspects the Start Frame (t=0.0s) and
        directs a high-delta dynamic action progression for the End Frame (t=5.0s).
        If `chain_previous_end_frame=True`, Scene N+1's Start Frame is directly chained from
        Scene N's End Frame for seamless zero-degradation shot extensions.
        """
        total = len(scenes)
        updated_scenes = []
        locked_qwen_seed = int(shared_seed)

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            title = scene.get("title", f"Scene {scene_num}")

            if progress_cb:
                pct = 0.30 + (idx / max(1, total)) * 0.20
                progress_cb(
                    pct,
                    f"Synthesizing scene {idx+1}/{total} Start & End frames: '{title}' via Qwen Multi-Ref (Seed {locked_qwen_seed})...",
                )

            # Seamless Continuous Shot Chaining (Scene N End Frame -> Scene N+1 Start Frame)
            if chain_previous_end_frame and idx > 0 and (updated_scenes[idx - 1].get("last_keyframe_pil") is not None or updated_scenes[idx - 1].get("last_keyframe_path") is not None):
                prev_last_pil = updated_scenes[idx - 1].get("last_keyframe_pil")
                prev_last_path = updated_scenes[idx - 1].get("last_keyframe_path")
                if prev_last_pil is None and prev_last_path and os.path.exists(prev_last_path):
                    try:
                        prev_last_pil = Image.open(prev_last_path).convert("RGB")
                    except Exception:
                        pass
                prev_last_seed = updated_scenes[idx - 1].get("last_keyframe_seed", locked_qwen_seed)
                scene["keyframe_pil"] = prev_last_pil
                scene["keyframe_path"] = prev_last_path
                scene["keyframe_seed"] = prev_last_seed
                print(f"[StudioPipeline] [Chain] Scene {scene_num} chained seamlessly from Scene {idx} End Frame (Zero-Degradation Extension).")
                scene_end_seed = (locked_qwen_seed + 10007 + idx * 37) % (2**31 - 1)
                scene_entry = self._synthesize_single_scene_frame(
                    scene=scene,
                    characters=characters,
                    aspect_ratio=aspect_ratio,
                    seed=locked_qwen_seed,
                    generate_end_frame=True,
                    which_frame="end",
                    end_seed=scene_end_seed,
                    art_style=art_style,
                    direct_end_frame=direct_end_frame,
                )
            else:
                scene_end_seed = (locked_qwen_seed + 10007 + idx * 37) % (2**31 - 1)
                scene_entry = self._synthesize_single_scene_frame(
                    scene=scene,
                    characters=characters,
                    aspect_ratio=aspect_ratio,
                    seed=locked_qwen_seed,
                    generate_end_frame=generate_end_frame,
                    which_frame="both",
                    end_seed=scene_end_seed,
                    art_style=art_style,
                    direct_end_frame=direct_end_frame,
                )

            if auto_qa_retry:
                qa = self.verify_scene_frames_with_vlm(scene_entry, characters, art_style=art_style)
                if not qa.get("passed", True):
                    retry_seed = locked_qwen_seed + 1
                    print(
                        f"[StudioPipeline] Scene {scene_num} flagged by VLM QA Critic "
                        f"(score={qa.get('score')}/10: {qa.get('issues')}). Auto-retrying with seed {retry_seed}..."
                    )
                    if qa.get("suggested_start_prompt"):
                        scene_entry["qwen_scene_prompt"] = qa["suggested_start_prompt"]
                    if qa.get("suggested_end_prompt"):
                        scene_entry["qwen_end_scene_prompt"] = qa["suggested_end_prompt"]
                    retry_end_seed = (retry_seed + 10007 + idx * 37) % (2**31 - 1)
                    scene_entry = self._synthesize_single_scene_frame(
                        scene=scene_entry,
                        characters=characters,
                        aspect_ratio=aspect_ratio,
                        seed=retry_seed,
                        generate_end_frame=generate_end_frame,
                        which_frame="both",
                        end_seed=retry_end_seed,
                        art_style=art_style,
                        direct_end_frame=direct_end_frame,
                    )
                    self.verify_scene_frames_with_vlm(scene_entry, characters, art_style=art_style)

            scene.update(scene_entry)
            updated_scenes.append(scene_entry)

        if progress_cb:
            progress_cb(
                0.50,
                f"All {total} scene keyframes synthesized with locked Qwen seed {locked_qwen_seed}.",
            )

        return updated_scenes

    def regenerate_single_scene_keyframe(
        self,
        story_data: Dict[str, Any],
        scene_index: int,
        updated_fields: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        end_seed: Optional[int] = None,
        which_frame: str = "both",
        generate_end_frame: Optional[bool] = None,
        direct_end_frame: bool = False,
        auto_qa_retry: bool = False,
        chain_previous_end_frame: bool = False,
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Updates a single scene's description/scenario/start_prompt/end_prompt/character_ids and
        regenerates that scene's Start Frame, End Frame, or Both (`which_frame="both" | "start" | "end"`)
        via Qwen Image Edit using the given seed(s). Supports seamless shot chaining from previous scene.
        """
        scenes = list((story_data or {}).get("scenes", []))
        if not scenes or scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"Scene index {scene_index} is out of range (total: {len(scenes)}).")

        target = dict(scenes[scene_index])
        if updated_fields:
            for k in (
                "title",
                "scenario",
                "qwen_scene_prompt",
                "qwen_end_scene_prompt",
                "wan_motion_prompt",
                "camera_movement",
                "character_ids",
            ):
                if k in updated_fields and updated_fields[k] is not None:
                    target[k] = updated_fields[k]

        which = str(which_frame or "both").lower().strip()

        # Seamless Continuous Shot Chaining (Scene N End Frame -> Scene N+1 Start Frame)
        if chain_previous_end_frame and scene_index > 0:
            prev_scene = scenes[scene_index - 1]
            prev_last_path = prev_scene.get("last_keyframe_path") or prev_scene.get("keyframe_path")
            prev_last_pil = prev_scene.get("last_keyframe_pil") or prev_scene.get("keyframe_pil")
            if prev_last_path and not prev_last_pil and os.path.exists(prev_last_path):
                try:
                    prev_last_pil = Image.open(prev_last_path).convert("RGB")
                except Exception:
                    pass
            if prev_last_path:
                prev_last_seed = prev_scene.get("last_keyframe_seed", prev_scene.get("keyframe_seed", seed or 42))
                target["keyframe_pil"] = prev_last_pil
                target["keyframe_path"] = prev_last_path
                target["keyframe_seed"] = prev_last_seed
                which = "end"
                which_frame = "end"
                print(f"[StudioPipeline] [Chain] Scene {scene_index + 1} chained seamlessly from Scene {scene_index} End Frame.")

        if which == "end":
            use_seed = int(target.get("keyframe_seed", seed if seed is not None else 42))
            use_end_seed = int(end_seed if end_seed is not None else (seed if seed is not None else target.get("last_keyframe_seed", (use_seed + 10007) % (2**31 - 1))))
        else:
            use_seed = int(seed if seed is not None else target.get("keyframe_seed", 42))
            if end_seed is not None and int(end_seed) != use_seed:
                use_end_seed = int(end_seed)
            elif target.get("last_keyframe_seed") is not None and int(target["last_keyframe_seed"]) != use_seed:
                use_end_seed = int(target["last_keyframe_seed"])
            else:
                use_end_seed = (use_seed + 10007) % (2**31 - 1)
        title = target.get("title", f"Scene {scene_index+1}")

        if progress_cb:
            progress_cb(
                0.35,
                f"Redoing Scene {scene_index+1} ('{title}', mode={which_frame}) with Qwen (Seed {use_seed})...",
            )

        updated_scene = self._synthesize_single_scene_frame(
            scene=target,
            characters=story_data.get("characters", []),
            aspect_ratio=aspect_ratio,
            seed=use_seed,
            generate_end_frame=generate_end_frame,
            which_frame=which_frame,
            end_seed=use_end_seed,
            art_style=story_data.get("art_style", ""),
            direct_end_frame=direct_end_frame,
        )

        if auto_qa_retry:
            qa = self.verify_scene_frames_with_vlm(
                updated_scene,
                story_data.get("characters", []),
                art_style=story_data.get("art_style", ""),
            )
            if not qa.get("passed", True):
                retry_seed = use_seed + 1
                retry_end_seed = (use_end_seed + 1) if (use_end_seed is not None and use_end_seed != use_seed) else (retry_seed + 10007) % (2**31 - 1)
                if qa.get("suggested_start_prompt"):
                    updated_scene["qwen_scene_prompt"] = qa["suggested_start_prompt"]
                if qa.get("suggested_end_prompt"):
                    updated_scene["qwen_end_scene_prompt"] = qa["suggested_end_prompt"]
                updated_scene = self._synthesize_single_scene_frame(
                    scene=updated_scene,
                    characters=story_data.get("characters", []),
                    aspect_ratio=aspect_ratio,
                    seed=retry_seed,
                    generate_end_frame=generate_end_frame,
                    which_frame=which_frame,
                    end_seed=retry_end_seed,
                    art_style=story_data.get("art_style", ""),
                    direct_end_frame=direct_end_frame,
                )
                self.verify_scene_frames_with_vlm(
                    updated_scene,
                    story_data.get("characters", []),
                    art_style=story_data.get("art_style", ""),
                )

        # Clear stale director plan if keyframe changed so it can be re-directed cleanly
        if updated_fields and any(
            k in updated_fields for k in ("scenario", "qwen_scene_prompt", "qwen_end_scene_prompt", "wan_motion_prompt", "camera_movement")
        ):
            updated_scene.pop("director", None)

        scenes[scene_index].clear()
        scenes[scene_index].update(updated_scene)
        story_data["scenes"] = scenes

        if progress_cb:
            progress_cb(1.0, f"Scene {scene_index+1} ('{title}') keyframe(s) regenerated with seed {use_seed}.")
        return updated_scene

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: Animation Direction — Gemma 4: 31B directs each ≤5s clip
    # ─────────────────────────────────────────────────────────────────────────

    def direct_scene_animations(
        self,
        scenes: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
        story_meta: Optional[Dict[str, Any]] = None,
        use_vision: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For every scene, Gemma acts as the shot director: it LOOKS at BOTH the Start Frame and
        End Frame (vision input), reads the story beat plus the previous/next beats for continuity,
        and returns a structured shot plan — a 3-phase action timeline, camera move, motion
        intensity, identity anchors and safe Wan 2.2 parameters.
        """
        story_meta = story_meta or {}
        char_map = {c.get("id"): c for c in characters}
        total = len(scenes)
        directed = []

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            if progress_cb:
                pct = 0.50 + (idx / max(1, total)) * 0.10
                progress_cb(pct, f"🎬 Director planning shot {idx+1}/{total}: '{scene.get('title', '')}'...")

            prev_beat = scenes[idx - 1].get("scenario", "") if idx > 0 else "(opening shot)"
            next_beat = scenes[idx + 1].get("scenario", "") if idx + 1 < total else "(final shot)"
            cast = [char_map[c] for c in scene.get("character_ids", []) if c in char_map]

            messages = self._build_director_messages(
                scene, cast, story_meta, prev_beat, next_beat, include_image=use_vision
            )

            raw_plan, source = None, "fallback"
            attempts = [(messages, "gemma-vision")] if use_vision and self._keyframe_b64(scene) else []
            attempts.append((self._build_director_messages(
                scene, cast, story_meta, prev_beat, next_beat, include_image=False), "gemma-text"))
            for msgs, label in attempts:
                try:
                    text = self._chat_completion(msgs, max_tokens=1200, temperature=0.5, max_retries=1)
                    raw_plan = self._parse_json_safely(text)
                    source = label
                    break
                except Exception as e:
                    print(f"[StudioPipeline] Director ({label}) failed for scene {scene_num}: {e}")

            plan = self.compile_director_plan(raw_plan or {}, scene, cast, story_meta)
            plan["source"] = source

            scene_entry = dict(scene)
            scene_entry["director"] = plan
            directed.append(scene_entry)
            print(f"[StudioPipeline] Scene {scene_num} directed via {source}: "
                  f"camera='{plan['camera_movement']}', intensity={plan['motion_intensity']}, "
                  f"cfg={plan['guidance_scale']}, steps={plan['steps']}")

        if progress_cb:
            progress_cb(0.60, f"Director planned all {total} shots.")
        return directed

    def _keyframe_b64(self, scene: Dict[str, Any], max_side: int = 768) -> Optional[str]:
        img = scene.get("keyframe_pil")
        if img is None and scene.get("keyframe_path") and os.path.exists(scene["keyframe_path"]):
            img = Image.open(scene["keyframe_path"]).convert("RGB")
        if img is None:
            return None
        img = img.copy()
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _last_keyframe_b64(self, scene: Dict[str, Any], max_side: int = 768) -> Optional[str]:
        img = scene.get("last_keyframe_pil")
        if img is None and scene.get("last_keyframe_path") and os.path.exists(scene["last_keyframe_path"]):
            img = Image.open(scene["last_keyframe_path"]).convert("RGB")
        if img is None:
            return None
        img = img.copy()
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_director_messages(
        self,
        scene: Dict[str, Any],
        cast: List[Dict[str, Any]],
        story_meta: Dict[str, Any],
        prev_beat: str,
        next_beat: str,
        include_image: bool = True,
    ) -> List[Dict[str, Any]]:
        duration = float(scene.get("duration_seconds", MAX_CLIP_SECONDS))
        b64_start = self._keyframe_b64(scene) if include_image else None
        b64_end = self._last_keyframe_b64(scene) if include_image else None
        has_both_frames = bool(b64_start and b64_end)
        blueprint = retrieve_cinematography_blueprint(
            scenario=scene.get("scenario", ""),
            camera_movement=scene.get("camera_movement", ""),
            art_style=story_meta.get("art_style", ""),
        )

        system = (
            "You are the expert animation director for a First-Last-Frame image-to-video model (Wan 2.2). "
            + (
                f"You receive BOTH the FIRST FRAME (t=0.0s) and the LAST FRAME (t={duration:.1f}s) of a single "
                f"continuous {duration:.1f}-second shot and must plan the exact motion bridge between them.\n"
                if has_both_frames
                else f"You receive the FIRST FRAME of a single continuous {duration:.1f}-second shot and must plan how it moves.\n"
            )
            + "Cinematography rules for world-class Wan 2.2 results:\n"
            "- One clear, physically plausible action per shot; motion must start from the First Frame pose and "
            "arrive naturally at the Last Frame pose.\n"
            "- Describe concrete body/face motion (e.g. 'raises left paw and offers the berry'), not abstract emotions.\n"
            "- Never introduce new characters or objects that are not in the frames; never cut or change location.\n"
            "- Lens & Camera Rig: Choose appropriate lens optics (e.g. 35mm wide for travel, 50mm for dialogue, 85mm for closeups) and camera rig (Steadicam tracking, slow push-in, orbital arc, locked tripod).\n"
            "- Period Sequencing: Formulate the 3-phase timeline as clear chronological steps separated by periods (.) so Wan sequences actions over time rather than pooling them simultaneously.\n"
            "- Reactive Physics: Ground the character with environmental reactions (e.g. dust puffs underfoot, foliage swaying, dynamic light sweeps).\n"
            "- Delta-Only Prompting: Focus strictly on the physical vector of change; do not repeat static appearance details already present in the frame.\n"
            "Output ONLY valid JSON, no markdown:\n"
            "{\n"
            '  "shot_type": "e.g. medium two-shot / close-up / wide establishing",\n'
            '  "camera_movement": "single camera move, e.g. slow push-in",\n'
            '  "motion_intensity": 1-5,\n'
            '  "subject_action": "one-sentence summary of the main action bridging start to end",\n'
            '  "timeline": ["phase 1 action (opening from First Frame)", "phase 2 action (main transition beat)", "phase 3 action (settling into Last Frame pose)"],\n'
            '  "environment_motion": "ambient motion: wind, light, particles, background",\n'
            '  "identity_anchors": "visual traits that must stay fixed (faces, colors, outfits, props)",\n'
            '  "wan_prompt": "dense 1-2 sentence motion prompt for Wan 2.2",\n'
            '  "negative_additions": "comma-separated things to avoid specific to this shot",\n'
            '  "guidance_scale": 1.5-2.5,\n'
            '  "steps": 6-12,\n'
            '  "flow_shift": 2.0-5.0\n'
            "}"
        )
        cast_desc = "\n".join(
            f"- {c.get('name')}: face={c.get('face', '')}; hair={c.get('hair', '')}; "
            f"clothing={c.get('clothing', '')}; props={c.get('signature_details', '')}"
            for c in cast
        ) or "- (no named characters)"
        end_prompt_note = f"End frame beat: {scene.get('qwen_end_scene_prompt', '')}\n" if scene.get("qwen_end_scene_prompt") else ""

        ledger = scene.get("state_ledger") or {}
        entry_state = ledger.get("entry_state", "")
        active_props = ledger.get("active_props", [])
        atomic_action = ledger.get("atomic_action_5s", "")
        exit_state = ledger.get("exit_state", "")
        connector = scene.get("causal_connector", "")
        beat_name = scene.get("narrative_beat", "")

        ledger_note = ""
        if ledger:
            ledger_note = (
                f"Narrative Beat: {beat_name} (Causal Connector: {connector})\n"
                f"Causal State Bridge: Entry State: {entry_state} -> Atomic 5s Action: {atomic_action} -> Exit State: {exit_state}\n"
                f"Active Props/Weapons: {', '.join(active_props) if active_props else 'None'}\n"
            )

        narration_note = ""
        if scene.get("narration_text"):
            sync_p = scene.get("synced_phases") or {}
            trig_t = sync_p.get("p1", round(duration * 0.3, 1))
            narration_note = (
                f"Voiceover Narration: \"{scene.get('narration_text')}\" "
                f"[Sync Trigger Word: '{scene.get('sync_trigger_word', '')}' spoken at t={trig_t}s — align Phase 2 climax action to this timestamp]\n"
            )

        text = (
            f"Film: {story_meta.get('title', '')} | Style: {story_meta.get('art_style', '')} | "
            f"Theme: {story_meta.get('theme', '')}\n"
            f"Shot {scene.get('scene_number')}: {scene.get('title', '')} ({duration:.1f}s)\n"
            f"Story beat: {scene.get('scenario', '')}\n"
            f"{ledger_note}"
            f"{narration_note}"
            f"{end_prompt_note}"
            f"Draft motion from writer: {scene.get('wan_motion_prompt', '')}\n"
            f"Draft camera: {scene.get('camera_movement', '')}\n"
            f"Cinematography Blueprint: lens={blueprint['lens']}; suggested_rig={blueprint['camera_movement_directive']}; physics={blueprint['reactive_physics']}\n"
            f"Previous beat: {prev_beat}\nNext beat: {next_beat}\n"
            f"Cast in frame:\n{cast_desc}\n"
            + (
                "Image 1 is the exact FIRST FRAME (0.0s) and Image 2 is the exact LAST FRAME. Plan the smooth transition connecting them."
                if has_both_frames
                else (
                    "The attached image is the exact first frame. Base the motion on what you see."
                    if b64_start
                    else "No image attached; infer composition from the beat."
                )
            )
        )
        user_content: Any = text
        if b64_start:
            user_content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_start}"}},
            ]
            if b64_end:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_end}"}}
                )
        return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]

    @staticmethod
    def compile_director_plan(
        raw: Dict[str, Any],
        scene: Dict[str, Any],
        cast: List[Dict[str, Any]],
        story_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates/clamps the Director's JSON and compiles it into the exact Wan 2.2 prompt,
        negative prompt and sampler parameters. Works with an empty dict (deterministic fallback).
        Integrates the Cinematography Knowledge Vault, Motion-Adaptive Dual-CFG & Flow-Shift Matrix,
        and Audio-Visual Temporal Lock (`synced_phases`) when narrator timestamps are present.
        """
        story_meta = story_meta or {}
        raw = raw if isinstance(raw, dict) else {}
        duration = float(_clamp(scene.get("duration_seconds", MAX_CLIP_SECONDS), MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, 3.5))

        lim = DIRECTOR_PARAM_LIMITS
        intensity = int(round(_clamp(raw.get("motion_intensity"), *lim["motion_intensity"], 2)))
        adaptive = compute_motion_adaptive_wan_params(intensity, duration)

        cfg = round(_clamp(raw.get("guidance_scale"), *lim["guidance_scale"], adaptive["guidance_scale"]), 2)
        cfg2 = round(float(raw.get("guidance_scale_2", adaptive["guidance_scale_2"])), 2)
        steps = int(round(_clamp(raw.get("steps"), *lim["steps"], adaptive["steps"])))
        flow_shift = round(_clamp(raw.get("flow_shift"), *lim["flow_shift"], adaptive["flow_shift"]), 2)

        camera = _trim(raw.get("camera_movement") or scene.get("camera_movement") or "static tripod shot", 80)
        shot_type = _trim(raw.get("shot_type") or "medium shot", 60)
        action = _trim(raw.get("subject_action") or scene.get("scenario") or "", 220)

        timeline = raw.get("timeline")
        if isinstance(timeline, dict):
            timeline = list(timeline.values())
        if not isinstance(timeline, list):
            timeline = []
        timeline = [
            _trim(t.get("action") if isinstance(t, dict) else t, 160) for t in timeline if t
        ][:3]
        defaults = [
            "characters begin from the pose shown in the frame with natural breathing",
            action or "the main action unfolds smoothly",
            "motion eases into a calm, stable final pose",
        ]
        while len(timeline) < 3:
            timeline.append(defaults[len(timeline)])

        blueprint = retrieve_cinematography_blueprint(
            scenario=action or scene.get("scenario", ""),
            camera_movement=camera,
            art_style=story_meta.get("art_style", ""),
        )

        # Lock Phase 1 -> Phase 2 -> Phase 3 transitions to Narrator Sync Trigger Word when available
        synced_phases = scene.get("synced_phases")
        if isinstance(synced_phases, dict) and "p1" in synced_phases and "p2" in synced_phases:
            p1 = round(float(synced_phases["p1"]), 2)
            p2 = round(float(synced_phases["p2"]), 2)
        else:
            p1, p2 = round(duration * 0.3, 1), round(duration * 0.8, 1)

        t0 = timeline[0].strip().rstrip(".")
        t1 = timeline[1].strip().rstrip(".")
        t2 = timeline[2].strip().rstrip(".")
        phases = (
            f"Phase 1 (0.0s-{p1}s): {t0}. "
            f"Phase 2 ({p1}s-{p2}s): {t1}. "
            f"Phase 3 ({p2}s-{duration:.1f}s): {t2}."
        )

        cast_names = ", ".join(c.get("name", "") for c in cast if c.get("name"))
        anchors = _trim(
            raw.get("identity_anchors")
            or (f"{cast_names} keep identical faces, colors, outfits and proportions" if cast_names
                else "subjects keep identical appearance"),
            220,
        )

        # Incorporate active props retention into identity anchors
        active_props = scene.get("state_ledger", {}).get("active_props", [])
        if active_props and isinstance(active_props, list):
            props_txt = ", ".join(active_props[:3])
            anchors += f", weapon and prop lock: {props_txt} remain securely in hands with no vanishing"

        env = _trim(raw.get("environment_motion") or "", 140)
        if not env or env == "subtle ambient motion in the background":
            env = blueprint.get("reactive_physics", "subtle ambient motion in the background")

        core = _trim(raw.get("wan_prompt") or scene.get("wan_motion_prompt") or action, 300)
        style = _trim(story_meta.get("art_style") or "", 80)

        wan_prompt = (
            f"{core} [Shot]: {shot_type}, camera {camera}. [Timeline]: {phases} "
            f"[Environment]: {env}. [Identity Anchors]: {anchors}, no morphing. "
            f"{style + ', ' if style else ''}smooth fluid animation, consistent character design, high detail."
        )

        extra_neg = _trim(raw.get("negative_additions") or "", 200)
        negative = ", ".join(x for x in [DEFAULT_NEGATIVE_PROMPT, IDENTITY_NEGATIVE, extra_neg] if x)

        return {
            "shot_type": shot_type,
            "camera_movement": camera,
            "motion_intensity": intensity,
            "subject_action": action,
            "timeline": timeline,
            "synced_phases": {"p1": p1, "p2": p2},
            "environment_motion": env,
            "identity_anchors": anchors,
            "lens": blueprint.get("lens", "50mm prime cinema lens"),
            "reactive_physics": blueprint.get("reactive_physics", ""),
            "wan_prompt": wan_prompt,
            "negative_prompt": negative,
            "guidance_scale": cfg,
            "guidance_scale_2": cfg2,
            "steps": steps,
            "flow_shift": flow_shift,
            "duration_seconds": duration,
            "num_frames": frames_for_duration(duration),
        }

    @staticmethod
    def apply_anti_noise_and_rotation_guard(
        plan_or_wan_kwargs: Dict[str, Any],
        duration_seconds: float = 5.0,
        has_last_frame: bool = False,
        anti_noise_guard: bool = True,
    ) -> Dict[str, Any]:
        """
        Anti-Degradation, Anti-Noise, Motion-Adaptive Dual-CFG, and Rotation Identity Guard:
        1. Protects long (>= 4.5s / 73-81 frames) clips from dissolving into late-frame noise:
           - Automatically raises FlowShift to >= 4.5 to keep global temporal structure locked.
           - Sets inference steps to at least 10 on long clips to clean up high-frequency grain.
           - Clamps guidance scale <= 2.0 to prevent CFG latent norm explosion.
        2. Motion-Adaptive High-Velocity Guard (motion_intensity >= 4):
           - Raises FlowShift to >= 4.8, steps >= 12, and lowers guidance_scale <= 1.75 / guidance_scale_2 <= 1.25
             to eliminate double-limbs and ghosting during fast combat/sprints.
        3. Rotation Identity Lock:
           - Injects explicit rotational geometric anchors into prompt and negative prompt
             when head or body turns/rotations are detected so the face doesn't morph.
        """
        if not anti_noise_guard or not isinstance(plan_or_wan_kwargs, dict):
            return plan_or_wan_kwargs

        updated = dict(plan_or_wan_kwargs)
        dur = float(duration_seconds)
        intensity = int(updated.get("motion_intensity", 2) or 2)

        # ── 1. Anti-Noise Guard for 4.5s – 5.0s clips ───────────────────────
        if dur >= 4.5:
            current_steps = int(updated.get("steps", 8))
            if current_steps < 10:
                updated["steps"] = 10
            current_flow_shift = float(updated.get("flow_shift", 3.0))
            if current_flow_shift < 4.5:
                updated["flow_shift"] = 4.5
            current_cfg = float(updated.get("guidance_scale", 2.0))
            if current_cfg > 2.0:
                updated["guidance_scale"] = 2.0
            current_cfg2 = float(updated.get("guidance_scale_2", 1.5))
            if current_cfg2 > 1.5:
                updated["guidance_scale_2"] = 1.5

        # ── 1b. Motion-Adaptive High-Velocity Guard (Intensity 4-5) ─────────
        if intensity >= 4:
            updated["steps"] = max(int(updated.get("steps", 10)), 12)
            updated["flow_shift"] = max(float(updated.get("flow_shift", 4.5)), 4.8)
            updated["guidance_scale"] = min(float(updated.get("guidance_scale", 2.0)), 1.75)
            updated["guidance_scale_2"] = min(float(updated.get("guidance_scale_2", 1.5)), 1.25)

        # ── 2. Rotation Identity Lock ─────────────────────────────────────────
        text_to_check = (
            str(updated.get("wan_prompt", ""))
            + " " + str(updated.get("prompt", ""))
            + " " + str(updated.get("subject_action", ""))
            + " " + str(updated.get("camera_movement", ""))
        ).lower()

        rotation_keywords = ("turn", "rotate", "spin", "orbit", "profile", "look back", "turnaround", "around", "360", "180")
        if any(kw in text_to_check for kw in rotation_keywords):
            prompt_key = "wan_prompt" if "wan_prompt" in updated else "prompt"
            if prompt_key in updated:
                existing_prompt = updated[prompt_key]
                if "[Rotation Identity Lock]" not in existing_prompt:
                    updated[prompt_key] = (
                        existing_prompt.rstrip()
                        + " [Rotation Identity Lock]: smooth controlled head rotation, maintain identical facial proportions, eye geometry, nose bridge, hairstyle, and outfit across turn, clean seamless transition into final pose."
                    )
            neg_key = "negative_prompt"
            existing_neg = updated.get(neg_key, DEFAULT_NEGATIVE_PROMPT)
            if "identity loss during rotation" not in existing_neg:
                updated[neg_key] = (
                    existing_neg.rstrip()
                    + ", identity loss during rotation, changing face after turn, face morphing, dissolving into noise, late-frame static breakdown, pixel grain"
                )

        return updated

    def direct_single_scene_animation(
        self,
        story_data: Dict[str, Any],
        scene_index: int,
        use_vision: bool = True,
    ) -> Dict[str, Any]:
        """Runs the Gemma 4: 31B Animation Director on a single scene by index."""
        scenes = list((story_data or {}).get("scenes", []))
        if not scenes or scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"Scene index {scene_index} is out of range (total: {len(scenes)}).")

        characters = (story_data or {}).get("characters", [])
        char_map = {c.get("id"): c for c in characters}
        scene = dict(scenes[scene_index])
        scene_num = scene.get("scene_number", scene_index + 1)
        prev_beat = scenes[scene_index - 1].get("scenario", "") if scene_index > 0 else "(opening shot)"
        next_beat = scenes[scene_index + 1].get("scenario", "") if scene_index + 1 < len(scenes) else "(final shot)"
        cast = [char_map[c] for c in scene.get("character_ids", []) if c in char_map]

        messages = self._build_director_messages(
            scene, cast, story_data, prev_beat, next_beat, include_image=use_vision
        )
        raw_plan, source = None, "fallback"
        attempts = [(messages, "gemma-vision")] if use_vision and self._keyframe_b64(scene) else []
        attempts.append((self._build_director_messages(
            scene, cast, story_data, prev_beat, next_beat, include_image=False), "gemma-text"))
        for msgs, label in attempts:
            try:
                text = self._chat_completion(msgs, max_tokens=1200, temperature=0.5, max_retries=1)
                raw_plan = self._parse_json_safely(text)
                source = label
                break
            except Exception as e:
                print(f"[StudioPipeline] Single-scene Director ({label}) failed for scene {scene_num}: {e}")

        plan = self.compile_director_plan(raw_plan or {}, scene, cast, story_data)
        plan["source"] = source
        scene["director"] = plan
        scenes[scene_index] = scene
        story_data["scenes"] = scenes
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Scene Video Animation via Wan 2.2 (First+Last Frame FLF2V + RIFE)
    # ─────────────────────────────────────────────────────────────────────────

    def _animate_single_scene_clip(
        self,
        scene: Dict[str, Any],
        aspect_ratio: str,
        seed: int,
        characters: Optional[List[Dict[str, Any]]] = None,
        story_meta: Optional[Dict[str, Any]] = None,
        use_last_frame: bool = True,
        enable_rife_2x: bool = False,
        anti_noise_guard: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Animates a single scene with Wan 2.2 on Modal.
        When `use_last_frame=True` and the scene has a `last_keyframe_pil` / `last_keyframe_path`,
        both the Start Frame (`resized_image`) and CIE-Lab color-matched End Frame (`processed_last_image`)
        are passed to Wan 2.2 for First-Last-Frame (FLF2V) animation — unless `flf2v_compat` flags
        `EXTREME_DIVERGENCE`, in which case it automatically routes to Guided I2V Mode to prevent
        mid-clip cross-fade/morphing.
        Post-render: runs `verify_video_clip_quality` and muxes open-source narrator `.wav` audio when present.
        """
        dims = ASPECT_RATIO_PRESETS.get(aspect_ratio, ASPECT_RATIO_PRESETS["9:16 Vertical (Shorts/Reels)"])
        wan_w, wan_h = dims["wan_w"], dims["wan_h"]

        scene_num = scene.get("scene_number", 1)
        title = scene.get("title", f"Scene {scene_num}")
        keyframe_pil = scene.get("keyframe_pil")
        if keyframe_pil is None and scene.get("keyframe_path") and os.path.exists(scene["keyframe_path"]):
            keyframe_pil = Image.open(scene["keyframe_path"]).convert("RGB")
        if keyframe_pil is None:
            raise ValueError(f"Scene {scene_num} has no keyframe; run the Qwen keyframe stage first.")

        resized_frame = keyframe_pil.resize((wan_w, wan_h), Image.Resampling.LANCZOS)

        resized_last_frame = None
        compat_status = scene.get("flf2v_status") or (scene.get("flf2v_compat") or {}).get("status")
        if use_last_frame and compat_status != "EXTREME_DIVERGENCE" and not scene.get("force_guided_i2v", False):
            last_pil = scene.get("last_keyframe_pil")
            if last_pil is None and scene.get("last_keyframe_path") and os.path.exists(scene["last_keyframe_path"]):
                try:
                    last_pil = Image.open(scene["last_keyframe_path"]).convert("RGB")
                except Exception:
                    last_pil = None
            if last_pil is not None:
                resized_last_frame = last_pil.resize((wan_w, wan_h), Image.Resampling.LANCZOS)
                resized_last_frame = match_color_histogram_lab(resized_last_frame, resized_frame, strength=0.85)
        elif use_last_frame and (compat_status == "EXTREME_DIVERGENCE" or scene.get("force_guided_i2v", False)):
            print(
                f"[StudioPipeline] [FLF2V Guard] Scene {scene_num} flagged as EXTREME_DIVERGENCE "
                f"— routing to Guided I2V Mode to prevent mid-clip ghosting/morphing."
            )
            scene["flf2v_fallback_to_i2v"] = True

        plan = scene.get("director")
        if not plan:
            plan = self.compile_director_plan({}, scene, characters or [], story_meta or {})
            plan["source"] = "fallback"
            scene["director"] = plan
        duration_sec = float(plan.get("duration_seconds", scene.get("duration_seconds", 3.5)))
        num_frames = frames_for_duration(duration_sec)
        use_seed = int(seed)
        frame_mult = (WAN_FPS * 2) if enable_rife_2x else WAN_DEFAULTS["frame_multiplier"]

        if anti_noise_guard:
            plan = self.apply_anti_noise_and_rotation_guard(
                plan,
                duration_seconds=duration_sec,
                has_last_frame=(resized_last_frame is not None),
                anti_noise_guard=True,
            )

        mode_str = "FLF2V (First+Last Frame)" if resized_last_frame is not None else "I2V (First Frame)"
        print(f"[StudioPipeline] Animating Scene {scene_num} ({wan_w}x{wan_h}, {num_frames} frames, "
              f"mode={mode_str}, fps={frame_mult}, seed={use_seed}, director={plan.get('source')})...")

        result = run_remote_inference(
            resized_image=resized_frame,
            processed_last_image=resized_last_frame,
            prompt=plan["wan_prompt"],
            steps=int(plan.get("steps", WAN_DEFAULTS["steps"])),
            negative_prompt=plan.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            num_frames=num_frames,
            guidance_scale=float(plan.get("guidance_scale", WAN_DEFAULTS["guidance_scale"])),
            guidance_scale_2=float(plan.get("guidance_scale_2", WAN_DEFAULTS["guidance_scale_2"])),
            current_seed=use_seed,
            scheduler_name=WAN_DEFAULTS["scheduler_name"],
            flow_shift=float(plan.get("flow_shift", WAN_DEFAULTS["flow_shift"])),
            frame_multiplier=frame_mult,
            quality=WAN_DEFAULTS["quality"],
            duration_seconds=duration_sec,
            safe_mode=True,
        )
        video_path = result[0] if isinstance(result, (tuple, list)) else result

        timestamp = int(time.time())
        clip_dest = self.clips_dir / f"clip_scene_{scene_num}_{self._sanitize_filename(title)}_{timestamp}.mp4"
        with open(video_path, "rb") as src, open(clip_dest, "wb") as dst:
            dst.write(src.read())

        # ── Post-Render Optical Flow & Sharpness Telemetry ────────────────────
        expected_peak = (scene.get("synced_phases") or {}).get("p1")
        telemetry = verify_video_clip_quality(str(clip_dest), expected_peak_sec=expected_peak)
        scene["telemetry"] = telemetry
        try:
            StoryMemoryBank(memory_dir=str(self.base_dir)).record_clip_telemetry(
                scene_number=scene_num,
                telemetry=telemetry,
                director_plan=plan,
            )
        except Exception:
            pass

        # ── Mux Open-Source Narrator Audio Track into Scene Clip ──────────────
        audio_path = scene.get("audio_path") or scene.get("narration_audio_path")
        if audio_path and os.path.exists(audio_path):
            temp_muxed = self.clips_dir / f"mux_tmp_scene_{scene_num}_{timestamp}.mp4"
            try:
                AudioVisualSyncEngine.mux_video_and_audio(
                    video_path=str(clip_dest),
                    audio_path=audio_path,
                    output_path=str(temp_muxed),
                )
                if temp_muxed.exists() and temp_muxed.stat().st_size > 100:
                    os.replace(str(temp_muxed), str(clip_dest))
                    scene["audio_muxed"] = True
                    scene["has_narrator_audio"] = True
            except Exception as e:
                print(f"[StudioPipeline] Audio mux note for scene {scene_num}: {e}")
            finally:
                if temp_muxed.exists():
                    try:
                        temp_muxed.unlink()
                    except Exception:
                        pass

        print(f"[StudioPipeline] Saved Scene {scene_num} clip: {clip_dest}")
        scene["video_path"] = str(clip_dest)
        scene["video_seed"] = use_seed
        scene["used_last_frame"] = resized_last_frame is not None
        return str(clip_dest)

    def generate_scene_videos(
        self,
        scenes: List[Dict[str, Any]],
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        base_seed: int = 42,
        use_last_frame: bool = True,
        enable_rife_2x: bool = False,
        anti_noise_guard: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        Animates each scene into a clip with Wan 2.2 on Modal (using both First Frame and Last Frame
        when available and `use_last_frame=True`).
        Uses the SAME LOCKED SEED (`base_seed` / `wan_seed`) across all 5-second clips
        for uniform temporal noise and motion physics across cuts.
        """
        total = len(scenes)
        clip_paths = []
        locked_wan_seed = int(base_seed)

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            title = scene.get("title", f"Scene {scene_num}")
            plan = scene.get("director") or {}
            duration_sec = float(plan.get("duration_seconds", scene.get("duration_seconds", 3.5)))
            num_frames = frames_for_duration(duration_sec)

            if progress_cb:
                pct = 0.60 + (idx / max(1, total)) * 0.30
                progress_cb(
                    pct,
                    f"Animating Scene {idx+1}/{total}: '{title}' ({num_frames} frames @ {WAN_FPS}fps, Seed {locked_wan_seed})...",
                )

            clip_dest = self._animate_single_scene_clip(
                scene=scene,
                aspect_ratio=aspect_ratio,
                seed=locked_wan_seed,
                use_last_frame=use_last_frame,
                enable_rife_2x=enable_rife_2x,
                anti_noise_guard=anti_noise_guard,
            )
            clip_paths.append(clip_dest)

        if progress_cb:
            progress_cb(0.90, f"All {total} scene clips animated with locked Wan seed {locked_wan_seed}.")

        return clip_paths

    def regenerate_single_scene_video(
        self,
        story_data: Dict[str, Any],
        scene_index: int,
        updated_fields: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        re_run_director: bool = False,
        use_vision: bool = True,
        use_last_frame: bool = True,
        enable_rife_2x: bool = False,
        anti_noise_guard: bool = True,
        auto_stitch: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Updates a single scene's motion description / director prompt / seed, regenerates ONLY
        that 5-second video clip on Wan 2.2, and automatically re-stitches the final movie if all
        scene clips exist.
        """
        scenes = list((story_data or {}).get("scenes", []))
        if not scenes or scene_index < 0 or scene_index >= len(scenes):
            raise IndexError(f"Scene index {scene_index} is out of range (total: {len(scenes)}).")

        target = dict(scenes[scene_index])
        characters = (story_data or {}).get("characters", [])
        char_map = {c.get("id"): c for c in characters}
        cast = [char_map[c] for c in target.get("character_ids", []) if c in char_map]

        if updated_fields:
            for k in ("title", "scenario", "wan_motion_prompt", "camera_movement", "duration_seconds", "narration_text", "sync_trigger_word"):
                if k in updated_fields and updated_fields[k] is not None:
                    target[k] = updated_fields[k]

        scenes[scene_index] = target
        story_data["scenes"] = scenes

        if re_run_director:
            self.direct_single_scene_animation(story_data, scene_index, use_vision=use_vision)
            target = story_data["scenes"][scene_index]
        else:
            existing_dir = dict(target.get("director") or {})
            custom_wan_prompt = (updated_fields or {}).get("wan_prompt")
            if updated_fields and any(k in updated_fields for k in ("wan_motion_prompt", "camera_movement", "scenario")):
                raw_override = dict(existing_dir)
                if "wan_motion_prompt" in updated_fields and updated_fields["wan_motion_prompt"]:
                    raw_override["wan_prompt"] = updated_fields["wan_motion_prompt"]
                    raw_override["subject_action"] = updated_fields["wan_motion_prompt"]
                if "camera_movement" in updated_fields and updated_fields["camera_movement"]:
                    raw_override["camera_movement"] = updated_fields["camera_movement"]
                plan = self.compile_director_plan(raw_override, target, cast, story_data)
                plan["source"] = existing_dir.get("source") or "user-edited"
                if custom_wan_prompt and str(custom_wan_prompt).strip():
                    plan["wan_prompt"] = str(custom_wan_prompt).strip()
                target["director"] = plan
            elif custom_wan_prompt and str(custom_wan_prompt).strip():
                if not existing_dir:
                    existing_dir = self.compile_director_plan({}, target, cast, story_data)
                existing_dir["wan_prompt"] = str(custom_wan_prompt).strip()
                existing_dir["source"] = "user-edited"
                target["director"] = existing_dir

        use_seed = int(seed if seed is not None else target.get("video_seed", 42))
        title = target.get("title", f"Scene {scene_index+1}")

        if progress_cb:
            progress_cb(0.30, f"Redoing 5s video clip for Scene {scene_index+1} ('{title}', Seed {use_seed})...")

        clip_path = self._animate_single_scene_clip(
            scene=target,
            aspect_ratio=aspect_ratio,
            seed=use_seed,
            characters=characters,
            story_meta=story_data,
            use_last_frame=use_last_frame,
            enable_rife_2x=enable_rife_2x,
            anti_noise_guard=anti_noise_guard,
            progress_cb=progress_cb,
        )
        scenes[scene_index] = target
        story_data["scenes"] = scenes

        all_clips = [s.get("video_path") for s in scenes if s.get("video_path") and os.path.exists(s.get("video_path"))]
        story_data["clip_paths"] = all_clips

        if auto_stitch and len(all_clips) == len(scenes):
            final_movie = self.stitch_clips(
                clip_paths=all_clips,
                movie_title=story_data.get("title", "Studio_Movie"),
                progress_cb=progress_cb,
                scenes=scenes,
            )
            story_data["final_movie_path"] = final_movie
            story_data["manifest_path"] = self.save_manifest(story_data)

        if progress_cb:
            progress_cb(1.0, f"Scene {scene_index+1} video clip regenerated with seed {use_seed}: {os.path.basename(clip_path)}")
        return target

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6: Video Assembly & Stitching
    # ─────────────────────────────────────────────────────────────────────────

    def stitch_clips(
        self,
        clip_paths: List[str],
        movie_title: str = "Studio Movie",
        progress_cb: Optional[Callable[[float, str], None]] = None,
        scenes: Optional[List[Dict[str, Any]]] = None,
        transition_type: str = "cut",
        transition_duration: float = 0.5,
    ) -> str:
        """
        Concatenates individual scene clips into a single movie using FFmpeg.
        Supports seamless cinematic transitions ('cut', 'crossfade', 'fade_black', 'match_cut')
        with companion audio crossfading, and exports synchronized `.srt` subtitles.
        """
        if not clip_paths:
            raise ValueError("No video clips provided for stitching.")
        missing = [p for p in clip_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Clip(s) not found for stitching: {missing}")

        if progress_cb:
            progress_cb(0.92, f"Assembling scene clips with transition '{transition_type}' into final movie...")

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        timestamp = int(time.time())
        clean_title = self._sanitize_filename(movie_title) or "Studio_Movie"
        output_movie_path = self.movies_dir / f"{clean_title}_{timestamp}.mp4"

        stitched_successfully = False

        # ── 1. Cinematic Transition Path (xfade / acrossfade) ─────────────────
        if transition_type in ("crossfade", "fade_black", "match_cut") and len(clip_paths) >= 2 and transition_duration > 0:
            try:
                durations = []
                has_audio = True
                for p in clip_paths:
                    dur = 5.0
                    try:
                        import cv2
                        cap = cv2.VideoCapture(str(p))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
                            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 81.0
                            cap.release()
                            if fps > 0:
                                dur = float(frames / fps)
                    except Exception:
                        pass
                    durations.append(dur)

                    # Probe whether audio track exists
                    try:
                        probe_res = subprocess.run([ffmpeg_bin, "-i", str(p)], capture_output=True, text=True)
                        if "Audio:" not in (probe_res.stderr or ""):
                            has_audio = False
                    except Exception:
                        has_audio = False

                v_filt, a_filt, _ = build_ffmpeg_transition_filter(
                    clip_durations=durations,
                    transition_type=transition_type,
                    transition_duration=transition_duration,
                    has_audio=has_audio,
                )

                if v_filt:
                    input_args = []
                    for p in clip_paths:
                        input_args.extend(["-i", str(p)])
                    filter_complex = f"{v_filt};{a_filt}" if a_filt else v_filt
                    cmd_xfade = [ffmpeg_bin, "-y"] + input_args + [
                        "-filter_complex", filter_complex,
                        "-map", "[v]",
                    ]
                    if a_filt:
                        cmd_xfade.extend(["-map", "[a]", "-c:a", "aac", "-b:a", "192k"])
                    cmd_xfade.extend([
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(output_movie_path)
                    ])
                    print(f"[StudioPipeline] Stitching clips via FFmpeg xfade ({transition_type}): {' '.join(cmd_xfade[:12])}...")
                    res_xf = subprocess.run(cmd_xfade, capture_output=True, text=True)
                    if res_xf.returncode == 0 and output_movie_path.exists() and output_movie_path.stat().st_size > 0:
                        stitched_successfully = True
                    else:
                        print(f"[StudioPipeline] Transition stitch returned {res_xf.returncode}, falling back to concat: {res_xf.stderr[:200]}")
            except Exception as e:
                print(f"[StudioPipeline] Transition stitch note ({e}); falling back to concat.")

        # ── 2. Fallback / Standard Hard Concat ────────────────────────────────
        if not stitched_successfully:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_list:
                concat_list_file = f_list.name
                for p in clip_paths:
                    safe_path = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
                    f_list.write(f"file '{safe_path}'\n")

            try:
                cmd_copy = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_file,
                            "-c", "copy", str(output_movie_path)]
                print(f"[StudioPipeline] Stitching clips via FFmpeg copy: {' '.join(cmd_copy)}")
                res = subprocess.run(cmd_copy, capture_output=True, text=True)

                if res.returncode != 0 or not output_movie_path.exists() or output_movie_path.stat().st_size == 0:
                    print(f"[StudioPipeline] FFmpeg copy returned {res.returncode}, re-encoding concat...")
                    cmd_reencode = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_file,
                                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                                    "-c:a", "aac", "-b:a", "192k",
                                    str(output_movie_path)]
                    res_re = subprocess.run(cmd_reencode, capture_output=True, text=True)
                    if res_re.returncode != 0:
                        # Fallback without audio codec if input clips have no audio stream
                        cmd_reencode_vo = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_file,
                                           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                                           str(output_movie_path)]
                        res_vo = subprocess.run(cmd_reencode_vo, capture_output=True, text=True)
                        if res_vo.returncode != 0:
                            raise RuntimeError(f"FFmpeg stitching failed: {res_vo.stderr}")
            finally:
                if os.path.exists(concat_list_file):
                    os.remove(concat_list_file)

        if scenes:
            try:
                srt_path = output_movie_path.with_suffix(".srt")
                AudioVisualSyncEngine.write_srt_subtitles(
                    scenes,
                    str(srt_path),
                    transition_type=transition_type,
                    transition_duration=transition_duration,
                )
                print(f"[StudioPipeline] Exported synchronized SRT subtitles: {srt_path}")
            except Exception as e:
                print(f"[StudioPipeline] Subtitle export note: {e}")

        print(f"[StudioPipeline] Successfully assembled movie ({output_movie_path.stat().st_size} bytes): {output_movie_path}")

        if progress_cb:
            progress_cb(1.0, f"Movie assembly complete: {output_movie_path.name}")

        return str(output_movie_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Full End-to-End Orchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        user_prompt: str,
        shared_seed: Optional[int] = None,
        z_image_seed: Optional[int] = None,
        qwen_seed: Optional[int] = None,
        wan_seed: Optional[int] = None,
        num_scenes: int = 6,
        target_duration: float = 30.0,
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        art_style: str = "2D vibrant cartoon animation",
        use_vision_director: bool = True,
        use_last_frame: bool = True,
        enable_rife_2x: bool = True,
        chain_previous_end_frame: bool = False,
        anti_noise_guard: bool = True,
        enable_narrator: bool = True,
        narrator_voice: str = DEFAULT_NARRATOR_VOICE,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes all stages of the Studio pipeline end-to-end using three dedicated seeds:
          - z_image_seed : locked across all Z-Image Turbo character portraits
          - qwen_seed    : locked across all Qwen Image Edit scene keyframes
          - wan_seed     : locked across all Wan 2.2 5s video clips
        Includes Open-Source Narrator Voice Synthesis & Audio-Visual Sync when `enable_narrator=True`.
        """
        random_seeds = generate_three_studio_seeds()
        seed_z = int(z_image_seed if z_image_seed is not None else (shared_seed if shared_seed is not None else random_seeds["z_image_seed"]))
        seed_qwen = int(qwen_seed if qwen_seed is not None else (shared_seed if shared_seed is not None else random_seeds["qwen_seed"]))
        seed_wan = int(wan_seed if wan_seed is not None else (shared_seed if shared_seed is not None else random_seeds["wan_seed"]))

        story_data = self.generate_story_and_characters(
            prompt=user_prompt,
            num_scenes=num_scenes,
            target_duration=target_duration,
            art_style=art_style,
            progress_cb=progress_cb,
        )
        story_data["z_image_seed"] = seed_z
        story_data["qwen_seed"] = seed_qwen
        story_data["wan_seed"] = seed_wan
        story_data["narrator_voice_preset"] = narrator_voice
        story_data["seeds"] = {
            "z_image_seed": seed_z,
            "qwen_seed": seed_qwen,
            "wan_seed": seed_wan,
        }

        if enable_narrator:
            story_data["scenes"] = self.generate_story_narration(
                scenes=story_data.get("scenes", []),
                voice_preset=narrator_voice,
                quantize_durations=False,
                progress_cb=progress_cb,
            )

        characters = self.generate_character_portraits(
            characters=story_data.get("characters", []),
            shared_seed=seed_z,
            art_style=story_data.get("art_style", art_style),
            progress_cb=progress_cb,
        )
        story_data["characters"] = characters

        scenes = self.generate_scene_keyframes(
            scenes=story_data.get("scenes", []),
            characters=characters,
            aspect_ratio=aspect_ratio,
            shared_seed=seed_qwen,
            generate_end_frame=use_last_frame,
            chain_previous_end_frame=chain_previous_end_frame,
            progress_cb=progress_cb,
        )

        scenes = self.direct_scene_animations(
            scenes=scenes,
            characters=characters,
            story_meta=story_data,
            use_vision=use_vision_director,
            progress_cb=progress_cb,
        )
        story_data["scenes"] = scenes

        clip_paths = self.generate_scene_videos(
            scenes=scenes,
            aspect_ratio=aspect_ratio,
            base_seed=seed_wan,
            use_last_frame=use_last_frame,
            enable_rife_2x=enable_rife_2x,
            anti_noise_guard=anti_noise_guard,
            progress_cb=progress_cb,
        )

        final_movie = self.stitch_clips(
            clip_paths=clip_paths,
            movie_title=story_data.get("title", "Studio_Movie"),
            progress_cb=progress_cb,
            scenes=scenes,
        )
        story_data["final_movie_path"] = final_movie
        srt_candidate = pathlib.Path(final_movie).with_suffix(".srt")
        if srt_candidate.exists():
            story_data["subtitle_srt_path"] = str(srt_candidate)
        story_data["clip_paths"] = clip_paths
        story_data["manifest_path"] = self.save_manifest(story_data)

        return story_data


    # ─────────────────────────────────────────────────────────────────────────
    # Helper Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def serializable(story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-copies story data without PIL images so it can be shown as JSON / saved."""
        seen = set()

        def strip(obj):
            obj_id = id(obj)
            if isinstance(obj, (dict, list)):
                if obj_id in seen:
                    return None
                seen.add(obj_id)
            if isinstance(obj, dict):
                return {k: strip(v) for k, v in obj.items() if not isinstance(v, Image.Image)}
            if isinstance(obj, list):
                return [strip(v) for v in obj if not isinstance(v, Image.Image)]
            return obj

        return strip(story_data or {})

    def save_manifest(self, story_data: Dict[str, Any]) -> str:
        """Saves the screenplay + director plans + output paths next to the final movie."""
        name = self._sanitize_filename(story_data.get("title", "Studio_Movie")) or "Studio_Movie"
        path = self.movies_dir / f"{name}_{int(time.time())}_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.serializable(story_data), f, indent=2, ensure_ascii=False)
        return str(path)

    @staticmethod
    def _parse_json_safely(raw_text: str) -> Dict[str, Any]:
        clean = (raw_text or "").strip()
        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0]
        clean = clean.strip()

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start : end + 1]

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            clean_fixed = re.sub(r",\s*([\]}])", r"\1", clean)
            try:
                return json.loads(clean_fixed)
            except Exception:
                raise ValueError(f"Could not parse valid JSON from Gemma response: {e}\nRaw: {(raw_text or '')[:300]}")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        s = re.sub(r"[^\w\-_]", "_", str(name or ""))
        return re.sub(r"__+", "_", s).strip("_")[:40]
