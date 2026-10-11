import os
import math
import random
import subprocess
import numpy as np
from PIL import Image
import cv2
import imageio_ffmpeg

# 1. Output setup
OUTPUT_DIR = r"H:\A\Imp\Wan\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "sharing_is_caring_30s_movie.mp4")

# 2. Scene images from artifacts
BRAIN_DIR = r"C:\Users\Ammad Raza\.gemini\antigravity\brain\8ab6e0be-cea9-4f3e-b652-8008347f879d"
SCENE_FILES = [
    os.path.join(BRAIN_DIR, "scene1_friends_meadow_1791717485162.jpg"),
    os.path.join(BRAIN_DIR, "scene2_bunny_treat_1791717515216.jpg"),
    os.path.join(BRAIN_DIR, "scene3_bear_sad_1791717554891.jpg"),
    os.path.join(BRAIN_DIR, "scene4_bunny_splitting_1791717607807.jpg"),
    os.path.join(BRAIN_DIR, "scene5_sharing_moment_1791717657108.jpg"),
    os.path.join(BRAIN_DIR, "scene6_sharing_is_caring_1791717715010.jpg"),
]

for p in SCENE_FILES:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing scene image: {p}")

print("[Movie Renderer] All 6 scene images verified!", flush=True)

# Video Specs: 9:16 vertical resolution, 30 FPS, 30.0s total duration
TARGET_W = 1080
TARGET_H = 1920
FPS = 30
TOTAL_DURATION = 30.0
TOTAL_FRAMES = int(FPS * TOTAL_DURATION)  # 900 frames

NUM_SCENES = len(SCENE_FILES)
CROSSFADE_SEC = 0.8
CROSSFADE_FRAMES = int(FPS * CROSSFADE_SEC)

SCENE_DUR_SEC = (TOTAL_DURATION + (NUM_SCENES - 1) * CROSSFADE_SEC) / NUM_SCENES
SCENE_FRAMES = int(round(SCENE_DUR_SEC * FPS))

print(f"[Movie Renderer] Target: {TARGET_W}x{TARGET_H} @ {FPS} FPS | Total: {TOTAL_FRAMES} frames ({TOTAL_DURATION}s)", flush=True)

# Load and prepare high-res source images
raw_images = []
for p in SCENE_FILES:
    img = Image.open(p).convert("RGB")
    overscan_w = int(TARGET_W * 1.15)
    overscan_h = int(TARGET_H * 1.15)
    img_resized = img.resize((overscan_w, overscan_h), Image.Resampling.LANCZOS)
    raw_images.append(np.array(img_resized))

CAMERA_TRAJECTORIES = [
    {"z0": 1.00, "z1": 1.08, "dx0": 0.0, "dx1": 0.0, "dy0": 0.02, "dy1": -0.02},
    {"z0": 1.00, "z1": 1.09, "dx0": 0.01, "dx1": -0.01, "dy0": 0.03, "dy1": 0.00},
    {"z0": 1.00, "z1": 1.08, "dx0": 0.0, "dx1": 0.0, "dy0": -0.02, "dy1": 0.03},
    {"z0": 1.08, "z1": 1.01, "dx0": 0.0, "dx1": 0.0, "dy0": 0.01, "dy1": -0.01},
    {"z0": 1.00, "z1": 1.07, "dx0": -0.02, "dx1": 0.01, "dy0": 0.01, "dy1": -0.01},
    {"z0": 1.00, "z1": 1.06, "dx0": 0.0, "dx1": 0.0, "dy0": 0.03, "dy1": -0.02},
]

def render_scene_frame(scene_idx, progress_ratio):
    img_np = raw_images[scene_idx]
    H, W, _ = img_np.shape
    traj = CAMERA_TRAJECTORIES[scene_idx]

    t = progress_ratio
    ease = t * t * (3.0 - 2.0 * t)

    zoom = traj["z0"] + (traj["z1"] - traj["z0"]) * ease
    dx = traj["dx0"] + (traj["dx1"] - traj["dx0"]) * ease
    dy = traj["dy0"] + (traj["dy1"] - traj["dy0"]) * ease

    crop_w = int(TARGET_W / zoom)
    crop_h = int(TARGET_H / zoom)

    center_x = int(W / 2 + dx * W)
    center_y = int(H / 2 + dy * H)

    x0 = max(0, min(W - crop_w, center_x - crop_w // 2))
    y0 = max(0, min(H - crop_h, center_y - crop_h // 2))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    cropped = img_np[y0:y1, x0:x1]
    return cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

# Particle system
random.seed(42)
NUM_PARTICLES = 40
particles = []
for _ in range(NUM_PARTICLES):
    particles.append({
        "x": random.uniform(0, TARGET_W),
        "y": random.uniform(0, TARGET_H),
        "speed_y": random.uniform(18, 42),
        "speed_x": random.uniform(-10, 15),
        "radius": random.uniform(2.5, 5.0),
        "alpha": random.uniform(0.35, 0.75),
        "color": random.choice([
            (255, 240, 200),
            (255, 200, 220),
            (220, 245, 255),
        ]),
        "phase": random.uniform(0, math.pi * 2)
    })

def apply_particles(frame_bgr, global_time):
    overlay = frame_bgr.copy()
    dt = 1.0 / FPS

    for p in particles:
        p["y"] = (p["y"] + p["speed_y"] * dt) % TARGET_H
        p["x"] = (p["x"] + p["speed_x"] * dt + math.sin(global_time * 2.0 + p["phase"]) * 0.8) % TARGET_W

        twinkle = 0.5 + 0.5 * math.sin(global_time * 4.0 + p["phase"])
        center = (int(p["x"]), int(p["y"]))
        r = int(p["radius"])
        cv2.circle(overlay, center, r, p["color"], -1, cv2.LINE_AA)
        cv2.circle(overlay, center, r + 2, p["color"], 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.22, frame_bgr, 0.78, 0, frame_bgr)
    return frame_bgr

scene_start_frames = [s * (SCENE_FRAMES - CROSSFADE_FRAMES) for s in range(NUM_SCENES)]

# Initialize FFmpeg process directly streaming to disk (zero memory bottleneck)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"[Movie Renderer] Initializing FFmpeg stream pipeline ({ffmpeg_exe})...", flush=True)

cmd = [
    ffmpeg_exe,
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{TARGET_W}x{TARGET_H}",
    "-pix_fmt", "bgr24",
    "-r", str(FPS),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    OUTPUT_VIDEO_PATH
]

process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("[Movie Renderer] Streaming 900 frames directly to FFmpeg...", flush=True)

for f in range(TOTAL_FRAMES):
    global_time = f / FPS
    frame_rgb = None

    for s in range(NUM_SCENES):
        start_f = scene_start_frames[s]
        end_f = start_f + SCENE_FRAMES

        if start_f <= f < end_f:
            prog = (f - start_f) / float(SCENE_FRAMES)
            curr_f = render_scene_frame(s, prog)

            if s < NUM_SCENES - 1:
                next_start = scene_start_frames[s + 1]
                if f >= next_start:
                    fade_prog = (f - next_start) / float(CROSSFADE_FRAMES)
                    fade_factor = fade_prog * fade_prog * (3.0 - 2.0 * fade_prog)
                    next_prog = (f - next_start) / float(SCENE_FRAMES)
                    next_f = render_scene_frame(s + 1, next_prog)
                    frame_rgb = cv2.addWeighted(curr_f, 1.0 - fade_factor, next_f, fade_factor, 0)
                    break

            frame_rgb = curr_f
            break

    if frame_rgb is None:
        frame_rgb = render_scene_frame(NUM_SCENES - 1, 1.0)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame_bgr = apply_particles(frame_bgr, global_time)

    # Stream BGR bytes directly to FFmpeg
    process.stdin.write(frame_bgr.tobytes())

    if (f + 1) % 150 == 0 or (f + 1) == TOTAL_FRAMES:
        print(f"  [Progress] Streamed {f + 1}/{TOTAL_FRAMES} frames ({(f+1)/TOTAL_FRAMES*100:.0f}%)...", flush=True)

process.stdin.close()
_, stderr = process.communicate()

if process.returncode != 0:
    print("FFmpeg error:", stderr.decode("utf-8", errors="ignore"), flush=True)
    raise RuntimeError("FFmpeg encoding failed.")

file_size_mb = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)
print(f"\n[Movie Renderer] SUCCESS! 30-Second Movie generated at:\n  --> {OUTPUT_VIDEO_PATH} ({file_size_mb:.2f} MB)", flush=True)
