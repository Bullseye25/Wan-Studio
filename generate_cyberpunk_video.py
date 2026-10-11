import sys
import app
from PIL import Image

print("[Cyberpunk Video] Loading input image: cyberpunk_hitman.png...")
img = Image.open("cyberpunk_hitman.png")

prompt = (
    "female cyberpunk hitman smoothly loading glowing neon magazine into handgun, chambering futuristic pistol, "
    "rain dripping on leather, cyan and magenta reflections, "
    "[Subject Invariant Anchors]: strictly preserve source subject's facial identity, exact bone structure, "
    "eye shape, nose geometry, jawline, and skin texture without alteration or morphing throughout. "
    "[Timeline Stages]: Phase 1 (0.0s-1.2s): natural rhythmic breathing, steady eye contact, locked head pose; "
    "Phase 2 (1.2s-2.8s): hands firmly push glowing magazine into grip with tactile metallic click, subtle 5-degree head tilt, "
    "rain drops dripping off chin; Phase 3 (2.8s-3.5s): settling into alert firing stance, intense steady gaze, "
    "photorealistic continuous lighting. [Camera Trajectory]: Slow Cinematic Push-In with subtle orbital parallax, "
    "dramatic depth of field, fixed focal length, 4k ultra-detailed, photorealistic cinematic lighting."
)

negative_prompt = (
    "face morphing, changing face, altered facial features, changing identity, inconsistent face, "
    "shifting features, melting face, face distortion, facial transformation, mutating face, "
    "asymmetrical eyes, cross-eyed, wandering gaze, sudden smile distortion, "
    "temporal flickering, pixel noise, pixel degradation, digital artifacts, warping, tearing, "
    "blurry, low quality, worst quality, bad anatomy, deformed"
)

print("[Cyberpunk Video] Initiating Wan 2.2 generation on Modal...")
video_comp, video_path, seed = app.generate_video(
    input_image=img,
    last_image=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    duration_seconds=3.5,
    guidance_scale=2.25,
    guidance_scale_2=1.5,
    steps=10,
    seed=42,
    randomize_seed=False,
    quality=6,
    scheduler="UniPCMultistep",
    flow_shift=3.0,
    frame_multiplier=16,
    deflicker_frames=True,
)
print(f"[Cyberpunk Video] Successfully generated video: {video_path}")
