"""
End-to-end Demo Story Runner for the Wan 2.2 AI Studio Pipeline.
Runs all 6 stages against the live Ollama Cloud (gemma4:31b) and Modal GPU backends:
  1. Gemma 4: 31B Story & Character Extraction
  2. Z-Image Turbo Seed-Locked Character Reference Portraits
  3. Qwen Image Edit Multi-Reference Scene Keyframes
  4. Gemma 4: 31B Multimodal Vision Animation Director
  5. Wan 2.2 Image-to-Video Animation
  6. FFmpeg Movie Stitching & Manifest Export
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from studio_pipeline import StudioPipeline


def on_progress(pct: float, msg: str):
    print(f"[{pct * 100:5.1f}%] {msg}", flush=True)


def main():
    user_prompt = (
        "A heartwarming 2D animated story about Luna, a curious little red fox in a star-patterned blue scarf, "
        "and Oliver, a gentle baby owl with round golden glasses, who discover a glowing fallen star in the "
        "whispering forest and work together to lift it back into the night sky."
    )

    print("=" * 72, flush=True)
    print(" Starting Live Wan 2.2 AI Studio Demo Story Generation ".center(72, "="), flush=True)
    print("=" * 72, flush=True)

    t0 = time.time()
    pipeline = StudioPipeline()
    story_data = pipeline.run_full_pipeline(
        user_prompt=user_prompt,
        shared_seed=77777,
        num_scenes=3,
        target_duration=15.0,
        aspect_ratio="9:16 Vertical (Shorts/Reels)",
        art_style="Vibrant 2D storybook cartoon illustration, warm magical lighting, clean expressive lines, 4K",
        use_vision_director=True,
        progress_cb=on_progress,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f" Demo Story Generation Complete in {elapsed:.1f}s! ".center(72, "="), flush=True)
    print("=" * 72, flush=True)
    print(f"Title       : {story_data.get('title')}", flush=True)
    print(f"Logline     : {story_data.get('logline')}", flush=True)
    print(f"Characters  : {len(story_data.get('characters', []))}", flush=True)
    for c in story_data.get("characters", []):
        print(f"  - {c.get('name')}: {c.get('portrait_path')}", flush=True)
    print(f"Scenes      : {len(story_data.get('scenes', []))}", flush=True)
    for s in story_data.get("scenes", []):
        dp = s.get("director", {})
        print(
            f"  - Scene {s.get('scene_number')} [{s.get('title')}]: "
            f"keyframe={s.get('keyframe_path')} | "
            f"director={dp.get('source')} ({dp.get('camera_movement')}) | "
            f"clip={s.get('video_path')}",
            flush=True,
        )
    print(f"Final Movie : {story_data.get('final_movie_path')}", flush=True)
    print(f"Manifest    : {story_data.get('manifest_path')}", flush=True)


if __name__ == "__main__":
    main()
