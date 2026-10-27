#!/usr/bin/env python3
"""
Full 30-Second Heart-Touching Short Story Production Runner & Telemetry Analyzer.
Executes all 6 pipeline stages across 6 consecutive 5.0-second shots (30.0s total)
with <= 3 characters, Open-Source Narrator Voiceover, 32fps RIFE 2x, FLF2V + Anti-Noise Guard,
FFmpeg 0.5s Crossfade Transitions, and Deep Telemetry & Continuous Improvement Logging.
"""

import os
import sys
import time
import json
import pathlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from studio_pipeline import StudioPipeline
from scripts.export_story_director_dataset import export_datasets


def on_progress(pct: float, msg: str) -> None:
    elapsed = time.strftime("%H:%M:%S")
    print(f"[{elapsed}] [{pct * 100:5.1f}%] {msg}", flush=True)


def run_30s_production() -> dict:
    user_prompt = (
        "A deeply heart-touching short story with 2 characters: Silas, a gentle elderly clockmaker "
        "with wire-rimmed spectacles and a brown woolen vest, and Pip, a tiny freezing European robin "
        "with a russet-orange breast. On a snowy winter morning, Silas discovers shivering Pip on his "
        "frosted workshop windowsill, cradles him in a warm woolen handkerchief by the glowing hearth, "
        "feeds him sweet breadcrumbs on his brass clock workbench until Pip perches happily atop his "
        "magnifying loupe, and when spring cherry blossoms bloom outside, Silas opens the window and "
        "watches Pip soar joyfully into the golden sky."
    )

    art_style = (
        "Warm whimsical storybook 2D animation, Studio Ghibli watercolor aesthetic, "
        "expressive character acting, soft hearth lighting, rich atmospheric depth, 4K"
    )

    print("=" * 84, flush=True)
    print(" WAN 2.2 ORCHESTRA AI STUDIO: 30-SECOND SHORT FILM PRODUCTION & TELEMETRY ".center(84, "="), flush=True)
    print("=" * 84, flush=True)
    print(f"Premise          : {user_prompt[:100]}...", flush=True)
    print(f"Target Duration  : 30.0s (6 scenes x 5.0s each)", flush=True)
    print(f"Max Characters   : <= 3 (Heart-Touching Intimacy Rule)", flush=True)
    print(f"Transition Mode  : Cross-Dissolve (xfade=fade, 0.5s overlap + acrossfade)", flush=True)
    print("=" * 84, flush=True)

    t0 = time.time()
    pipeline = StudioPipeline()

    story_data = pipeline.run_full_pipeline(
        user_prompt=user_prompt,
        z_image_seed=808101,
        qwen_seed=808202,
        wan_seed=808303,
        num_scenes=6,
        target_duration=30.0,
        aspect_ratio="9:16 Vertical (Shorts/Reels)",
        art_style=art_style,
        use_vision_director=True,
        use_last_frame=True,
        enable_rife_2x=True,
        chain_previous_end_frame=False,
        anti_noise_guard=True,
        enable_narrator=True,
        narrator_voice="warm_storyteller",
        transition_type="crossfade",
        transition_duration=0.5,
        run_story_audit=True,
        auto_qa_retry=True,
        manage_gpu_phases=True,
        collect_telemetry=True,
        progress_cb=on_progress,
    )

    elapsed = round(time.time() - t0, 2)

    # Save exemplar to Episodic Memory Bank & refresh training datasets
    try:
        pipeline.record_story_to_memory(
            story_data=story_data,
            premise=user_prompt,
            rating=9.8,
            user_edited=False,
            notes="30s heart-touching production run with full telemetry and crossfade transitions",
        )
        export_datasets("dataset")
    except Exception as e:
        print(f"[Runner] Memory/dataset export note: {e}", flush=True)

    telemetry = story_data.get("production_telemetry") or {}
    summary = telemetry.get("summary") or {}

    print("\n" + "=" * 84, flush=True)
    print(f" 30-SECOND SHORT STORY COMPLETE IN {elapsed:.1f}s! ".center(84, "="), flush=True)
    print("=" * 84, flush=True)
    print(f"Title                : {story_data.get('title')}", flush=True)
    print(f"Cast Count           : {len(story_data.get('characters', []))} (Max Allowed: 3)", flush=True)
    for c in story_data.get("characters", []):
        print(f"  - [{c.get('id')}] {c.get('name')}: {c.get('portrait_path')}", flush=True)

    print(f"\nScenes Generated     : {len(story_data.get('scenes', []))}", flush=True)
    for s in story_data.get("scenes", []):
        dp = s.get("director") or {}
        tel = s.get("telemetry") or {}
        print(
            f"  - Scene #{s.get('scene_number')} [{s.get('title')}]: "
            f"delta={s.get('pose_delta')} | "
            f"flow={tel.get('mean_flow')} | "
            f"surge={tel.get('surge_ratio')} | "
            f"sharp={tel.get('sharpness_retention')} | "
            f"sync_err={tel.get('audio_sync_delta_sec')}s | "
            f"clip={pathlib.Path(s.get('video_path', '')).name}",
            flush=True,
        )

    print("\n" + "-" * 84, flush=True)
    print(f"Final Stitched Movie : {story_data.get('final_movie_path')}", flush=True)
    print(f"Synchronized SRT     : {story_data.get('subtitle_srt_path')}", flush=True)
    print(f"Manifest JSON        : {story_data.get('manifest_path')}", flush=True)
    print(f"Telemetry JSON       : {story_data.get('telemetry_report_path')}", flush=True)
    print(f"Analysis Markdown    : {story_data.get('analysis_report_path')}", flush=True)
    print(f"Summary Metrics      : {json.dumps(summary)}", flush=True)
    print("=" * 84, flush=True)

    return story_data


if __name__ == "__main__":
    run_30s_production()
