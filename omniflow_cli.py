"""
OmniFlow-30 CLI: Interactive Story-to-Animation Production Interface
====================================================================
Command-line tool to produce cinema-grade 30-second 9:16 vertical animations
using Wan 2.2, Qwen Image Edit, Z-Image Turbo, and Modal Cloud Parallel Swarms.

Usage:
    python omniflow_cli.py "A lonely robot tends a glowing botanical garden on a derelict starship"
    python omniflow_cli.py --duration 30.0 --style "Cyberpunk neon watercolor"
    python omniflow_cli.py (interactive prompt)
"""

import os
import sys
import time
import argparse
import pathlib

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from omniflow_engine import (
    OmniFlowEngine,
    OmniFlowConfig,
    calculate_omniflow_scenes,
)
from fast_story_cli import TerminalProgressTracker, format_time
from self_learning_critic import DirectorKnowledgeBase, SelfLearningCritic


BANNER = r"""
========================================================================================
   ____                  _ _____ _                  _____  ___  
  / __ \                (_)  ___| |                |____ |/ _ \ 
 | |  | |_ __ ___  _ __  _| |_  | | _____      __      / / | | |
 | |  | | '_ ` _ \| '_ \| |  _| | |/ _ \ \ /\ / /      \ \ | | |
 | |__| | | | | | | | | | | |   | | (_) \ V  V /   .___/ / |_| |
  \____/|_| |_| |_|_| |_|_|_|   |_|\___/ \_/\_/    \____/ \___/ 
                                                                
     🎬 NEXT-GEN STORY-TO-ANIMATION STUDIO · 30-SEC 9:16 VERTICAL CINEMA
========================================================================================
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniFlow-30: Autonomous 30-Second 9:16 AI Animation Production CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "premise",
        nargs="?",
        default=None,
        help="Story premise or idea (if omitted, you will be prompted interactively)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Target animation runtime in seconds (default: 30.0s for Reels/Shorts)",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="Number of scenes (default: 4 scenes for 30s runtime)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="9:16 Vertical (Shorts/Reels)",
        choices=[
            "9:16 Vertical (Shorts/Reels)",
            "16:9 Cinematic Widescreen",
            "1:1 Square (Feed)",
        ],
        help="Aspect ratio framing",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="2D vibrant cartoon animation, Studio Ghibli watercolor aesthetic, lush hand-painted textures, soft edge lighting, 4K",
        help="Visual art style specification",
    )
    parser.add_argument(
        "--no-swarm",
        action="store_true",
        help="Disable parallel cloud swarm and render scenes sequentially",
    )
    parser.add_argument(
        "--no-midpoint-refine",
        action="store_true",
        help="Disable in-flight MILIS midpoint latent refinement during Wan denoising",
    )
    parser.add_argument(
        "--no-narrator",
        action="store_true",
        help="Disable narrator voiceover synthesis",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Warm Storyteller (Female - af_heart)",
        help="Narrator voice preset: 'af_heart' (Warm Female), 'am_adam' (Charming Male), 'af_bella' (Anime Female), 'am_fenrir' (Trailer Male), 'bf_emma' (British Fable)",
    )
    parser.add_argument(
        "--transition",
        type=str,
        default="narrative",
        choices=["narrative", "crossfade", "fade_black", "dissolve", "cut"],
        help="Cinematic scene transition style",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default="",
        help="Optional Wan 2.2 LoRA weights repository or local path",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.8,
        help="LoRA blending weight scale",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="groq",
        choices=["groq", "ollama_cloud", "local_ollama"],
        help="LLM provider for Screenplay and Vision Director (default: groq with auto-fallback to ollama_cloud)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for Screenplay and Vision Director (defaults to provider's best model)",
    )
    parser.add_argument(
        "--story-folder",
        "--story-name",
        type=str,
        default=None,
        help="Dedicated subfolder inside outputs/ for this story (e.g. 'fox_story')",
    )
    return parser.parse_args()


def run_cli() -> None:
    print(BANNER)
    args = parse_arguments()

    premise = args.premise
    if not premise or not premise.strip():
        print("💡 What story would you like to animate today? (30s 9:16 Vertical)")
        print("   Example: 'A small steampunk owl learns to fly under a neon clocktower'")
        try:
            premise = input("\n👉 Enter your story premise: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting OmniFlow-30.")
            sys.exit(0)

    if not premise:
        print("❌ Error: A story premise is required.")
        sys.exit(1)

    config = OmniFlowConfig(
        prompt=premise,
        target_duration=args.duration,
        aspect_ratio=args.aspect_ratio,
        art_style=args.style,
        num_scenes=args.scenes,
        parallel_swarm=not args.no_swarm,
        enable_midpoint_refine=not args.no_midpoint_refine,
        enable_narrator=not args.no_narrator,
        narrator_voice=args.voice,
        transition_type=args.transition,
        transition_duration=0.5,
        lora_repo_or_path=args.lora,
        lora_scale=args.lora_scale,
        provider=args.provider,
        model=args.model,
        story_folder_name=args.story_folder,
    )

    engine = OmniFlowEngine()
    total_eta, breakdown = engine.calculate_eta(config)
    past_rules = engine.get_past_learned_directives(premise, limit=3)

    num_scenes = breakdown["num_scenes"]
    per_scene_dur = config.target_duration / max(1, num_scenes)

    print("\n" + "=" * 88)
    print(" 🎬 OMNIFLOW-30 PRODUCTION MANIFEST ".center(88, "="))
    print("=" * 88)
    print(f"📖 Story Premise     : {premise}")
    print(f"⏱️ Runtime / Pacing  : {config.target_duration:.1f}s ({num_scenes} scenes @ ~{per_scene_dur:.1f}s each)")
    print(f"📱 Aspect Ratio      : {config.aspect_ratio}")
    print(f"🎨 Visual Style      : {config.art_style[:70]}...")
    print(f"⚡ Cloud Execution   : {'Parallel Modal Swarm (Concurrent A100s)' if config.parallel_swarm else 'Sequential'}")
    print(f"🔬 Latent Refinement : {'In-Flight MILIS Active (t=0.5)' if config.enable_midpoint_refine else 'Standard'}")
    print(f"🎙️ Narration Engine  : {config.narrator_voice if config.enable_narrator else 'Disabled'}")
    print(f"🎬 Transitions       : {config.transition_type.capitalize()} (Duration: {config.transition_duration:.1f}s)")

    if past_rules:
        print(f"\n🧠 Evolutionary Memory: Loaded {len(past_rules)} learned directive(s) from past films:")
        for idx, r in enumerate(past_rules, 1):
            print(f"   {idx}. [{r.get('category')}]: {r.get('directive_to_enforce')[:68]}...")

    print(f"\n⏳ Estimated Production Time: ~{format_time(total_eta)} ({total_eta:.0f} seconds)")
    print("=" * 88 + "\n")

    tracker = TerminalProgressTracker(total_eta_seconds=total_eta)

    def progress_callback(pct: float, msg: str) -> None:
        tracker.update(pct, msg)

    t_start = time.time()
    try:
        story_data = engine.produce_film(config, progress_callback=progress_callback)
    except Exception as e:
        print(f"\n\n❌ Production Encountered Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed_actual = round(time.time() - t_start, 1)
    tracker.complete(f"30s Film Production Finished in {format_time(elapsed_actual)}!")

    critic_report = story_data.get("critic_evaluation") or {}
    print("\n" + "=" * 88)
    print(f" 🎬 OMNIFLOW-30 FINAL MOVIE DELIVERABLES ".center(88, "="))
    print("=" * 88)
    print(f"🎞️ Title            : {story_data.get('title', 'OmniFlow Masterpiece')}")
    print(f"⏱️ Total Wall Clock : {format_time(elapsed_actual)} (Estimated: ~{format_time(total_eta)})")
    print(f"📁 Story Directory  : {story_data.get('story_dir')}")
    print(f"🎬 Video File (MP4) : {story_data.get('final_movie_path')}")
    print(f"📝 Subtitles (SRT)  : {story_data.get('subtitle_srt_path')}")
    print(f"📊 Telemetry Report : {story_data.get('telemetry_report_path')}")
    print(f"🏆 Studio Score     : {critic_report.get('overall_studio_score', 8.5):.1f} / 10.0")

    print("\n" + "-" * 88)
    print(" 🧠 MULTI-TIER AI CRITIC EVALUATION & CONTINUOUS LEARNING ".center(88, "-"))
    print("-" * 88)

    # Character evaluations
    print("\n[Tier 1: Character Genesis (Z-Image Turbo)]")
    for ce in critic_report.get("character_evaluations", []):
        badge = "✅" if ce["score"] >= 8.5 else ("⚠️" if ce["score"] >= 7.0 else "❌")
        print(f"  {badge} {ce['target']} — Score: {ce['score']}/10.0 (Sharpness: {ce.get('sharpness', 0):.1f})")

    # Keyframe evaluations
    print("\n[Tier 2: Dual Keyframe Motion Dynamics (Qwen Image Edit)]")
    for ke in critic_report.get("keyframe_evaluations", []):
        badge = "✅" if ke["score"] >= 7.5 else ("⚠️" if ke["score"] >= 6.0 else "❌")
        print(f"  {badge} {ke['target']} — Score: {ke['score']}/10.0 (Pose Delta: {ke.get('pose_delta', 0):.1f})")

    # Animation evaluation
    print("\n[Tier 3: Spatio-Temporal Animation & Flow (Wan 2.2 FLF2V)]")
    for me in critic_report.get("movie_evaluations", []):
        badge = "✅" if me["score"] >= 7.5 else ("⚠️" if me["score"] >= 6.0 else "❌")
        print(f"  {badge} {me['target']} — Score: {me['score']}/10.0")
        for find in me.get("findings", []):
            print(f"     • {find}")

    new_rules = critic_report.get("new_rules_learned", [])
    if new_rules:
        print(f"\n💡 Continuous Improvement: Stored {len(new_rules)} new directive(s) in 'director_learned_rules.jsonl'.")
        print("   The AI Director will automatically enforce these rules in future films to avoid repeating flaws!")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    run_cli()
