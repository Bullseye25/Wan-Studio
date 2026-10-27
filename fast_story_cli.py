#!/usr/bin/env python3
"""
Wan 2.2 Fast Story Animation CLI & Self-Learning Production Studio.
==================================================================
Interactive and one-command CLI that transforms any story premise into a
complete 30-second animated short film with:
  - Instant Dynamic-ETA Calculation on Enter
  - High-Speed Pipelined Cloud Rendering on Modal.com (Zero local GPU load)
  - Synchronized Open-Source Narrator Voiceover & Subtitles (.srt)
  - Multi-Tier AI Critic (Characters, Keyframes, Animation)
  - Continuous Memory Learning Loop (Stores mistakes & evolves future films)

Usage:
  # 1. Interactive Mode (prompts for story premise):
  python fast_story_cli.py

  # 2. Direct Premise Mode:
  python fast_story_cli.py "A tiny robot discovers a glowing sunflower in an abandoned greenhouse"

  # 3. Fast Mode with GroqCloud & Wan 2.2 LoRA:
  python fast_story_cli.py "Cyberpunk cat rescues a holographic sparrow" --fast --provider groq
"""

import os
import re
import sys
import time
import json
import pathlib
import argparse
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from studio_pipeline import (
    StudioPipeline,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    plan_scene_count,
)
from self_learning_critic import (
    SelfLearningCritic,
    DirectorKnowledgeBase,
)


# ── Dynamic ETA & Duration Estimator ──────────────────────────────────────────

def calculate_dynamic_eta(
    premise: str,
    target_duration: float = 30.0,
    provider: str = "groq",
    fast_mode: bool = True,
    is_modal_warm: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculates the expected time in seconds to produce the full animated short film.
    Returns: (total_eta_seconds, stage_breakdown_dict)
    """
    scenes_count = plan_scene_count(target_duration)

    # Detect estimated cast size from premise
    text = (premise or "").lower()
    words = set(re.findall(r"\w+", text))
    char_indicators = {
        "and", "with", "between", "both", "meets", "rescues", "befriends",
        "versus", "vs", "protects", "accompanies", "guides", "companion"
    }
    detected_chars = 2 if bool(words.intersection(char_indicators)) else 1

    # Stage duration benchmarks (derived from empirical cloud telemetry)
    # 1. Screenplay generation
    p_clean = str(provider or "groq").lower().strip()
    if p_clean == "groq":
        t_screenplay = 1.6
    elif p_clean == "local_ollama":
        t_screenplay = 6.5
    else:
        t_screenplay = 14.0

    # 2. Voiceover narration (local SAPI5)
    t_voice = round(scenes_count * 0.1, 1)

    # 3. Character portraits (Z-Image Turbo on Modal)
    t_chars = round(detected_chars * (3.0 if fast_mode else 4.0), 1)

    # 4. Keyframes (Qwen Image Edit on Modal: Start + End frame per scene)
    t_keyframes = round(scenes_count * 2 * (4.8 if fast_mode else 6.2), 1)

    # 5. Video rendering (Wan 2.2 FLF2V on Modal A100 GPU)
    # ~30s per 81-frame clip in 8-step Lightning mode
    wan_per_clip = 30.0 if fast_mode else 36.0
    t_video = round(scenes_count * wan_per_clip, 1)

    # 6. Overlapped pipelining savings
    t_overlap_savings = round(scenes_count * 6.5, 1) if fast_mode else 0.0

    # 7. Final FFmpeg transition assembly + Subtitle sync
    t_assembly = 1.5

    # 8. Multi-tier AI Critic & Self-learning post-mortem
    t_critic = 1.0

    # Cold boot allowance if container is idle
    t_cold_boot = 0.0 if is_modal_warm else 30.0

    total_eta = max(
        30.0,
        t_screenplay
        + t_voice
        + t_chars
        + t_keyframes
        + t_video
        + t_assembly
        + t_critic
        + t_cold_boot
        - t_overlap_savings,
    )

    breakdown = {
        "scenes_count": scenes_count,
        "detected_chars": detected_chars,
        "screenplay_sec": t_screenplay,
        "voice_sec": t_voice,
        "chars_sec": t_chars,
        "keyframes_sec": t_keyframes,
        "video_sec": t_video,
        "overlap_savings_sec": t_overlap_savings,
        "assembly_sec": t_assembly,
        "critic_sec": t_critic,
        "total_eta_sec": round(total_eta, 1),
    }

    return round(total_eta, 1), breakdown


def format_time(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
    s = int(round(seconds))
    mins, secs = divmod(s, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


# ── Interactive Progress Renderer ─────────────────────────────────────────────

class TerminalProgressTracker:
    """Renders a dynamic, modern terminal progress bar with dynamic remaining countdown."""

    def __init__(self, total_eta_seconds: float):
        self.t0 = time.time()
        self.total_eta = max(1.0, total_eta_seconds)
        self.last_pct = 0.0
        self.current_stage = "Initializing"

    def update(self, pct: float, message: str) -> None:
        self.last_pct = max(self.last_pct, min(1.0, float(pct)))
        elapsed = time.time() - self.t0
        remaining = max(0.0, (elapsed / max(0.01, self.last_pct)) - elapsed) if self.last_pct > 0.05 else max(0.0, self.total_eta - elapsed)

        bar_len = 30
        filled = int(round(bar_len * self.last_pct))
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed_str = format_time(elapsed)
        rem_str = format_time(remaining)
        pct_display = int(round(self.last_pct * 100))

        # Format clean line
        line = f"\r⏳ [{elapsed_str} / ~{rem_str} left] [{bar}] {pct_display:3d}% ➔ {message[:45]:<45}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def complete(self, message: str = "Production complete!") -> None:
        elapsed = time.time() - self.t0
        sys.stdout.write(f"\r✅ [{format_time(elapsed)}] [{'█'*30}] 100% ➔ {message:<45}\n")
        sys.stdout.flush()


# ── Production Runner ─────────────────────────────────────────────────────────

def run_fast_story_production(
    premise: str,
    target_duration: float = 30.0,
    art_style: str = "2D vibrant cartoon animation, Studio Ghibli watercolor aesthetic, warm lighting, 4K",
    provider: str = "groq",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    lora_path: str = "",
    lora_scale: float = 0.8,
    transition_type: str = "narrative",
    transition_duration: float = 0.5,
    narrator_voice: str = "warm_storyteller",
    fast_mode: bool = True,
    story_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint that coordinates the high-speed production run:
      1. Pre-warms Modal cloud GPU containers in parallel with screenplay drafting.
      2. Injects relevant directives learned from past films into the prompt.
      3. Renders characters, keyframes, and Wan 2.2 FLF2V video clips on Modal.
      4. Assembles movie with narrative-driven transitions and synced subtitles (.srt).
      5. Runs Multi-Tier AI Critic to evaluate assets and save new learned directives.
    """
    t_start = time.time()

    # 1. Compute dynamic ETA
    eta_sec, eta_breakdown = calculate_dynamic_eta(
        premise=premise,
        target_duration=target_duration,
        provider=provider,
        fast_mode=fast_mode,
        is_modal_warm=True,
    )

    scenes_count = eta_breakdown["scenes_count"]
    detected_chars = eta_breakdown["detected_chars"]

    # 2. Load past learned directives from memory bank
    kb = DirectorKnowledgeBase()
    past_directives = kb.retrieve_relevant_directives(premise, top_k=3)

    print("=" * 86)
    print(" 🎬 WAN 2.2 FAST ANIMATION STUDIO · 30-SECOND PRODUCTION CLI ".center(86, "="))
    print("=" * 86)
    print(f"📖 Premise          : {premise}")
    print(f"⏱️ Target Duration  : {target_duration:.1f}s ({scenes_count} atomic 5.0s scenes)")
    print(f"👥 Estimated Cast   : {detected_chars} character(s)")
    print(f"🌐 AI Director      : {provider.upper()} ({model or ('llama-3.3-70b-versatile' if provider == 'groq' else 'default')})")
    print(f"⚡ Video Backend    : Wan 2.2 FLF2V (8-Step Lightning on Modal Cloud GPU)")
    print(f"🎬 Transitions      : {transition_type.capitalize()} (Duration: {transition_duration:.1f}s)")
    if lora_path:
        print(f"🎯 Wan 2.2 LoRA     : {os.path.basename(lora_path)} (scale: {lora_scale})")
    if past_directives:
        print(f"🧠 Learned Directives: {len(past_directives)} past rules loaded from memory to avoid repeating mistakes")
        for idx, d in enumerate(past_directives, 1):
            print(f"    {idx}. [{d.get('category')}]: {d.get('directive_to_enforce')[:65]}...")
    print(f"⏳ Estimated Time   : ~{format_time(eta_sec)} ({eta_sec:.0f} seconds)")
    print("=" * 86 + "\n")

    tracker = TerminalProgressTracker(total_eta_seconds=eta_sec)

    def progress_callback(pct: float, msg: str) -> None:
        tracker.update(pct, msg)

    # 3. Instantiate Studio Pipeline
    pipeline = StudioPipeline(
        api_key=api_key,
        model=model,
        provider=provider,
    )

    # 4. Execute Full Pipeline
    progress_callback(0.02, "Drafting screenplay & consulting Director Memory...")
    story_data = pipeline.run_full_pipeline(
        user_prompt=premise,
        num_scenes=scenes_count,
        target_duration=target_duration,
        aspect_ratio="9:16 Vertical (Shorts/Reels)",
        art_style=art_style,
        use_vision_director=True,
        use_last_frame=True,
        enable_rife_2x=True,
        chain_previous_end_frame=False,
        anti_noise_guard=True,
        enable_narrator=True,
        narrator_voice=narrator_voice,
        transition_type=transition_type,
        transition_duration=transition_duration,
        lora_repo_or_path=lora_path,
        lora_scale=lora_scale,
        run_story_audit=True,
        auto_qa_retry=True,
        manage_gpu_phases=True,
        collect_telemetry=True,
        story_folder_name=story_folder,
        progress_cb=progress_callback,
    )

    elapsed_actual = round(time.time() - t_start, 1)
    tracker.complete(f"Film production finished in {format_time(elapsed_actual)}!")

    # 5. Extract Multi-Tier AI Critic Evaluation
    critic_report = story_data.get("critic_evaluation")
    if not critic_report:
        critic = SelfLearningCritic()
        critic_report = critic.evaluate_production_run(story_data)
        story_data["critic_evaluation"] = critic_report

    # 6. Display Production Summary & Post-Mortem Table
    print("\n" + "=" * 86)
    print(f" 🎬 PRODUCTION SUMMARY: '{story_data.get('title', 'Animation')}' ".center(86, "="))
    print("=" * 86)
    print(f"⏱️ Actual Duration   : {format_time(elapsed_actual)} (Estimated: ~{format_time(eta_sec)})")
    print(f"📁 Story Directory  : {story_data.get('story_dir')}")
    print(f"🎬 Final Movie Path : {story_data.get('final_movie_path')}")
    print(f"📝 Subtitle Path    : {story_data.get('subtitle_srt_path')}")
    print(f"📊 Telemetry Path   : {story_data.get('telemetry_report_path')}")
    print(f"🏆 Studio Score     : {critic_report.get('overall_studio_score', 9.0):.1f} / 10.0")

    print("\n" + "-" * 86)
    print(" 🧠 MULTI-TIER AI CRITIC EVALUATION & LESSONS LEARNED ".center(86, "-"))
    print("-" * 86)

    # Characters evaluation
    print("\n[Tier 1: Character Images (Z-Image Turbo)]")
    for ce in critic_report.get("character_evaluations", []):
        badge = "✅" if ce["score"] >= 8.5 else ("⚠️" if ce["score"] >= 7.0 else "❌")
        print(f"  {badge} {ce['target']} — Score: {ce['score']}/10.0 (Sharpness: {ce.get('sharpness', 0):.1f})")
        for finding in ce.get("findings", []):
            print(f"     • Finding: {finding}")
        for rec in ce.get("recommendations", []):
            print(f"     ➔ Recommendation: {rec}")

    # Keyframes evaluation
    print("\n[Tier 2: Event Keyframes Start vs End (Qwen Image Edit)]")
    for se in critic_report.get("scene_evaluations", []):
        badge = "✅" if se["score"] >= 8.5 else ("⚠️" if se["score"] >= 7.0 else "❌")
        print(f"  {badge} {se['target']} — Score: {se['score']}/10.0 (Pose Delta: {se.get('pose_delta', 0):.1f})")
        for finding in se.get("findings", []):
            print(f"     • Finding: {finding}")
        for rec in se.get("recommendations", []):
            print(f"     ➔ Recommendation: {rec}")

    # Overall animation evaluation
    print("\n[Tier 3: Overall Animation & Transition Flow]")
    me = critic_report.get("movie_evaluation", {})
    badge = "✅" if me.get("score", 10) >= 8.5 else "⚠️"
    print(f"  {badge} {me.get('target', 'Movie')} — Score: {me.get('score', 9.0)}/10.0 ({me.get('transition_type', 'narrative')})")
    for finding in me.get("findings", []):
        print(f"     • Finding: {finding}")
    for rec in me.get("recommendations", []):
        print(f"     ➔ Recommendation: {rec}")

    # Learned Directives
    new_rules_count = critic_report.get("new_directives_learned", 0)
    print("\n" + "-" * 86)
    if new_rules_count > 0:
        print(f"💡 Continuous Improvement: Stored {new_rules_count} new directive(s) in 'director_learned_rules.jsonl'.")
        print("   The AI Director will automatically enforce these rules in future films to avoid repeating flaws!")
    else:
        print("⭐ Flawless Run: Zero major flaws detected. Storyboard promoted to gold exemplar bank.")
    print("=" * 86 + "\n")

    return story_data


# ── Interactive CLI Parser ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wan 2.2 Fast Story Animation CLI & Self-Learning Production Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fast_story_cli.py
  python fast_story_cli.py "A tiny robot discovers a glowing sunflower in an abandoned greenhouse"
  python fast_story_cli.py "Cyberpunk cat rescues a holographic sparrow" --duration 30.0 --provider groq
        """,
    )
    parser.add_argument("premise", nargs="?", default="", help="Story premise / narrative premise")
    parser.add_argument("--duration", type=float, default=30.0, help="Target total movie duration in seconds (default: 30.0s)")
    parser.add_argument("--style", default="2D vibrant cartoon animation, Studio Ghibli watercolor aesthetic, warm lighting, 4K", help="Visual art style aesthetic")
    parser.add_argument("--provider", default="groq", choices=["groq", "local_ollama", "ollama_cloud"], help="AI Director LLM provider (default: groq)")
    parser.add_argument("--model", default=None, help="LLM model name override")
    parser.add_argument("--api-key", default=None, help="API key for Groq or Ollama Cloud")
    parser.add_argument("--lora", default="", help="Path or HuggingFace repo of Wan 2.2 LoRA weights")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="Wan 2.2 LoRA blend weight (default: 0.8)")
    parser.add_argument("--transition", default="narrative", help="Transition style ('narrative', 'crossfade', 'fade_black', 'whip_pan')")
    parser.add_argument("--transition-duration", type=float, default=0.5, help="Transition overlap duration in seconds (default: 0.5s)")
    parser.add_argument("--voice", default="af_heart", help="Voiceover narrator preset ('af_heart' Warm Storyteller, 'am_adam' Charming Male, 'af_bella' Anime, 'am_fenrir' Cinema, 'bf_emma' Fable)")
    parser.add_argument("--fast", action="store_true", default=True, help="Enable Fast Lightning mode (8-step DiT + Groq LPU)")
    parser.add_argument("--story-folder", "--story-name", default=None, help="Dedicated subfolder inside outputs/ for this story (e.g. 'fox_story')")

    args = parser.parse_args()

    user_premise = str(args.premise or "").strip()
    if not user_premise:
        print("\n" + "=" * 70)
        print(" 🎬 WAN 2.2 FAST ANIMATION STUDIO — INTERACTIVE MODE ".center(70, "="))
        print("=" * 70)
        user_premise = input("📖 Enter what kind of story you would like to animate:\n➔ ").strip()

    if not user_premise:
        user_premise = (
            "A deeply heart-touching short story: Silas, a gentle elderly clockmaker, rescues Pip, "
            "a tiny freezing robin on his frosted windowsill, warms him by the hearth, and watches "
            "him soar into the golden spring sky."
        )
        print(f"\n[Default Premise Selected]: \"{user_premise}\"")

    try:
        run_fast_story_production(
            premise=user_premise,
            target_duration=args.duration,
            art_style=args.style,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            lora_path=args.lora,
            lora_scale=args.lora_scale,
            transition_type=args.transition,
            transition_duration=args.transition_duration,
            narrator_voice=args.voice,
            fast_mode=args.fast,
            story_folder=args.story_folder,
        )
    except KeyboardInterrupt:
        print("\n\n[Aborted] Production canceled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ [Error] Production failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
