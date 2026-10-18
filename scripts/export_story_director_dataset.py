#!/usr/bin/env python3
"""
Dataset Exporter & Formatter for Logical Story Director Fine-Tuning.
Compiles the Long-Term Episodic Memory Bank, Causal Narrative Blueprints,
and DPO Preference Pairs into standard training datasets for:
1. Supervised Fine-Tuning (SFT) - ShareGPT & Alpaca formats
2. Direct Preference Optimization (DPO) - Preference pairs
"""

import os
import sys
import json
import pathlib
import argparse
from typing import Dict, Any, List

# Ensure parent directory is on python path
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cinematography_vault import CAUSAL_NARRATIVE_BLUEPRINTS, StoryMemoryBank, NARRATIVE_BEATS_BY_COUNT


def build_sft_conversation(premise: str, story: Dict[str, Any], art_style: str = "") -> Dict[str, Any]:
    """Builds a ShareGPT-formatted multi-turn conversation pair."""
    num_scenes = len(story.get("scenes", []))
    style = art_style or story.get("art_style", "High-end cinematic animation")
    user_prompt = (
        f"Story Premise: {premise}\n"
        f"Target Style: {style}\n"
        f"Number of Scenes: {num_scenes}\n"
        f"Total Duration: {round(num_scenes * 5.0, 1)}s."
    )
    # Target output is pure JSON with state ledgers and causal connectors
    assistant_output = json.dumps(
        {
            "title": story.get("title", "Studio Story"),
            "art_style": style,
            "theme": story.get("theme", premise),
            "characters": story.get("characters", []),
            "scenes": [
                {
                    "scene_number": s.get("scene_number"),
                    "title": s.get("title"),
                    "duration_seconds": s.get("duration_seconds", 5.0),
                    "narrative_beat": s.get("narrative_beat"),
                    "causal_connector": s.get("causal_connector"),
                    "state_ledger": s.get("state_ledger", {}),
                    "scenario": s.get("scenario"),
                    "narration_text": s.get("narration_text", ""),
                    "narration_voice_style": s.get("narration_voice_style", "warm_storyteller"),
                    "sync_trigger_word": s.get("sync_trigger_word", ""),
                    "qwen_scene_prompt": s.get("qwen_scene_prompt"),
                    "qwen_end_scene_prompt": s.get("qwen_end_scene_prompt"),
                    "wan_motion_prompt": s.get("wan_motion_prompt"),
                    "camera_movement": s.get("camera_movement"),
                    "character_ids": s.get("character_ids", ["char_1"]),
                }
                for s in story.get("scenes", [])
            ],
        },
        indent=2,
        ensure_ascii=False,
    )

    return {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "You are a master animation director directing a causally coherent animated short film. "
                    "Output a complete JSON storyboard with Causal State Ledgers, 'THEREFORE/BUT' transitions, "
                    "5-second atomic physical actions, and synchronized 6-11 word voiceover narration (`narration_text`, `sync_trigger_word`).\n\n"
                    + user_prompt
                ),
            },
            {
                "from": "gpt",
                "value": assistant_output,
            },
        ]
    }


def export_datasets(output_dir: str = "dataset"):
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sft_file = out_path / "story_director_sft.jsonl"
    dpo_file = out_path / "story_director_dpo.jsonl"
    alpaca_file = out_path / "story_director_alpaca.jsonl"

    memory_bank = StoryMemoryBank()
    all_exemplars = list(CAUSAL_NARRATIVE_BLUEPRINTS)

    # 1. Load memory exemplars
    if memory_bank.memory_file.exists():
        with open(memory_bank.memory_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_exemplars.append(json.loads(line))
                    except Exception:
                        pass

    # 2. Export SFT Datasets
    sft_records = []
    alpaca_records = []
    for ex in all_exemplars:
        premise = ex.get("premise") or ex.get("theme") or ex.get("title", "")
        if not premise:
            continue
        conv = build_sft_conversation(premise, ex, art_style=ex.get("art_style", ""))
        sft_records.append(conv)

        alpaca_records.append({
            "instruction": "Direct a causally coherent animation storyboard with Causal State Ledgers and 5s clip feasibility.",
            "input": conv["conversations"][0]["value"],
            "output": conv["conversations"][1]["value"],
        })

    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(alpaca_file, "w", encoding="utf-8") as f:
        for r in alpaca_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. Export DPO Preference Dataset
    dpo_count = 0
    if memory_bank.dpo_file.exists():
        dpo_pairs = []
        with open(memory_bank.dpo_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        dpo_pairs.append(json.loads(line))
                    except Exception:
                        pass

        with open(dpo_file, "w", encoding="utf-8") as f:
            for pair in dpo_pairs:
                prompt_text = (
                    f"Direct a logical animation storyboard for premise: {pair.get('premise', '')}. "
                    "Ensure causal connectors, prop permanence, and 5-second feasibility."
                )
                chosen_text = json.dumps(pair.get("chosen", {}), indent=2, ensure_ascii=False)
                rejected_text = json.dumps(pair.get("rejected", {}), indent=2, ensure_ascii=False)
                dpo_record = {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "critique": pair.get("critique_feedback", ""),
                }
                f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                dpo_count += 1

    print(f"\n[Export Completed]")
    print(f"  -> SFT (ShareGPT): {sft_file} ({len(sft_records)} examples)")
    print(f"  -> SFT (Alpaca):   {alpaca_file} ({len(alpaca_records)} examples)")
    print(f"  -> DPO Preference: {dpo_file} ({dpo_count} preference pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Story Director datasets for fine-tuning")
    parser.add_argument("--output-dir", default="dataset", help="Destination folder for exported datasets")
    args = parser.parse_args()
    export_datasets(args.output_dir)
