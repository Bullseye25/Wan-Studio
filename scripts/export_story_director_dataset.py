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
    dpo_pairs = []
    if memory_bank.dpo_file.exists():
        with open(memory_bank.dpo_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        dpo_pairs.append(json.loads(line))
                    except Exception:
                        pass

    # 4. Generate Orchestra AI Decision Maker Pairs (What to do vs What NOT to do)
    orchestra_sft_file = out_path / "orchestra_decision_sft.jsonl"
    orchestra_dpo_file = out_path / "orchestra_decision_dpo.jsonl"

    orchestra_dpo_records = [
        # Z-Image Turbo Genesis Rule
        {
            "prompt": "Orchestra Decision: Formulate character genesis prompt for Silas the elderly clockmaker.",
            "chosen": (
                "Full-body vertical character reference sheet of Silas, gentle crinkled blue eyes, silver mutton chops beard, "
                "soft brown woolen vest, canvas apron, holding brass jeweler loupe in hand. 9:16 vertical portrait, head-to-toe visible, "
                "clean neutral studio backdrop. Negative prompt: multiple people, crowd, cropped limbs, distorted hands, complex background."
            ),
            "rejected": (
                "Silas the clockmaker in his busy crowded workshop full of clocks, close up face portrait with other apprentices in the background."
            ),
            "critique": "Z-Image Turbo reference sheets must be single-character full-body on neutral backdrops to prevent identity corruption in Qwen.",
        },
        # Qwen Dual-Keyframe Rule
        {
            "prompt": "Orchestra Decision: Formulate Qwen end keyframe prompt for Scene 1 (Silas finds robin).",
            "chosen": (
                "image 1 is the starting frame. For this ending frame at t=5.0s, transform Silas: Silas is now kneeling beside his stone fireplace hearth, "
                "tenderly cradling the robin wrapped in a soft wool handkerchief against his chest with a warm smile. Keep identical workshop clocks, "
                "wall textures, and rustic room geometry."
            ),
            "rejected": (
                "A different camera angle showing Silas sitting in his living room having tea with the robin outside."
            ),
            "critique": "End frame must anchor on image 1 with high pose delta while explicitly locking background geometry and camera angle.",
        },
        # Wan 2.2 Temporal Timeline Rule
        {
            "prompt": "Orchestra Decision: Formulate Wan 2.2 animation motion prompt for Scene 1 (5.0s clip).",
            "chosen": (
                "Slow dolly-in moving gently toward Silas on a 50mm lens. "
                "Phase 1 (0.0s-1.5s): Initiating from Silas standing by frosted window. "
                "Phase 2 (1.5s-3.5s): Silas scoops the shivering robin into his handkerchief and turns toward hearth. "
                "Phase 3 (3.5s-5.0s): Settling onto one knee by the warm embers. "
                "Rigid background architecture, locked lighting, smooth continuous motion, zero camera jitter, no morphing."
            ),
            "rejected": (
                "Silas picks up the robin while simultaneously clock cuckoos fly out and then a snowstorm breaks the window and Silas runs away."
            ),
            "critique": "Wan 2.2 clips can only execute ONE atomic action and reaction in 5s. Rambling run-ons cause latent noise boiling.",
        },
        # Character Budget Rule
        {
            "prompt": "Orchestra Decision: Direct a heart-touching short story cast budget.",
            "chosen": (
                "Limit cast to 2-3 characters (e.g. Silas and Robin) for maximum emotional intimacy and visual stability."
            ),
            "rejected": (
                "Introduce 8 village townspeople, three guards, and a flock of birds in a 5-scene short."
            ),
            "critique": "Heart-touching shorts must adhere to <= 3 characters max to avoid multi-character identity confusion in Qwen.",
        },
        # Transition Mode Rule
        {
            "prompt": "Orchestra Decision: Select transition between emotional scenes.",
            "chosen": (
                "Apply 'crossfade' (0.5s gentle dissolve) via FFmpeg xfade with acrossfade audio to convey emotional passage of time."
            ),
            "rejected": (
                "Use high-speed whip-pan or jarring cut for an intimate hearth-side emotional beat."
            ),
            "critique": "Intimate heart-touching narratives require soft cross-dissolves or dip-to-black rather than fast action transitions.",
        },
    ]

    for pair in dpo_pairs:
        prompt_text = (
            f"Direct a logical animation storyboard for premise: {pair.get('premise', '')}. "
            "Ensure causal connectors, prop permanence, and 5-second feasibility."
        )
        chosen_text = json.dumps(pair.get("chosen", {}), indent=2, ensure_ascii=False)
        rejected_text = json.dumps(pair.get("rejected", {}), indent=2, ensure_ascii=False)
        orchestra_dpo_records.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "critique": pair.get("critique_feedback", ""),
        })

    with open(dpo_file, "w", encoding="utf-8") as f:
        for r in orchestra_dpo_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(orchestra_dpo_file, "w", encoding="utf-8") as f:
        for r in orchestra_dpo_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Orchestra SFT Dataset
    orchestra_sft_records = [
        {
            "conversations": [
                {"from": "human", "value": r["prompt"]},
                {"from": "gpt", "value": r["chosen"]},
            ]
        }
        for r in orchestra_dpo_records
    ]
    with open(orchestra_sft_file, "w", encoding="utf-8") as f:
        for r in orchestra_sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[Export Completed]")
    print(f"  -> SFT (ShareGPT):        {sft_file} ({len(sft_records)} examples)")
    print(f"  -> SFT (Alpaca):          {alpaca_file} ({len(alpaca_records)} examples)")
    print(f"  -> DPO Preference:        {dpo_file} ({len(orchestra_dpo_records)} preference pairs)")
    print(f"  -> Orchestra Decision SFT: {orchestra_sft_file} ({len(orchestra_sft_records)} pairs)")
    print(f"  -> Orchestra Decision DPO: {orchestra_dpo_file} ({len(orchestra_dpo_records)} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Story Director datasets for fine-tuning")
    parser.add_argument("--output-dir", default="dataset", help="Destination folder for exported datasets")
    args = parser.parse_args()
    export_datasets(args.output_dir)
