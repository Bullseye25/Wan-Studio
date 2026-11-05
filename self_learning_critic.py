"""
Autonomous Multi-Tier AI Critic, Quality Evaluator & Self-Learning Memory Engine.
================================================================================
Evaluates every asset in the production lifecycle:
  1. Character Reference Images (Z-Image Turbo)
  2. Scene Event Keyframes: Start (t=0s) vs End (t=5s) (Qwen Image Edit)
  3. Video Clips & Overall Final Animation (Wan 2.2 FLF2V + FFmpeg)

Compiles actionable diagnostics into persistent 'Learned Production Directives'
(director_learned_rules.jsonl). On future production runs, the AI Director
retrieves relevant directives to automatically prevent repeating past mistakes.
"""

import os
import re
import json
import time
import pathlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

from orchestra_rules import (
    MAX_STORY_CHARACTERS,
    audit_prompt_for_anti_patterns,
)


class CharacterImageEvaluator:
    """Evaluates character reference portraits for studio isolation, sharpness, and prompt adherence."""

    @staticmethod
    def evaluate(char_dict: Dict[str, Any]) -> Dict[str, Any]:
        name = char_dict.get("name", "Character")
        cid = char_dict.get("id", "char")
        path = char_dict.get("portrait_path")
        prompt = char_dict.get("z_image_prompt", "")

        score = 10.0
        findings = []
        recommendations = []

        # 1. Prompt anti-pattern audit
        violations = audit_prompt_for_anti_patterns(prompt, "z_image_turbo")
        if violations:
            score -= len(violations) * 1.5
            findings.append(f"Prompt contains anti-patterns: {', '.join(violations)}")
            recommendations.append("Remove prohibited phrases (e.g. 'cinematic lighting', multi-angle shots) from character prompt.")

        sharpness = 0.0
        border_std = 0.0
        is_isolated = True

        # 2. Vision analysis on generated image file
        if path and os.path.exists(path):
            try:
                with Image.open(path) as img:
                    rgb = np.array(img.convert("RGB"))
                    h, w, _ = rgb.shape
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

                    # Inspect outer 8% border ring
                    bx = max(4, int(w * 0.08))
                    by = max(4, int(h * 0.08))
                    top = gray[:by, :]
                    left = gray[:, :bx]
                    right = gray[:, w - bx:]
                    border_pixels = np.concatenate([top.flatten(), left.flatten(), right.flatten()])
                    border_std = round(float(np.std(border_pixels)), 2)

                    # Background isolation check
                    if border_std > 48.0:
                        is_isolated = False
                        score -= 2.5
                        findings.append(f"Border studio backdrop has high variance (std={border_std:.1f} > 48.0), indicating environmental clutter.")
                        recommendations.append("Specify 'solid neutral light-grey studio backdrop, zero background furniture or landscape' in character prompt.")

                    # Sharpness check
                    if sharpness < 30.0:
                        score -= 1.5
                        findings.append(f"Character Laplacian sharpness is relatively soft ({sharpness:.1f} < 30.0).")
                        recommendations.append("Reinforce 'high-contrast specular rim light, crisp 2D animation lineart' in character prompt.")

            except Exception as e:
                findings.append(f"Image read error: {e}")
        else:
            score = 5.0
            findings.append("Character portrait file is missing on disk.")
            recommendations.append("Ensure character generation phase completes before evaluating.")

        score = max(1.0, min(10.0, round(score, 1)))
        status = "EXCELLENT" if score >= 8.5 else ("ACCEPTABLE" if score >= 7.0 else "NEEDS_IMPROVEMENT")

        return {
            "target": f"Character '{name}' ({cid})",
            "type": "character_image",
            "score": score,
            "status": status,
            "sharpness": sharpness,
            "border_uniformity_std": border_std,
            "is_isolated_backdrop": is_isolated,
            "findings": findings or ["Flawless studio portrait with pristine background isolation."],
            "recommendations": recommendations or ["Maintain current prompt formulation and seed parameters."],
        }


class KeyframeEvaluator:
    """Evaluates Start (t=0s) and End (t=5s) scene keyframes for composition, pose delta, and background stability."""

    @staticmethod
    def evaluate(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        snum = scene_dict.get("scene_number", 1)
        title = scene_dict.get("title", f"Scene {snum}")
        start_path = scene_dict.get("keyframe_path")
        end_path = scene_dict.get("last_keyframe_path")
        compat = scene_dict.get("flf2v_compat") or {}
        pose_delta = float(scene_dict.get("pose_delta") or compat.get("pose_delta") or 0.0)
        bg_ssim = float(compat.get("bg_ssim") or compat.get("bg_similarity") or 1.0)
        state_ledger = scene_dict.get("state_ledger") or {}
        active_props = state_ledger.get("active_props") or []

        score = 10.0
        findings = []
        recommendations = []

        # 1. Pose Delta Gatekeeper (Wan 2.2 FLF2V requires noticeable motion)
        if pose_delta < 18.0:
            score -= 3.0
            findings.append(f"Start-to-End Pose Delta is too low ({pose_delta:.1f} < 18.0 threshold), risking a static/frozen video clip.")
            recommendations.append(
                "Inject a visibly contrasting physical follow-through posture into the End Frame prompt (e.g. 'crouched low shielding face' or 'arms raised high')."
            )
        elif pose_delta > 85.0:
            score -= 2.0
            findings.append(f"Start-to-End Pose Delta is excessively high ({pose_delta:.1f} > 85.0), risking mid-clip morphing/ghosting.")
            recommendations.append("Route high-velocity action to Guided I2V mode or restrict extreme limb jumps between keyframes.")
        else:
            findings.append(f"Optimal FLF2V Pose Delta ({pose_delta:.1f} in sweet spot 18.0–85.0).")

        # 2. Background Geometry Consistency
        if bg_ssim < 0.68:
            score -= 2.5
            findings.append(f"Peripheral background similarity is low ({bg_ssim:.2f} < 0.68), indicating room geometry or tree placement shifted.")
            recommendations.append("Apply character-safe peripheral border lock (border_ratio<=0.08, character_safe=True) and instruct End Frame prompt to lock room geometry identically.")
        else:
            findings.append(f"Solid background stability ({bg_ssim:.2f} >= 0.68).")

        # 3. Prop / Weapon Retention Audit
        start_p = str(scene_dict.get("qwen_scene_prompt") or "").lower()
        end_p = str(scene_dict.get("qwen_end_scene_prompt") or "").lower()
        for prop in active_props:
            if not prop:
                continue
            p_clean = prop.lower()
            if p_clean in start_p and p_clean not in end_p:
                score -= 1.0
                findings.append(f"Held prop '{prop}' mentioned in start prompt was omitted in end prompt.")
                recommendations.append(f"Explicitly re-state '{prop}' in End Frame prompt to prevent vanishing mid-shot.")

        score = max(1.0, min(10.0, round(score, 1)))
        status = "EXCELLENT" if score >= 8.5 else ("ACCEPTABLE" if score >= 7.0 else "NEEDS_IMPROVEMENT")

        return {
            "target": f"Scene #{snum}: '{title}'",
            "type": "event_keyframes",
            "scene_number": snum,
            "score": score,
            "status": status,
            "pose_delta": pose_delta,
            "bg_similarity": bg_ssim,
            "findings": findings,
            "recommendations": recommendations or ["Keyframe pair has balanced kinetic progression and locked room geometry."],
        }


class AnimationMovieEvaluator:
    """Evaluates final movie animation flow, optical flow dynamics, audio-visual sync, and transition rhythm."""

    @staticmethod
    def evaluate(story_data: Dict[str, Any]) -> Dict[str, Any]:
        scenes = story_data.get("scenes") or []
        movie_path = story_data.get("final_movie_path")
        trans_type = story_data.get("transition_type", "narrative")

        score = 10.0
        findings = []
        recommendations = []

        if not movie_path or not os.path.exists(movie_path):
            return {
                "target": "Final Movie",
                "type": "overall_animation",
                "score": 4.0,
                "status": "FAILED",
                "findings": ["Final stitched movie file is missing."],
                "recommendations": ["Re-run FFmpeg clip assembly."],
            }

        # 1. Optical Flow & Motion Surge Analysis
        surges = []
        sync_deltas = []
        for s in scenes:
            tel = s.get("telemetry") or {}
            surge = tel.get("surge_ratio")
            if surge is not None:
                surges.append(float(surge))
            sync_err = tel.get("audio_sync_delta_sec")
            if sync_err is not None:
                sync_deltas.append(abs(float(sync_err)))

        if surges:
            avg_surge = float(np.mean(surges))
            if avg_surge > 3.0:
                score -= 1.5
                findings.append(f"High optical flow motion surge detected ({avg_surge:.2f}x baseline), indicating brief jitter or sudden camera shake.")
                recommendations.append("Set Wan motion_intensity to 2 and enable Anti-Noise Rotation Guard.")
            else:
                findings.append(f"Smooth, cinematic optical flow across clips (mean surge: {avg_surge:.2f}x).")

        # 2. Audio-Visual Lip/Beat Sync Offset
        if sync_deltas:
            max_sync_err = max(sync_deltas)
            if max_sync_err > 0.45:
                score -= 1.5
                findings.append(f"Voiceover sync offset reached {max_sync_err:.2f}s (threshold: <=0.40s).")
                recommendations.append("Align sync trigger action verb with the 1.2s–3.8s visual acceleration window.")
            else:
                findings.append(f"Tight audio-visual synchronization (max offset: {max_sync_err:.2f}s).")

        # 3. Transition Pacing
        if len(scenes) >= 2:
            findings.append(f"Assembled with cinematic transition flow: '{trans_type}'.")

        score = max(1.0, min(10.0, round(score, 1)))
        status = "EXCELLENT" if score >= 8.5 else ("ACCEPTABLE" if score >= 7.0 else "NEEDS_IMPROVEMENT")

        return {
            "target": f"Movie: '{story_data.get('title', 'Animation')}'",
            "type": "overall_animation",
            "score": score,
            "status": status,
            "scene_count": len(scenes),
            "transition_type": trans_type,
            "findings": findings or ["Continuous narrative flow with clean cinematic transitions."],
            "recommendations": recommendations or ["Film achieves professional pacing and synchronized climax cadence."],
        }


class DirectorKnowledgeBase:
    """
    Persistent self-evolving memory bank for the AI Director.
    Extracts structured 'Learned Production Directives' from post-mortem evaluations
    and queries them on subsequent runs to prevent repeating mistakes.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            base = pathlib.Path(storage_dir)
        else:
            base = pathlib.Path(__file__).parent / "outputs" / "director_memory"
        base.mkdir(parents=True, exist_ok=True)
        self.rules_file = base / "director_learned_rules.jsonl"

    def record_learned_directives(
        self,
        directives: List[Dict[str, Any]],
        story_title: str = "",
        premise: str = "",
    ) -> int:
        """Appends new actionable directives learned from an animation run to disk."""
        if not directives:
            return 0
        written = 0
        timestamp = int(time.time())
        try:
            with open(self.rules_file, "a", encoding="utf-8") as f:
                for idx, d in enumerate(directives):
                    record = {
                        "id": f"rule_{timestamp}_{idx+1}",
                        "timestamp": timestamp,
                        "story_title": story_title,
                        "category": d.get("category", "General"),
                        "severity": d.get("severity", "MEDIUM"),
                        "mistake_observed": d.get("mistake_observed", ""),
                        "directive_to_enforce": d.get("directive_to_enforce", ""),
                        "tags": d.get("tags", []),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            return written
        except Exception as e:
            print(f"[DirectorKnowledgeBase] Failed to save directives: {e}")
            return 0

    def retrieve_relevant_directives(self, premise: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant learned directives based on story premise keywords.
        Guarantees that past mistakes in similar genres/topics are actively avoided.
        """
        all_rules = []
        files_to_check = [self.rules_file]
        legacy_file = pathlib.Path(__file__).parent / "outputs" / "studio" / "director_learned_rules.jsonl"
        if legacy_file.exists() and legacy_file != self.rules_file:
            files_to_check.append(legacy_file)

        for rf in files_to_check:
            if not rf.exists():
                continue
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_rules.append(json.loads(line))
            except Exception as e:
                print(f"[DirectorKnowledgeBase] Error loading rules from {rf}: {e}")

        if not all_rules:
            return []

        # Score rules by keyword and tag overlap
        query_words = set(re.findall(r"\w+", premise.lower()))
        scored = []
        for r in all_rules:
            rule_text = f"{r.get('mistake_observed', '')} {r.get('directive_to_enforce', '')} {' '.join(r.get('tags', []))}".lower()
            rule_words = set(re.findall(r"\w+", rule_text))
            score = len(query_words.intersection(rule_words))
            if r.get("severity") == "HIGH":
                score += 2
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate by directive text
        seen = set()
        unique_results = []
        for _, r in scored:
            text = r.get("directive_to_enforce", "")
            if text and text not in seen:
                seen.add(text)
                unique_results.append(r)
            if len(unique_results) >= top_k:
                break

        return unique_results

    # Alias for API compatibility across modules
    get_top_rules_for_premise = retrieve_relevant_directives

    def format_directives_prompt(self, directives: List[Dict[str, Any]]) -> str:
        """Formats retrieved directives into an authoritative system prompt injection."""
        if not directives:
            return ""

        lines = [
            "### 🧠 LEARNED PRODUCTION DIRECTIVES (Rules from Previous Film Post-Mortems - AVOID PAST MISTAKES):"
        ]
        for idx, d in enumerate(directives, 1):
            lines.append(
                f"{idx}. [{d.get('category', 'Directive')} - {d.get('severity', 'MEDIUM')}]: "
                f"Previous flaw observed: '{d.get('mistake_observed')}'. "
                f"MANDATORY DIRECTIVE: {d.get('directive_to_enforce')}"
            )
        lines.append("")
        return "\n".join(lines)


class SelfLearningCritic:
    """
    Master coordinator for autonomous multi-tier production critique & continuous learning.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        self.kb = DirectorKnowledgeBase(storage_dir=storage_dir)

    def evaluate_production_run(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete multi-tier critique across characters, keyframes, and final animation.
        Extracts new actionable directives and persists them into the knowledge base.
        """
        characters = story_data.get("characters") or []
        scenes = story_data.get("scenes") or []
        title = story_data.get("title", "Untitled Film")
        premise = story_data.get("theme", "") or title

        # 1. Tier 1: Character reference portraits
        char_evals = [CharacterImageEvaluator.evaluate(c) for c in characters]

        # 2. Tier 2: Scene event keyframes
        scene_evals = [KeyframeEvaluator.evaluate(s) for s in scenes]

        # 3. Tier 3: Overall movie animation
        movie_eval = AnimationMovieEvaluator.evaluate(story_data)

        # 4. Extract actionable learned directives from sub-optimal signals (score < 8.5)
        new_directives: List[Dict[str, Any]] = []
        premise_tags = list(set(re.findall(r"\w+", premise.lower())))[:6]

        for ce in char_evals:
            if ce["score"] < 8.5:
                for finding, rec in zip(ce["findings"], ce["recommendations"]):
                    new_directives.append({
                        "category": "Character Isolation & Silhouette",
                        "severity": "HIGH" if ce["score"] < 7.0 else "MEDIUM",
                        "mistake_observed": finding,
                        "directive_to_enforce": rec,
                        "tags": premise_tags + ["character", "portrait"],
                    })

        for se in scene_evals:
            if se["score"] < 8.5:
                for finding, rec in zip(se["findings"], se["recommendations"]):
                    new_directives.append({
                        "category": "Keyframe Dynamics & Pose Delta",
                        "severity": "HIGH" if se["score"] < 7.0 else "MEDIUM",
                        "mistake_observed": finding,
                        "directive_to_enforce": rec,
                        "tags": premise_tags + ["scene", "flf2v", f"scene_{se['scene_number']}"],
                    })

        if movie_eval["score"] < 8.5:
            for finding, rec in zip(movie_eval["findings"], movie_eval["recommendations"]):
                new_directives.append({
                    "category": "Animation Rhythm & Audio-Visual Sync",
                    "severity": "HIGH" if movie_eval["score"] < 7.0 else "MEDIUM",
                    "mistake_observed": finding,
                    "directive_to_enforce": rec,
                    "tags": premise_tags + ["video", "animation", "transitions"],
                })

        # Save to persistent knowledge base
        num_saved = self.kb.record_learned_directives(new_directives, story_title=title, premise=premise)

        # Calculate weighted composite studio score (10.0 scale)
        all_scores = [c["score"] for c in char_evals] + [s["score"] for s in scene_evals] + [movie_eval["score"]]
        overall_score = round(float(np.mean(all_scores)) if all_scores else 10.0, 1)

        report = {
            "title": title,
            "overall_studio_score": overall_score,
            "character_evaluations": char_evals,
            "scene_evaluations": scene_evals,
            "movie_evaluation": movie_eval,
            "new_directives_learned": num_saved,
            "directives_list": new_directives,
        }

        return report
