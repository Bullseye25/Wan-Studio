"""
Unit and Integration Test Suite for Orchestra AI Decision Maker.
Verifies model operational rules, anti-pattern detection, max 3 characters budget,
heart-touching story blueprints, and FFmpeg cinematic transition generation.
"""

import unittest
from unittest.mock import patch, MagicMock
import pathlib
import json

from orchestra_rules import (
    MAX_STORY_CHARACTERS,
    MAX_CLIP_DURATION_SEC,
    ORCHESTRA_KNOWLEDGE_BASE,
    TransitionType,
    validate_story_budget,
    enforce_orchestra_story_constraints,
    audit_prompt_for_anti_patterns,
    compile_z_image_prompt,
    compile_qwen_end_frame_prompt,
    compile_wan_motion_timeline,
    build_ffmpeg_transition_filter,
    calculate_transition_srt_timestamps,
)
from cinematography_vault import CAUSAL_NARRATIVE_BLUEPRINTS
from narrator_engine import AudioVisualSyncEngine
from studio_pipeline import StudioPipeline


class TestOrchestraRulesAndKnowledgeBase(unittest.TestCase):
    def test_knowledge_base_structure(self):
        """Verifies knowledge base exists for all 3 engines with approved and forbidden rules."""
        for engine in ["z_image_turbo", "qwen_image_edit", "wan_2_2"]:
            self.assertIn(engine, ORCHESTRA_KNOWLEDGE_BASE)
            kb = ORCHESTRA_KNOWLEDGE_BASE[engine]
            self.assertIn("what_to_do", kb)
            self.assertIn("what_not_to_do", kb)
            self.assertIn("anti_patterns", kb)
            self.assertGreater(len(kb["what_to_do"]), 0)
            self.assertGreater(len(kb["what_not_to_do"]), 0)

    def test_anti_pattern_auditing(self):
        """Verifies that prohibited patterns are detected by the auditor."""
        # Z-Image: Multiple people
        bad_z_prompt = "Two people standing together in a crowded street"
        violations_z = audit_prompt_for_anti_patterns(bad_z_prompt, "z_image_turbo")
        self.assertGreater(len(violations_z), 0)

        # Qwen: Changed background
        bad_q_prompt = "image 1 is starting frame. Now show completely different scene and new background"
        violations_q = audit_prompt_for_anti_patterns(bad_q_prompt, "qwen_image_edit")
        self.assertGreater(len(violations_q), 0)

        # Wan: Simultaneous run-on action
        bad_w_prompt = "Hero jumps while simultaneously shooting laser and teleports away"
        violations_w = audit_prompt_for_anti_patterns(bad_w_prompt, "wan_2_2")
        self.assertGreater(len(violations_w), 0)

    def test_prompt_compilation(self):
        """Verifies gold-standard prompt builders adhere to model constraints."""
        char = {
            "name": "Silas",
            "face": "crinkled blue eyes",
            "hair": "silver beard",
            "clothing": "wool vest",
            "signature_details": "brass loupe",
        }
        pos, neg = compile_z_image_prompt(char)
        self.assertIn("Full-body vertical", pos)
        self.assertIn("holding brass loupe", pos)
        self.assertIn("neutral studio backdrop", pos)
        self.assertIn("multiple people", neg)

        qwen_prompt = compile_qwen_end_frame_prompt(
            start_scenario="Silas at window",
            target_action="kneeling by hearth cradling robin",
            character_names=["Silas"],
        )
        self.assertIn("image 1 is the starting frame", qwen_prompt)
        self.assertIn("Keep identical environment geometry", qwen_prompt)

        wan_prompt = compile_wan_motion_timeline(
            scene_number=1,
            entry_state="standing at frosted window",
            atomic_action="cradles robin and turns toward hearth",
            exit_state="kneeling by fireplace",
            duration_sec=5.0,
        )
        self.assertIn("Phase 1 (0.0s-1.5s)", wan_prompt)
        self.assertIn("Phase 2 (1.5s-3.5s)", wan_prompt)
        self.assertIn("Phase 3 (3.5s-5.0s)", wan_prompt)


class TestHeartTouchingStoryBudget(unittest.TestCase):
    def test_character_budget_validation(self):
        """Validates that stories with <= 3 characters pass, and > 3 characters fail."""
        story_valid = {
            "characters": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
            "scenes": [{"scene_number": 1, "duration_seconds": 4.5, "narration_text": "A sweet winter morning arrives."}],
        }
        ok, errs = validate_story_budget(story_valid)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

        story_overflow = {
            "characters": [{"id": f"c{i}"} for i in range(5)],
            "scenes": [{"scene_number": 1, "duration_seconds": 4.5}],
        }
        ok, errs = validate_story_budget(story_overflow)
        self.assertFalse(ok)
        self.assertTrue(any("budget exceeded" in e for e in errs))

    def test_enforce_orchestra_story_constraints(self):
        """Validates automatic character pruning to max 3 and duration clamping."""
        story_raw = {
            "characters": [
                {"id": "char_1", "name": "Silas"},
                {"id": "char_2", "name": "Robin"},
                {"id": "char_3", "name": "Cat"},
                {"id": "char_4", "name": "Villager A"},
                {"id": "char_5", "name": "Villager B"},
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "duration_seconds": 8.5,  # Exceeds Wan max 5.0s
                    "character_ids": ["char_1", "char_4"],
                    "qwen_end_scene_prompt": "Silas hands Robin to Cat",
                }
            ],
        }
        repaired = enforce_orchestra_story_constraints(story_raw)
        self.assertEqual(len(repaired["characters"]), MAX_STORY_CHARACTERS)
        self.assertEqual(repaired["scenes"][0]["duration_seconds"], MAX_CLIP_DURATION_SEC)
        # char_4 was pruned, so scene character_ids must only contain allowed ids
        self.assertEqual(repaired["scenes"][0]["character_ids"], ["char_1"])
        # Qwen end prompt must have background lock added
        self.assertIn("Keep identical environment geometry", repaired["scenes"][0]["qwen_end_scene_prompt"])

    def test_curated_heart_touching_blueprints(self):
        """Verifies all curated narrative blueprints adhere strictly to <= 3 characters constraint."""
        self.assertGreaterEqual(len(CAUSAL_NARRATIVE_BLUEPRINTS), 5)
        for bp in CAUSAL_NARRATIVE_BLUEPRINTS:
            chars = bp.get("characters", [])
            self.assertLessEqual(
                len(chars),
                MAX_STORY_CHARACTERS,
                f"Blueprint '{bp.get('title')}' has {len(chars)} characters; must be <= {MAX_STORY_CHARACTERS}",
            )
            for sc in bp.get("scenes", []):
                self.assertIn("state_ledger", sc)
                self.assertIn("narration_text", sc)
                self.assertIn("sync_trigger_word", sc)


class TestCinematicTransitions(unittest.TestCase):
    def test_build_ffmpeg_transition_filter_modes(self):
        """Tests filter construction for cut, crossfade, fade_black, and match_cut."""
        durations = [5.0, 5.0, 5.0]

        # 1. Cut
        v_cut, a_cut, dur_cut = build_ffmpeg_transition_filter(durations, "cut", 0.5, has_audio=True)
        self.assertIn("concat=n=3:v=1:a=0", v_cut)
        self.assertIn("concat=n=3:v=0:a=1", a_cut)
        self.assertEqual(dur_cut, 15.0)

        # 2. Crossfade
        v_cross, a_cross, dur_cross = build_ffmpeg_transition_filter(durations, "crossfade", 0.5, has_audio=True)
        self.assertIn("xfade=transition=fade", v_cross)
        self.assertIn("acrossfade=d=0.50", a_cross)
        # 15.0s minus two 0.5s overlaps = 14.0s
        self.assertEqual(dur_cross, 14.0)

        # 3. Fade to Black
        v_black, a_black, dur_black = build_ffmpeg_transition_filter(durations, "fade_black", 0.5, has_audio=False)
        self.assertIn("xfade=transition=fadeblack", v_black)
        self.assertEqual(a_black, "")
        self.assertEqual(dur_black, 14.0)

        # 4. Match Cut / Directional Wipe
        v_match, a_match, _ = build_ffmpeg_transition_filter(durations, "match_cut", 0.4, has_audio=True)
        self.assertIn("xfade=transition=wipeleft", v_match)

    def test_calculate_transition_srt_timestamps(self):
        """Verifies that SRT subtitle timing offsets are adjusted by the transition overlap."""
        durations = [5.0, 5.0, 5.0]
        # With 0.5s crossfade, scene 1 is [0, 5], scene 2 starts at 4.5 and ends at 9.5, scene 3 starts at 9.0 and ends at 14.0
        ts = calculate_transition_srt_timestamps(durations, "crossfade", 0.5)
        self.assertEqual(len(ts), 3)
        self.assertEqual(ts[0], (0.0, 5.0))
        self.assertEqual(ts[1], (4.5, 9.5))
        self.assertEqual(ts[2], (9.0, 14.0))

    def test_audiovisual_sync_engine_srt_export_with_transitions(self, tmp_path_factory=None):
        """Verifies AudioVisualSyncEngine.write_srt_subtitles computes correct overlapping subtitles."""
        import tempfile
        scenes = [
            {"duration_seconds": 5.0, "narration_text": "Silas finds a tiny frozen guest."},
            {"duration_seconds": 5.0, "narration_text": "Robin pecks the sweet crumbs eagerly."},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as f:
            out_srt = f.name

        AudioVisualSyncEngine.write_srt_subtitles(scenes, out_srt, transition_type="crossfade", transition_duration=0.5)
        content = pathlib.Path(out_srt).read_text(encoding="utf-8")
        self.assertIn("Silas finds a tiny frozen guest.", content)
        self.assertIn("Robin pecks the sweet crumbs eagerly.", content)
        # Check that subtitle 2 starts at ~00:00:04,700 (4.5s cursor + 0.20s lead-in) instead of 5.2s
        self.assertIn("00:00:04,700", content)
        pathlib.Path(out_srt).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
