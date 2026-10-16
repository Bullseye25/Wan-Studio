"""
Comprehensive Test Suite for the Logical Short Films AI Director & Training System.
Tests:
  1. Causal narrative beat mappings and connectors (Therefore / But).
  2. Gold-standard causal blueprints and State Ledgers.
  3. StoryMemoryBank save, retrieval, few-shot formatting, and DPO pair recording.
  4. State Ledger normalization, prop inheritance, and cross-scene continuity bridging.
  5. Multi-Agent Story Doctor logic audit, scoring, and auto-repair.
  6. Wan prompt compilation with weapon/prop retention locking.
  7. Gradio UI handlers: studio_run_logic_audit and studio_save_memory.
"""

import os
import json
import tempfile
import pathlib
import unittest
from unittest.mock import patch, MagicMock

import cinematography_vault
from cinematography_vault import (
    NARRATIVE_BEATS_BY_COUNT,
    CAUSAL_NARRATIVE_BLUEPRINTS,
    StoryMemoryBank,
)
import studio_pipeline
from studio_pipeline import StudioPipeline
import app


class TestLogicalStoryDirector(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory_file = pathlib.Path(self.temp_dir.name) / "test_memory.jsonl"
        self.dpo_file = pathlib.Path(self.temp_dir.name) / "test_dpo.jsonl"
        self.memory_bank = StoryMemoryBank(
            memory_dir=self.temp_dir.name,
            memory_filename="test_memory.jsonl",
            dpo_filename="test_dpo.jsonl",
        )
        self.pipeline = StudioPipeline(
            api_key="test_dummy_key",
            model="gemma4:31b",
            output_base_dir=self.temp_dir.name,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # ── 1. Causal Beat Mappings ──────────────────────────────────────────────

    def test_narrative_beats_by_count(self):
        """Verify standard scene counts (2, 3, 4, 5, 6) have valid causal connectors."""
        for count in [2, 3, 4, 5, 6]:
            self.assertIn(count, NARRATIVE_BEATS_BY_COUNT)
            beats = NARRATIVE_BEATS_BY_COUNT[count]
            self.assertEqual(len(beats), count)
            # Opening scene should be SETUP
            self.assertEqual(beats[0]["connector"], "SETUP")
            # Subsequent scenes must use THEREFORE or BUT
            for b in beats[1:]:
                self.assertIn(b["connector"], ["THEREFORE", "BUT"])

    # ── 2. Gold Blueprints ───────────────────────────────────────────────────

    def test_causal_narrative_blueprints_structure(self):
        """Curated blueprints must contain state ledgers with all 5 mandatory keys."""
        self.assertGreaterEqual(len(CAUSAL_NARRATIVE_BLUEPRINTS), 2)
        for bp in CAUSAL_NARRATIVE_BLUEPRINTS:
            self.assertIn("title", bp)
            self.assertIn("premise", bp)
            self.assertIn("scenes", bp)
            for sc in bp["scenes"]:
                self.assertIn("state_ledger", sc)
                ledger = sc["state_ledger"]
                for key in ["entry_state", "active_props", "spatial_zone", "atomic_action_5s", "exit_state"]:
                    self.assertIn(key, ledger, f"Missing {key} in blueprint {bp['title']}")

    # ── 3. Story Memory Bank ─────────────────────────────────────────────────

    def test_story_memory_bank_save_and_retrieve(self):
        """Saving an approved story makes it retrievable by semantic keywords."""
        sample_story = {
            "title": "Samurai Night",
            "theme": "Honor and courage in a misty bamboo forest",
            "characters": [{"id": "char_1", "name": "Kenshin", "signature_details": "Nodachi blade"}],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "Ambush",
                    "scenario": "Kenshin draws his blade in the bamboo mist",
                    "causal_connector": "SETUP",
                    "state_ledger": {
                        "entry_state": "Standing still",
                        "active_props": ["Nodachi blade"],
                        "spatial_zone": "Left facing right",
                        "atomic_action_5s": "Draws blade and assumes guard",
                        "exit_state": "Low guard",
                    },
                }
            ],
        }

        success = self.memory_bank.save_exemplar(
            premise="samurai in bamboo forest",
            story_data=sample_story,
            rating=10.0,
            user_edited=True,
        )
        self.assertTrue(success)
        self.assertTrue(self.memory_file.exists())

        # Retrieve with relevant keyword
        retrieved = self.memory_bank.retrieve_exemplars("samurai fight in bamboo woods", top_k=2)
        self.assertGreater(len(retrieved), 0)
        titles = [r.get("title") for r in retrieved]
        self.assertIn("Samurai Night", titles)

    def test_story_memory_bank_dpo_pair_recording(self):
        """DPO preference pair is written cleanly to disk."""
        rejected = {"title": "Flawed", "scenes": [{"scene_number": 1, "scenario": "and then he did 5 things"}]}
        chosen = {"title": "Polished", "scenes": [{"scene_number": 1, "scenario": "therefore he parried"}]}
        ok = self.memory_bank.record_dpo_pair(
            premise="sword clash",
            rejected_story=rejected,
            chosen_story=chosen,
            critique_feedback="Replaced multi-action cramming with 1 atomic 5s parry",
        )
        self.assertTrue(ok)
        self.assertTrue(self.dpo_file.exists())
        with open(self.dpo_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["premise"], "sword clash")
            self.assertEqual(data["chosen"]["title"], "Polished")

    # ── 4. State Ledger Normalization & Continuity Bridging ───────────────────

    def test_normalize_story_populates_state_ledgers_and_connectors(self):
        """Normalizing raw LLM output auto-populates State Ledgers and causal connectors."""
        raw = {
            "title": "Cyberpunk Heist",
            "characters": [{"id": "char_1", "name": "Vex", "signature_details": "plasma cutter"}],
            "scenes": [
                {"title": "Vault Entry", "scenario": "Vex ignites the plasma cutter"},
                {"title": "Vault Breach", "scenario": "Alarm blares and laser grid activates"},
            ],
        }
        norm = StudioPipeline.normalize_story(
            raw, prompt="heist", num_scenes=2, target_duration=10.0, art_style="anime"
        )
        s1 = norm["scenes"][0]
        s2 = norm["scenes"][1]

        # Check connectors
        self.assertEqual(s1["causal_connector"], "SETUP")
        self.assertEqual(s2["causal_connector"], "THEREFORE")

        # Check State Ledgers
        self.assertIn("state_ledger", s1)
        self.assertIn("state_ledger", s2)

        # Prop inheritance: plasma cutter carried to s1 and inherited in s2
        self.assertIn("plasma cutter", s1["state_ledger"]["active_props"])
        self.assertIn("plasma cutter", s2["state_ledger"]["active_props"])

        # Entry-Exit Continuity: s2 entry state references s1 exit state
        self.assertIn("Continuing from previous shot", s2["state_ledger"]["entry_state"])

    # ── 5. Story Doctor Logic Audit & Auto-Repair ─────────────────────────────

    def test_audit_and_repair_story_logic_heuristic(self):
        """Audit accurately detects flaws and scores healthy causal stories >= 8.0."""
        # Story with lazy 'AND THEN' and identical start/end prompts
        flawed_story = {
            "title": "Broken Flow",
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "Scene 1",
                    "causal_connector": "AND THEN something happened",
                    "scenario": "Hero runs and jumps and finds gold and sleeps and wakes up",
                    "qwen_scene_prompt": "Hero standing in hallway",
                    "qwen_end_scene_prompt": "Hero standing in hallway",  # identical = frozen video
                }
            ],
        }
        report = self.pipeline.audit_and_repair_story_logic(flawed_story)
        self.assertFalse(report["passed"])
        self.assertLess(report["overall_score"], 8.0)
        self.assertGreater(len(report["issues"]), 0)

        # Well-formed causal story
        clean_story = {
            "title": "Clean Story",
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "Opening",
                    "causal_connector": "SETUP",
                    "scenario": "Hero raises shield against incoming arrows",
                    "qwen_scene_prompt": "Hero standing with shield at hip",
                    "qwen_end_scene_prompt": "Hero crouching behind raised steel shield deflection sparks",
                },
                {
                    "scene_number": 2,
                    "title": "Counter",
                    "causal_connector": "THEREFORE — Hero thrusts forward with spear",
                    "scenario": "Hero launches from behind shield into forward thrust",
                    "qwen_scene_prompt": "Hero bursting forward from behind shield",
                    "qwen_end_scene_prompt": "Hero spear extended forward in follow-through pose",
                },
            ],
        }
        report_clean = self.pipeline.audit_and_repair_story_logic(clean_story)
        self.assertTrue(report_clean["passed"])
        self.assertGreaterEqual(report_clean["overall_score"], 8.0)

    # ── 6. Director Plan Weapon Retention Locking ────────────────────────────

    def test_compile_director_plan_injects_weapon_lock(self):
        """compile_director_plan locks active props inside identity anchors."""
        scene = {
            "scenario": "Warrior attacks with flaming longsword",
            "state_ledger": {"active_props": ["flaming longsword", "kite shield"]},
            "duration_seconds": 5.0,
        }
        cast = [{"name": "Aria"}]
        plan = StudioPipeline.compile_director_plan(raw={}, scene=scene, cast=cast)
        wan_prompt = plan["wan_prompt"]
        self.assertIn("weapon and prop lock", wan_prompt)
        self.assertIn("flaming longsword", wan_prompt)

    # ── 7. UI Handlers ───────────────────────────────────────────────────────

    def test_app_studio_audit_and_memory_handlers(self):
        """app.py logic audit and memory save helpers function cleanly."""
        state = {
            "title": "Skyward Pilot",
            "theme": "Flight",
            "scenes": [
                {
                    "scene_number": 1,
                    "causal_connector": "SETUP",
                    "scenario": "Pilot primes engine",
                    "qwen_scene_prompt": "Pilot in cockpit hands on throttle",
                    "qwen_end_scene_prompt": "Throttle pulled back engine humming",
                }
            ],
        }
        # Audit handler
        _, audit_status, _ = app.studio_run_logic_audit(state, "dummy_key", "nemotron-3-super")
        self.assertIn("Story Doctor Causal Audit Report", audit_status)

        # Memory handler
        mem_msg = app.studio_save_memory(state)
        self.assertIn("saved to AI Training Memory", mem_msg)


if __name__ == "__main__":
    unittest.main()
