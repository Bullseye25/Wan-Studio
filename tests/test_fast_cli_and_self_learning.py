"""
Unit and Integration Tests for Fast Story CLI, Dynamic ETA Calculation,
and Autonomous Self-Learning AI Critic.
"""

import os
import json
import tempfile
import pathlib
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from fast_story_cli import (
    calculate_dynamic_eta,
    format_time,
)
from self_learning_critic import (
    CharacterImageEvaluator,
    KeyframeEvaluator,
    AnimationMovieEvaluator,
    DirectorKnowledgeBase,
    SelfLearningCritic,
)
from studio_pipeline import StudioPipeline


class TestFastCLIEstimation(unittest.TestCase):
    """Verifies Dynamic ETA calculation and time formatting."""

    def test_dynamic_eta_calculation(self):
        # 30-second story (6 scenes)
        eta_30s, breakdown_30s = calculate_dynamic_eta(
            premise="A clockmaker and a little robin befriend each other",
            target_duration=30.0,
            provider="groq",
            fast_mode=True,
        )
        self.assertEqual(breakdown_30s["scenes_count"], 6)
        self.assertEqual(breakdown_30s["detected_chars"], 2)
        self.assertTrue(1500.0 <= eta_30s <= 3500.0)

        # 15-second story (3 scenes) should take approximately half the time
        eta_15s, breakdown_15s = calculate_dynamic_eta(
            premise="A solitary wanderer",
            target_duration=15.0,
            provider="groq",
            fast_mode=True,
        )
        self.assertEqual(breakdown_15s["scenes_count"], 3)
        self.assertEqual(breakdown_15s["detected_chars"], 1)
        self.assertTrue(eta_15s < eta_30s)

        # Provider switch affects screenplay estimation
        eta_ollama, breakdown_ollama = calculate_dynamic_eta(
            premise="Test",
            target_duration=30.0,
            provider="ollama_cloud",
        )
        self.assertTrue(breakdown_ollama["screenplay_sec"] > breakdown_30s["screenplay_sec"])

    def test_format_time(self):
        self.assertEqual(format_time(45), "00m 45s")
        self.assertEqual(format_time(165), "02m 45s")
        self.assertEqual(format_time(3665), "01h 01m 05s")


class TestSelfLearningCritic(unittest.TestCase):
    """Verifies Multi-Tier AI Critic and Continuous Memory Learning Loop."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.storage_dir = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    def test_character_image_evaluator(self):
        # Create a mock clean character image
        clean_img_path = os.path.join(self.storage_dir, "char_clean.png")
        img_arr = np.full((512, 512, 3), 200, dtype=np.uint8)  # uniform studio background
        img_arr[100:400, 150:350] = [50, 80, 120]  # central character silhouette
        Image.fromarray(img_arr).save(clean_img_path)

        char_clean = {
            "name": "Silas",
            "id": "char_1",
            "portrait_path": clean_img_path,
            "z_image_prompt": "Clean studio portrait of Silas, solid grey background",
        }
        res_clean = CharacterImageEvaluator.evaluate(char_clean)
        self.assertGreaterEqual(res_clean["score"], 8.0)
        self.assertTrue(res_clean["is_isolated_backdrop"])

        # Cluttered image (high border variance)
        cluttered_path = os.path.join(self.storage_dir, "char_clutter.png")
        cluttered_arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(cluttered_arr).save(cluttered_path)

        char_cluttered = {
            "name": "Noisy",
            "id": "char_2",
            "portrait_path": cluttered_path,
            "z_image_prompt": "Noisy portrait with busy city background",
        }
        res_cluttered = CharacterImageEvaluator.evaluate(char_cluttered)
        self.assertLess(res_cluttered["score"], 8.0)
        self.assertFalse(res_cluttered["is_isolated_backdrop"])
        self.assertTrue(len(res_cluttered["recommendations"]) > 0)

    def test_keyframe_evaluator(self):
        # 1. Optimal keyframes
        optimal_scene = {
            "scene_number": 1,
            "title": "Meeting",
            "pose_delta": 32.5,
            "flf2v_compat": {"bg_ssim": 0.88, "pose_delta": 32.5},
            "state_ledger": {"active_props": ["lantern"]},
            "qwen_scene_prompt": "Hero holds a glowing lantern and enters the cave",
            "qwen_end_scene_prompt": "Hero sets lantern on rock and shields face",
        }
        res_opt = KeyframeEvaluator.evaluate(optimal_scene)
        self.assertGreaterEqual(res_opt["score"], 8.5)
        self.assertEqual(res_opt["status"], "EXCELLENT")

        # 2. Too static scene (pose delta < 18.0)
        static_scene = {
            "scene_number": 2,
            "title": "Frozen",
            "pose_delta": 8.0,
            "flf2v_compat": {"bg_ssim": 0.95, "pose_delta": 8.0},
            "state_ledger": {"active_props": []},
            "qwen_scene_prompt": "Hero stands looking at stone",
            "qwen_end_scene_prompt": "Hero stands looking at stone still",
        }
        res_static = KeyframeEvaluator.evaluate(static_scene)
        self.assertLessEqual(res_static["score"], 7.0)
        self.assertIn("too low", " ".join(res_static["findings"]))

        # 3. Prop vanished in end frame
        vanished_prop_scene = {
            "scene_number": 3,
            "title": "Missing Sword",
            "pose_delta": 28.0,
            "flf2v_compat": {"bg_ssim": 0.85, "pose_delta": 28.0},
            "state_ledger": {"active_props": ["Golden Sword"]},
            "qwen_scene_prompt": "Hero wields the Golden Sword heroically",
            "qwen_end_scene_prompt": "Hero leaps backward empty-handed",
        }
        res_prop = KeyframeEvaluator.evaluate(vanished_prop_scene)
        self.assertIn("Golden Sword", " ".join(res_prop["findings"]))

    def test_director_knowledge_base_persistence_and_retrieval(self):
        kb = DirectorKnowledgeBase(storage_dir=self.storage_dir)

        directives = [
            {
                "category": "Keyframe Dynamics & Pose Delta",
                "severity": "HIGH",
                "mistake_observed": "Scene 2 character remained static with pose delta 6.5.",
                "directive_to_enforce": "Ensure extreme crouching stance in end prompt.",
                "tags": ["robot", "cave", "scene_2"],
            },
            {
                "category": "Character Isolation & Silhouette",
                "severity": "MEDIUM",
                "mistake_observed": "Background clutter detected on Silas portrait.",
                "directive_to_enforce": "Explicitly specify neutral light-grey studio backdrop.",
                "tags": ["clockmaker", "workshop", "portrait"],
            },
        ]

        count = kb.record_learned_directives(directives, story_title="The Clockwork Soul", premise="robot in cave")
        self.assertEqual(count, 2)
        self.assertTrue(kb.rules_file.exists())

        # Query matching robot/cave
        retrieved = kb.retrieve_relevant_directives("A futuristic robot explores an ancient cave", top_k=2)
        self.assertTrue(len(retrieved) >= 1)
        self.assertEqual(retrieved[0]["category"], "Keyframe Dynamics & Pose Delta")

        # Format prompt
        prompt_block = kb.format_directives_prompt(retrieved)
        self.assertIn("LEARNED PRODUCTION DIRECTIVES", prompt_block)
        self.assertIn("Ensure extreme crouching stance", prompt_block)

    def test_self_learning_critic_end_to_end(self):
        critic = SelfLearningCritic(storage_dir=self.storage_dir)

        mock_story = {
            "title": "Little Robot",
            "theme": "A robot finds a fragile flower",
            "final_movie_path": os.path.join(self.storage_dir, "dummy_movie.mp4"),
            "transition_type": "narrative",
            "characters": [
                {
                    "name": "Robo",
                    "id": "char_1",
                    "z_image_prompt": "Solo studio portrait of little robot, solid plain grey studio background",
                }
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "The Meadow",
                    "pose_delta": 12.0,  # Below 18 threshold -> will trigger directive learning
                    "flf2v_compat": {"bg_ssim": 0.85, "pose_delta": 12.0},
                    "state_ledger": {"active_props": ["watering can"]},
                    "qwen_scene_prompt": "Robo stands with watering can",
                    "qwen_end_scene_prompt": "Robo stands still",
                }
            ],
        }

        # Create dummy movie file so movie evaluator passes
        with open(mock_story["final_movie_path"], "wb") as f:
            f.write(b"dummy_mp4_bytes")

        report = critic.evaluate_production_run(mock_story)
        self.assertIn("overall_studio_score", report)
        self.assertEqual(len(report["character_evaluations"]), 1)
        self.assertEqual(len(report["scene_evaluations"]), 1)
        # Verify that sub-optimal scene pose delta triggered a new learned directive
        self.assertGreater(report["new_directives_learned"], 0)


class TestStudioPipelineLearnedDirectiveInjection(unittest.TestCase):
    """Verifies that learned directives from memory are injected into screenplay generation."""

    def test_pipeline_injects_learned_directives_into_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Seed knowledge base with a directive
            kb = DirectorKnowledgeBase(storage_dir=tmp_dir)
            kb.record_learned_directives(
                [
                    {
                        "category": "Prop Permanence",
                        "severity": "HIGH",
                        "mistake_observed": "Character lantern disappeared mid-scene.",
                        "directive_to_enforce": "Lock lantern in hero hands in both start and end frames.",
                        "tags": ["cave", "lantern", "adventure"],
                    }
                ],
                story_title="Prior Run",
                premise="cave lantern adventure",
            )

            pipeline = StudioPipeline(output_base_dir=tmp_dir, provider="groq", api_key="gsk_test")

            captured_prompts = []

            def mock_chat_completion(messages, **kwargs):
                captured_prompts.append(messages[0]["content"])
                return json.dumps({
                    "title": "Cave Adventure",
                    "characters": [{"id": "c1", "name": "Explorer"}],
                    "scenes": [{"scene_number": 1, "title": "Entry"}],
                })

            pipeline._chat_completion = mock_chat_completion

            pipeline.generate_story_and_characters(
                prompt="An adventure inside a dark cave with a lantern",
                num_scenes=2,
                target_duration=10.0,
            )

            self.assertTrue(len(captured_prompts) > 0)
            system_prompt = captured_prompts[0]
            # Verify that the learned directive was inlined into the AI Director system prompt!
            self.assertIn("LEARNED PRODUCTION DIRECTIVES", system_prompt)
            self.assertIn("Lock lantern in hero hands", system_prompt)


if __name__ == "__main__":
    unittest.main()
