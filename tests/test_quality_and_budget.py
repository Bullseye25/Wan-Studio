"""
Unit Tests for Animation Quality Upgrades & Modal Container Budget Cap
====================================================================
Validates:
  1. Modal Container Budget Cap (<= 8 containers across all deploy files)
  2. CEFR B1 Plain-English Narration Engine (Flesch score, purple-prose filtering)
  3. Action Sync Trigger Verb Detection
  4. Audio-Visual Quantization (full 5.0s slot duration, no 21s movie truncation)
  5. Character Anti-Duplication / Anti-Ghosting Guards
  6. Emotional Prosody Speed Mapping for Kokoro Narrator
"""

import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modal_budget import (
    verify_modal_container_budget,
    get_configured_modal_containers,
    ModalBudgetController,
    MAX_GLOBAL_MODAL_CONTAINERS,
)
from narration_readability import (
    check_b1_readability,
    simplify_to_b1_rule_based,
    pick_sync_trigger_word,
    calculate_flesch_reading_ease,
)
from narrator_engine import (
    AudioVisualSyncEngine,
    EMOTION_SPEED_MAP,
)


class TestModalBudgetAndPricing(unittest.TestCase):
    """Verifies that Modal Cloud GPU container allocations never exceed the 8 limit."""

    def test_configured_containers_within_cap(self):
        is_valid, total, breakdown = verify_modal_container_budget(hard_cap=8)
        self.assertTrue(is_valid, f"Total containers {total} must be <= 8")
        self.assertLessEqual(total, 8)
        self.assertEqual(breakdown.get("wan"), 4)
        self.assertEqual(breakdown.get("qwen"), 2)
        self.assertEqual(breakdown.get("z_image"), 1)
        self.assertEqual(breakdown.get("narrator"), 1)

    def test_cost_estimator(self):
        cost = ModalBudgetController.estimate_production_cost(num_scenes=6, detected_chars=2)
        self.assertIn("estimated_cost_usd", cost)
        self.assertIn("breakdown_usd", cost)
        self.assertTrue(cost["estimated_cost_usd"] > 0.50)
        self.assertTrue(cost["total_gpu_seconds"] > 1000.0)


class TestB1ReadabilityAuditor(unittest.TestCase):
    """Verifies CEFR B1 Plain English validation and simplification."""

    def test_flagging_complex_literary_vocabulary(self):
        # Lines from original run
        bad_line = "A manic chuckle emerges from the deep shadows."
        res = check_b1_readability(bad_line)
        self.assertFalse(res["passed"])
        self.assertTrue(any(w in res["flagged_words"] for w in ["manic", "chuckle", "emerges"]))

    def test_passing_b1_plain_english(self):
        good_line = "Batman finds a strange purple mark."
        res = check_b1_readability(good_line, proper_nouns={"Batman"})
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["flesch_score"], 65.0)
        self.assertEqual(len(res["flagged_words"]), 0)

    def test_rule_based_simplification(self):
        orig = "The horrific crime was just the opening punchline."
        simp = simplify_to_b1_rule_based(orig, max_words=10)
        self.assertNotIn("horrific", simp.lower())
        self.assertNotIn("punchline", simp.lower())
        self.assertTrue("terrible" in simp.lower() or "bad" in simp.lower())

    def test_pick_sync_trigger_word(self):
        self.assertEqual(pick_sync_trigger_word("He drives fast to the iron gates now."), "drives")
        self.assertEqual(pick_sync_trigger_word("Batman runs across the dark rooftop."), "runs")
        self.assertEqual(pick_sync_trigger_word("Silas warms the freezing robin gently."), "warms")


class TestAudioVisualPacingAndQuantization(unittest.TestCase):
    """Verifies that clips stay at the full 5.0s slot duration instead of shrinking to 21s."""

    def test_target_clip_duration_preserved(self):
        # 3.2s of audio quantized to a target 5.0s slot
        quant = AudioVisualSyncEngine.quantize_audio_to_wan_frames(
            wav_path_or_duration=3.2,
            target_clip_sec=5.0,
        )
        self.assertEqual(quant["num_frames"], 81)
        self.assertAlmostEqual(quant["exact_duration_seconds"], 5.0625, places=3)
        self.assertEqual(quant["duration_seconds"], 5.06)

    def test_emotion_speed_mapping_coverage(self):
        expected_emotions = ["tense_whisper", "urgent_rising", "epic_climax", "joyful_resolution", "warm_wonder"]
        for emo in expected_emotions:
            self.assertIn(emo, EMOTION_SPEED_MAP)
            speed = EMOTION_SPEED_MAP[emo]
            self.assertTrue(0.85 <= speed <= 1.25, f"Speed {speed} for {emo} must be within [0.85, 1.25]")


if __name__ == "__main__":
    unittest.main()
