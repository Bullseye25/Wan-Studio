import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend


class TestHermesPrompting(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    def test_empty_prompt_raises_error(self):
        """refine_prompt_hermes should raise ValueError when raw_prompt is empty."""
        with self.assertRaises(ValueError):
            backend.refine_prompt_hermes("")

        with self.assertRaises(ValueError):
            backend.refine_prompt_hermes("   ")

    def test_fallback_prompt_generation_styles(self):
        """Fallback engine should produce rich prompts for all standard style presets."""
        presets = [
            "Cinematic Film (35mm)",
            "Hyperrealistic 4K",
            "Cyberpunk & Sci-Fi",
            "Anime / Studio Ghibli",
            "Dark Noir & Atmospheric",
            "Fantasy & Mythical",
            "Drone Aerial Nature",
        ]
        for preset in presets:
            res = backend.build_local_fallback_prompt(
                raw_prompt="A sports car speeding along the highway",
                style_preset=preset,
                motion_dynamics="Smooth & Flowing Motion",
                camera_movement="Slow Push-In Tracking Shot"
            )
            self.assertIn("refined_prompt", res)
            self.assertIn("negative_prompt", res)
            self.assertIn(preset, res["style_applied"])
            self.assertTrue(res.get("fallback", False))
            self.assertIn("sports car speeding", res["refined_prompt"])

    @patch("modal.Cls.from_name")
    def test_hermes_remote_call_mock(self, mock_from_name):
        """Verify refine_prompt_hermes correctly invokes Modal HermesPromptRefiner."""
        mock_instance = MagicMock()
        mock_instance.refine_prompt.remote.return_value = {
            "refined_prompt": "A majestic cybernetic samurai standing under cherry blossoms, 35mm anamorphic lens, 4k",
            "negative_prompt": "blurry, low quality",
            "style_applied": "Cinematic Film (35mm)",
            "elapsed_seconds": 1.25
        }
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        result = backend.refine_prompt_hermes(
            raw_prompt="cyber samurai cherry blossom",
            style_preset="Cinematic Film (35mm)",
            motion_dynamics="Smooth & Flowing Motion",
            camera_movement="Auto Best Fit",
            temperature=0.7
        )

        self.assertIn("cybernetic samurai", result["refined_prompt"])
        self.assertEqual(result["style_applied"], "Cinematic Film (35mm)")
        self.assertEqual(result["elapsed_seconds"], 1.25)
        mock_instance.refine_prompt.remote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
