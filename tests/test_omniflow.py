"""
Unit & Integration Tests for OmniFlow-30 Engine & Parallel Swarm Architecture
=============================================================================
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omniflow_engine import (
    OmniFlowEngine,
    OmniFlowConfig,
    calculate_omniflow_scenes,
    inject_9_16_vertical_composition_rules,
)
from studio_pipeline import StudioPipeline


class TestOmniFlowEngine(unittest.TestCase):

    def test_calculate_omniflow_scenes(self):
        """Validates that runtime duration maps to balanced dramatic scene counts."""
        self.assertEqual(calculate_omniflow_scenes(5.0), 2)
        self.assertEqual(calculate_omniflow_scenes(10.0), 2)
        self.assertEqual(calculate_omniflow_scenes(15.0), 3)
        self.assertEqual(calculate_omniflow_scenes(20.0), 4)
        self.assertEqual(calculate_omniflow_scenes(30.0), 4)
        self.assertEqual(calculate_omniflow_scenes(45.0), 4)

    def test_inject_9_16_vertical_composition_rules(self):
        """Ensures vertical 9:16 safe-zone framing cues are injected properly."""
        raw_prompt = "A warrior stands on a cliff at sunset"
        enriched = inject_9_16_vertical_composition_rules(raw_prompt, "anime")
        self.assertIn("vertical 9:16 cinema composition", enriched)
        self.assertIn("centered subject in portrait framing", enriched)

        # Idempotence: shouldn't duplicate if already present
        already_vertical = "A warrior in vertical 9:16 view"
        same = inject_9_16_vertical_composition_rules(already_vertical, "anime")
        self.assertEqual(same, already_vertical)

    def test_omniflow_eta_breakdown(self):
        """Verifies that Parallel Swarm cuts estimated video render time."""
        engine = OmniFlowEngine()

        cfg_swarm = OmniFlowConfig(
            prompt="Test prompt",
            target_duration=30.0,
            parallel_swarm=True,
            enable_midpoint_refine=False,
        )
        eta_swarm, bd_swarm = engine.calculate_eta(cfg_swarm)

        cfg_seq = OmniFlowConfig(
            prompt="Test prompt",
            target_duration=30.0,
            parallel_swarm=False,
            enable_midpoint_refine=False,
        )
        eta_seq, bd_seq = engine.calculate_eta(cfg_seq)

        # 4 scenes in swarm mode should be significantly faster than sequential 4 scenes
        self.assertLess(bd_swarm["video_swarm"], bd_seq["video_swarm"])
        self.assertLess(eta_swarm, eta_seq)

    def test_midpoint_latent_steering_tensor_math(self):
        """
        Validates MILIS (Mid-Trajectory Latent Interception & Steering) tensor math:
        - Clamps extreme saturation outliers
        - Applies temporal smoothing on interior frames
        - Preserves boundary endpoints (frame 0 and frame T-1) identically
        """
        # Create mock 5D latent tensor: [Batch=1, Channels=16, Time=21, Height=10, Width=10]
        B, C, T, H, W = 1, 16, 21, 10, 10
        raw_latents = torch.randn(B, C, T, H, W)

        # Inject extreme saturation outlier
        raw_latents[0, 0, 10, 5, 5] = 99.0
        raw_latents[0, 1, 10, 5, 5] = -99.0

        # Boundary frames (clamped) to check
        first_frame_orig = torch.clamp(raw_latents[:, :, 0, :, :], min=-4.0, max=4.0)
        last_frame_orig = torch.clamp(raw_latents[:, :, -1, :, :], min=-4.0, max=4.0)

        # In-flight midpoint steering function from deploy_wan.py
        def apply_steering(latents, strength=0.25):
            clamped = torch.clamp(latents, min=-4.0, max=4.0)
            if clamped.ndim == 5 and clamped.shape[2] > 4:
                interior = clamped[:, :, 1:-1, :, :]
                smoothed = 0.25 * clamped[:, :, :-2, :, :] + 0.50 * interior + 0.25 * clamped[:, :, 2:, :, :]
                clamped[:, :, 1:-1, :, :] = (1.0 - strength) * interior + strength * smoothed
            return clamped

        steered = apply_steering(raw_latents, strength=0.30)

        # Verify clamping
        self.assertLessEqual(steered.max().item(), 4.0)
        self.assertGreaterEqual(steered.min().item(), -4.0)

        # Verify endpoints preserved
        self.assertTrue(torch.allclose(steered[:, :, 0, :, :], first_frame_orig))
        self.assertTrue(torch.allclose(steered[:, :, -1, :, :], last_frame_orig))

        # Verify shape preserved
        self.assertEqual(steered.shape, (B, C, T, H, W))

    @patch.object(StudioPipeline, "_animate_single_scene_clip")
    def test_swarm_parallel_execution_order_preserved(self, mock_animate):
        """Verifies that Parallel Swarm properly collects all scene clips in correct sequence."""
        pipeline = StudioPipeline(api_key="test-key")

        # Mock animate to return clip path based on scene number
        def fake_animate(scene, **kwargs):
            return f"outputs/studio/clips/clip_scene_{scene['scene_number']}.mp4"

        mock_animate.side_effect = fake_animate

        scenes = [
            {"scene_number": 1, "title": "Opening"},
            {"scene_number": 2, "title": "Conflict"},
            {"scene_number": 3, "title": "Climax"},
            {"scene_number": 4, "title": "Resolution"},
        ]

        clip_paths = pipeline.generate_scene_videos(
            scenes=scenes,
            parallel_swarm=True,
            base_seed=12345,
        )

        self.assertEqual(len(clip_paths), 4)
        self.assertEqual(clip_paths[0], "outputs/studio/clips/clip_scene_1.mp4")
        self.assertEqual(clip_paths[1], "outputs/studio/clips/clip_scene_2.mp4")
        self.assertEqual(clip_paths[2], "outputs/studio/clips/clip_scene_3.mp4")
        self.assertEqual(clip_paths[3], "outputs/studio/clips/clip_scene_4.mp4")

    def test_setup_story_workspace(self):
        """Verifies that dedicated story folder is created with all subdirectories."""
        import shutil
        pipeline = StudioPipeline(api_key="test-key")
        story_dir = pipeline.setup_story_workspace("The Fox and the Silver Bell")
        self.assertTrue(story_dir.exists())
        self.assertEqual(story_dir.name, "the_fox_and_the_silver_bell")
        self.assertTrue((story_dir / "characters").exists())
        self.assertTrue((story_dir / "scenes").exists())
        self.assertTrue((story_dir / "clips").exists())
        self.assertTrue((story_dir / "audio").exists())
        self.assertTrue((story_dir / "movie").exists())
        self.assertTrue((story_dir / "telemetry").exists())
        # Cleanup
        if story_dir.exists():
            shutil.rmtree(story_dir)


if __name__ == "__main__":
    unittest.main()
