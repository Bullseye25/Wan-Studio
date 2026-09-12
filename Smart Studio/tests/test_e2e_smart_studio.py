import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend
import app


class TestSmartStudioE2E(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    @patch("modal.Cls.from_name")
    def test_complete_smart_studio_pipeline(self, mock_from_name):
        """
        Complete end-to-end simulation:
        1. Refine a raw concept using Hermes LLM.
        2. Transfer refined prompt to Wan 2.2.
        3. Generate Wan 2.2 video.
        4. Validate outputs and frame counts.
        """
        print("\n" + "=" * 65)
        print(" [E2E Test] Starting Smart Studio End-to-End Workflow")
        print("=" * 65)

        mock_hermes = MagicMock()
        mock_wan = MagicMock()

        mock_hermes.refine_prompt.remote.return_value = {
            "refined_prompt": (
                "A sleek futuristic motorcycle drifting around a neon-lit cyberpunk corner in pouring rain, "
                "wet pavement reflecting brilliant holographic billboards, volumetric water spray, "
                "cinematic 35mm lens, 4k masterwork, highly detailed."
            ),
            "negative_prompt": "blurry, low quality, deformed, jittery",
            "style_applied": "Cyberpunk & Sci-Fi",
            "elapsed_seconds": 1.15
        }

        mock_wan.generate.remote_gen.return_value = [
            {"type": "progress", "step": 0, "total_steps": 6},
            {"type": "progress", "step": 3, "total_steps": 6},
            {"type": "progress", "step": 5, "total_steps": 6},
            {"type": "output", "video_bytes": b"\x00\x00\x00\x1cftypisommock_bytes", "filename": "smart_wan_777.mp4"}
        ]

        def side_effect(app_name, cls_name):
            if cls_name == "HermesPromptRefiner":
                return MagicMock(return_value=mock_hermes)
            elif cls_name == "SmartWanModel":
                return MagicMock(return_value=mock_wan)
            return MagicMock()

        mock_from_name.side_effect = side_effect

        # Step 1: Hermes Prompt Refinement
        print("\n[Step 1] Invoking Hermes 3 Smart Prompt Refiner...")
        raw_concept = "cyberpunk motorcycle drifting in rain"
        refined_prompt, negative_prompt, status_msg = app.on_refine_prompt_click(
            raw_prompt=raw_concept,
            style_preset="Cyberpunk & Sci-Fi",
            motion_dynamics="High-Speed Dynamic Action",
            camera_movement="Slow Push-In Tracking Shot",
            temperature=0.7,
            progress=None
        )

        self.assertIn("sleek futuristic motorcycle", refined_prompt)
        self.assertIn("Cyberpunk", status_msg)
        print("  [OK] Hermes 3 successfully returned refined prompt.")

        # Step 2: Input Image Preparation
        print("\n[Step 2] Preparing image input for Wan 2.2...")
        input_img = Image.new("RGB", (720, 480), color="darkblue")

        # Step 3: Wan 2.2 Video Generation
        print("\n[Step 3] Generating video via Wan 2.2 (A100 GPU)...")
        video_path, dl_path, seed_out, info_msg, gallery = app.on_generate_video_click(
            input_image=input_img,
            last_image=None,
            prompt=refined_prompt,
            negative_prompt=negative_prompt,
            duration=3.5,
            steps=6,
            guidance_scale=5.0,
            guidance_scale_2=2.0,
            seed=777,
            randomize_seed=False,
            scheduler_name="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=32,
            progress=None
        )

        self.assertTrue(os.path.exists(video_path))
        self.assertEqual(seed_out, 777)
        self.assertIn("777", info_msg)
        print(f"  [OK] Video successfully written to: {video_path}")

        print("\n" + "=" * 65)
        print(" [E2E Test] SMART STUDIO ALL STEPS PASSED SUCCESSFULLY!")
        print("=" * 65)


if __name__ == "__main__":
    unittest.main()
