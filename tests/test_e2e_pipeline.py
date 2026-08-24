import io
import os
import sys
import pathlib
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

import app
import backend


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-End Workflow Test:
    1. Take a picture (from workspace 'test_bus.png' or freshly generated image).
    2. Modify it via Step 1 (Qwen Image Edit Rapid AIO generation/editing flow).
    3. Feed modified image into Step 2 (Wan 2.2 Video Generator).
    4. Animate with Wan 2.2 I2V and verify output video generation and frame extraction.
    """

    def setUp(self):
        backend.reset_cached_models()
        self.workspace_root = pathlib.Path(__file__).parent.parent
        self.test_image_path = self.workspace_root / "test_bus.png"

        # Ensure a test image exists
        if not self.test_image_path.exists():
            img = Image.new("RGB", (640, 480), color=(70, 130, 180))
            draw = ImageDraw.Draw(img)
            draw.rectangle([100, 150, 540, 350], fill=(220, 20, 60))
            img.save(self.test_image_path)

    def tearDown(self):
        backend.reset_cached_models()

    @patch("modal.Cls.from_name")
    def test_full_take_picture_modify_animate_workflow(self, mock_modal_cls):
        print("\n" + "=" * 70)
        print(" [E2E Test] Starting End-to-End Picture -> Modify -> Animate Test")
        print("=" * 70)

        # -------------------------------------------------------------
        # STEP 1: Take / Load Picture
        # -------------------------------------------------------------
        print("\n[Step A] Taking / Loading initial picture...")
        self.assertTrue(self.test_image_path.exists(), "Test picture must exist")
        with Image.open(self.test_image_path) as img:
            initial_pic = img.convert("RGB")
        orig_w, orig_h = initial_pic.size
        print(f"  [OK] Initial picture loaded successfully: {orig_w}x{orig_h} (RGB)")
        self.assertGreater(orig_w, 0)
        self.assertGreater(orig_h, 0)

        # -------------------------------------------------------------
        # STEP 2: Modify Picture via Step 1 (Qwen Image Edit)
        # -------------------------------------------------------------
        print("\n[Step B] Modifying picture via Step 1 (Qwen Image Edit Studio)...")
        
        # Prepare a modified image that the mock Modal backend will return
        modified_synthetic = initial_pic.copy().resize((1024, 768))
        draw = ImageDraw.Draw(modified_synthetic)
        draw.text((50, 50), "QWEN MODIFIED", fill=(255, 255, 0))
        
        mod_bytes_io = io.BytesIO()
        modified_synthetic.save(mod_bytes_io, format="JPEG", quality=95)
        mock_qwen_bytes = mod_bytes_io.getvalue()

        mock_qwen_instance = MagicMock()
        mock_qwen_instance.generate.remote.return_value = mock_qwen_bytes

        mock_wan_instance = MagicMock()

        # Configure Modal Cls dispatcher
        def modal_cls_router(app_name, cls_name):
            if app_name == "qwen-image-backend":
                return MagicMock(return_value=mock_qwen_instance)
            elif app_name == "wan-backend":
                return MagicMock(return_value=mock_wan_instance)
            raise ValueError(f"Unknown app: {app_name}")

        mock_modal_cls.side_effect = modal_cls_router

        # Run Step 1 generate/modify
        edit_prompt = "transform the bus into a sleek cyber-bus at sunset, neon reflections"
        modified_pil, used_seed = app.run_qwen_generate(
            base_image=initial_pic,
            ref_image_1=None,
            ref_image_2=None,
            prompt=edit_prompt,
            negative_prompt="blurry, distorted",
            seed=999,
            randomize_seed=False,
            steps=20,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            custom_width=1024,
            custom_height=768,
        )

        self.assertIsInstance(modified_pil, Image.Image)
        self.assertEqual(used_seed, 999)
        print(f"  [OK] Picture modified successfully: {modified_pil.size} (Seed: {used_seed})")

        # -------------------------------------------------------------
        # STEP 3: Route Modified Image to Video Generator (Step 2)
        # -------------------------------------------------------------
        print("\n[Step C] Routing modified image to Step 2 Video Generator...")
        video_input_img = app.use_generated_as_video_input(modified_pil)
        self.assertEqual(video_input_img, modified_pil)
        print("  [OK] Image successfully passed into Step 2 video input component.")

        # -------------------------------------------------------------
        # STEP 4: Animate via Step 2 (Wan 2.2 I2V Animation)
        # -------------------------------------------------------------
        print("\n[Step D] Animating modified picture with Wan 2.2...")

        # Setup mock video streaming response
        fake_video_content = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42fake_mp4_stream"
        mock_wan_instance.generate.remote_gen.return_value = [
            {"type": "progress", "step": 1, "total_steps": 6},
            {"type": "progress", "step": 3, "total_steps": 6},
            {"type": "progress", "step": 6, "total_steps": 6},
            {"type": "status", "message": "Processing frames (RIFE Multiplier: 1x)..."},
            {"type": "output", "filename": "e2e_animated_bus.mp4", "video_bytes": fake_video_content},
        ]

        motion_prompt = "cinematic camera tracking shot, bus speeding along coastal highway, glowing lights"
        video_comp, video_path, gen_seed = app.generate_video(
            input_image=video_input_img,
            last_image=None,
            prompt=motion_prompt,
            steps=6,
            duration_seconds=3.5,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            seed=777,
            randomize_seed=False,
            quality=6,
            scheduler="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=16,
            safe_mode=True,
        )

        self.assertIsNotNone(video_path)
        self.assertTrue(os.path.exists(video_path), f"Video file {video_path} should exist on disk")
        self.assertEqual(gen_seed, 777)
        print(f"  [OK] Video successfully generated and written to: {video_path}")
        print(f"  [OK] File size: {os.path.getsize(video_path)} bytes")

        # -------------------------------------------------------------
        # STEP 5: Verify Video Dimensional & Frame Constraints
        # -------------------------------------------------------------
        print("\n[Step E] Verifying internal dimensions and constraints...")
        resized_for_model = app.resize_image(modified_pil)
        w, h = resized_for_model.size
        print(f"  [OK] Wan input resolution: {w}x{h}")
        self.assertEqual(w % app.MULTIPLE_OF, 0)
        self.assertEqual(h % app.MULTIPLE_OF, 0)
        self.assertLessEqual(w, app.MAX_DIM)
        self.assertGreaterEqual(w, app.MIN_DIM)
        self.assertLessEqual(h, app.MAX_DIM)
        self.assertGreaterEqual(h, app.MIN_DIM)

        expected_frames = app.get_num_frames(3.5)
        print(f"  [OK] Wan frame calculation for 3.5s: {expected_frames} frames")
        self.assertEqual(expected_frames, 57)

        print("\n" + "=" * 70)
        print(" [E2E Test] ALL WORKFLOW STEPS PASSED SUCCESSFULLY!")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    unittest.main()
