import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend
import app


class TestSmartStudioContracts(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    def test_image_resize_dimensions(self):
        """resize_image should return images with dimensions divisible by 16 within [480, 832]."""
        # Landscape
        img_wide = Image.new("RGB", (1920, 1080), color="blue")
        resized_wide = app.resize_image(img_wide)
        w, h = resized_wide.size
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertLessEqual(w, 832)
        self.assertGreaterEqual(w, 480)

        # Portrait
        img_tall = Image.new("RGB", (1080, 1920), color="green")
        resized_tall = app.resize_image(img_tall)
        w2, h2 = resized_tall.size
        self.assertEqual(w2 % 16, 0)
        self.assertEqual(h2 % 16, 0)
        self.assertLessEqual(h2, 832)
        self.assertGreaterEqual(h2, 480)

    def test_resize_and_crop_to_match(self):
        """resize_and_crop_to_match should match target image size exactly."""
        ref = Image.new("RGB", (832, 480), color="black")
        target = Image.new("RGB", (1000, 1000), color="red")
        matched = app.resize_and_crop_to_match(target, ref)
        self.assertEqual(matched.size, (832, 480))

    def test_get_num_frames(self):
        """Frame counts must follow Wan's 1 + 16*n formula."""
        self.assertEqual(app.get_num_frames(1.0), 17)   # 1 + 16*1
        self.assertEqual(app.get_num_frames(2.0), 33)   # 1 + 16*2
        self.assertEqual(app.get_num_frames(3.5), 57)   # 1 + 16*3.5 -> 57
        self.assertEqual(app.get_num_frames(5.0), 81)   # 1 + 16*5
        # Verify min clamp
        self.assertEqual(app.get_num_frames(0.1), 17)

    def test_extract_frame_invalid(self):
        """extract_frame_from_video returns None for invalid or missing files."""
        self.assertIsNone(app.extract_frame_from_video(None, 0.0))
        self.assertIsNone(app.extract_frame_from_video("non_existent_file.mp4", 1.0))

    def test_heartbeat_toggle(self):
        """Heartbeat toggle should update backend state."""
        backend.set_smart_heartbeat_state(True)
        self.assertTrue(backend.smart_heartbeat_enabled)
        backend.set_smart_heartbeat_state(False)
        self.assertFalse(backend.smart_heartbeat_enabled)

    @patch("modal.Cls.from_name")
    def test_remote_inference_mock(self, mock_from_name):
        """Verify run_remote_inference streams updates and writes MP4 output."""
        mock_instance = MagicMock()
        mock_instance.generate.remote_gen.return_value = [
            {"type": "progress", "step": 0, "total_steps": 6},
            {"type": "progress", "step": 5, "total_steps": 6},
            {"type": "output", "video_bytes": b"\x00\x00\x00\x1cftypisommock_mp4_bytes", "filename": "output.mp4"}
        ]
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        dummy_img = Image.new("RGB", (640, 480), color="purple")
        progress_calls = []

        def progress_cb(frac, desc=None):
            progress_calls.append((frac, desc))

        video_path = backend.run_remote_inference(
            resized_image=dummy_img,
            processed_last_image=None,
            prompt="A cinematic test bus",
            steps=6,
            negative_prompt="blurry",
            num_frames=57,
            guidance_scale=5.0,
            guidance_scale_2=2.0,
            current_seed=999,
            scheduler_name="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=32,
            quality=5,
            duration_seconds=3.5,
            progress=progress_cb
        )

        self.assertTrue(os.path.exists(video_path))
        self.assertTrue(video_path.endswith(".mp4"))
        self.assertGreater(len(progress_calls), 0)


if __name__ == "__main__":
    unittest.main()
