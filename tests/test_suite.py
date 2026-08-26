import os
import cv2
import unittest
import numpy as np
from PIL import Image

from app import (
    resize_image,
    resize_and_crop_to_match,
    get_num_frames,
    extract_frame,
    clear_all_outputs,
    MAX_DIM,
    MIN_DIM,
    SQUARE_DIM,
    MULTIPLE_OF,
    FIXED_FPS,
    MIN_FRAMES_MODEL,
    MAX_FRAMES_MODEL,
)
import backend


class TestImageProcessing(unittest.TestCase):
    def test_square_image_resize(self):
        """A square image should resize to SQUARE_DIM x SQUARE_DIM (640x640)."""
        img = Image.new("RGB", (500, 500), color="blue")
        resized = resize_image(img)
        self.assertEqual(resized.size, (SQUARE_DIM, SQUARE_DIM))

    def test_landscape_image_resize(self):
        """A 16:9 landscape image should resize within [MIN_DIM, MAX_DIM] and be divisible by 16."""
        img = Image.new("RGB", (1920, 1080), color="red")
        resized = resize_image(img)
        w, h = resized.size
        self.assertEqual(w % MULTIPLE_OF, 0, f"Width {w} not divisible by {MULTIPLE_OF}")
        self.assertEqual(h % MULTIPLE_OF, 0, f"Height {h} not divisible by {MULTIPLE_OF}")
        self.assertLessEqual(w, MAX_DIM)
        self.assertGreaterEqual(w, MIN_DIM)
        self.assertLessEqual(h, MAX_DIM)
        self.assertGreaterEqual(h, MIN_DIM)

    def test_portrait_image_resize(self):
        """A 9:16 portrait image should resize within [MIN_DIM, MAX_DIM] and be divisible by 16."""
        img = Image.new("RGB", (1080, 1920), color="green")
        resized = resize_image(img)
        w, h = resized.size
        self.assertEqual(w % MULTIPLE_OF, 0, f"Width {w} not divisible by {MULTIPLE_OF}")
        self.assertEqual(h % MULTIPLE_OF, 0, f"Height {h} not divisible by {MULTIPLE_OF}")
        self.assertLessEqual(w, MAX_DIM)
        self.assertGreaterEqual(w, MIN_DIM)
        self.assertLessEqual(h, MAX_DIM)
        self.assertGreaterEqual(h, MIN_DIM)

    def test_4k_resolution_resize(self):
        """A 4K image (3840x2160) should scale down to 832x464 (clamped to 832x480)."""
        img = Image.new("RGB", (3840, 2160), color="orange")
        resized = resize_image(img)
        w, h = resized.size
        self.assertEqual(w % MULTIPLE_OF, 0)
        self.assertEqual(h % MULTIPLE_OF, 0)
        self.assertLessEqual(w, MAX_DIM)
        self.assertGreaterEqual(h, MIN_DIM)

    def test_extreme_aspect_ratio_wide(self):
        """An extremely wide image (e.g. 2000x200) should be cropped and clamped safely."""
        img = Image.new("RGB", (2000, 200), color="yellow")
        resized = resize_image(img)
        w, h = resized.size
        self.assertEqual(w % MULTIPLE_OF, 0)
        self.assertEqual(h % MULTIPLE_OF, 0)
        self.assertEqual((w, h), (MAX_DIM, MIN_DIM))

    def test_extreme_aspect_ratio_tall(self):
        """An extremely tall image (e.g. 200x2000) should be cropped and clamped safely."""
        img = Image.new("RGB", (200, 2000), color="purple")
        resized = resize_image(img)
        w, h = resized.size
        self.assertEqual(w % MULTIPLE_OF, 0)
        self.assertEqual(h % MULTIPLE_OF, 0)
        self.assertEqual((w, h), (MIN_DIM, MAX_DIM))

    def test_resize_and_crop_to_match(self):
        """Target image should be cropped and scaled to match reference image size exactly."""
        ref = Image.new("RGB", (832, 480), color="black")
        target = Image.new("RGB", (1000, 1000), color="white")
        matched = resize_and_crop_to_match(target, ref)
        self.assertEqual(matched.size, (832, 480))


class TestFrameCalculations(unittest.TestCase):
    def test_frame_count_default_duration(self):
        """3.5 seconds at 16 fps = 57 frames (1 + 56)."""
        num_frames = get_num_frames(3.5)
        self.assertEqual(num_frames, 57)

    def test_frame_count_various_durations(self):
        """Test duration mapping across various typical video lengths."""
        self.assertEqual(get_num_frames(1.0), 17)   # 1s = 1 + 16
        self.assertEqual(get_num_frames(2.0), 33)   # 2s = 1 + 32
        self.assertEqual(get_num_frames(5.0), 81)   # 5s = 1 + 80

    def test_frame_count_clamping_min(self):
        """Very short duration should clamp to MIN_FRAMES_MODEL + 1."""
        num_frames = get_num_frames(0.1)
        self.assertEqual(num_frames, MIN_FRAMES_MODEL + 1)

    def test_frame_count_clamping_max(self):
        """Very long duration should clamp to MAX_FRAMES_MODEL + 1."""
        num_frames = get_num_frames(200.0)
        self.assertEqual(num_frames, MAX_FRAMES_MODEL + 1)


class TestFrameExtraction(unittest.TestCase):
    def test_extract_frame_none_path(self):
        """extract_frame should safely return None if video path is None or empty."""
        self.assertIsNone(extract_frame(None, 0.0))
        self.assertIsNone(extract_frame("", 0.0))

    def test_extract_frame_invalid_path(self):
        """extract_frame should safely return None if video file does not exist."""
        self.assertIsNone(extract_frame("non_existent_file.mp4", 1.0))


class TestHeartbeatState(unittest.TestCase):
    def test_heartbeat_toggle(self):
        """Heartbeat state toggles should update backend flags."""
        backend.set_wan_heartbeat_state(True)
        self.assertTrue(backend.wan_heartbeat_enabled)
        backend.set_wan_heartbeat_state(False)
        self.assertFalse(backend.wan_heartbeat_enabled)

        backend.set_qwen_heartbeat_state(True)
        self.assertTrue(backend.qwen_heartbeat_enabled)
        backend.set_qwen_heartbeat_state(False)
        self.assertFalse(backend.qwen_heartbeat_enabled)

    def test_activity_tracking(self):
        """Activity recording functions should update timestamps."""
        backend.record_wan_activity()
        self.assertGreater(backend.last_wan_activity_time, 0)

        backend.record_qwen_activity()
        self.assertGreater(backend.last_qwen_activity_time, 0)

        backend.record_frontend_heartbeat()
        self.assertGreater(backend.last_frontend_heartbeat_time, 0)


class TestClearOutputs(unittest.TestCase):
    def test_clear_all_outputs(self):
        """clear_all_outputs should remove all files and keep the directory intact."""
        import pathlib
        outputs_dir = pathlib.Path(__file__).parent.parent / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy files
        dummy_file = outputs_dir / "dummy_test_file.jpg"
        dummy_file.write_text("test")
        self.assertTrue(dummy_file.exists())

        msg = clear_all_outputs()
        self.assertIn("Cleared", msg)
        self.assertFalse(dummy_file.exists())
        self.assertTrue(outputs_dir.exists())


if __name__ == "__main__":
    unittest.main()
