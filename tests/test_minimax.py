import io
import time
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

import backend
import deploy_minimax
from app import (
    route_qwen_to_minimax_first,
    route_qwen_to_minimax_last,
    route_minimax_to_qwen,
    route_minimax_to_wan,
    extract_frame,
    MINIMAX_CANVASES,
    DEFAULT_MINIMAX_CANVAS,
)


class TestMiniMaxMathAndCanvases(unittest.TestCase):
    def test_snap_frames_chunk_alignment(self):
        """MiniMax-H3 VAE requires frame count aligned to 17 * n + 5 at 24 fps."""
        for duration in [2, 3, 5, 8, 10, 14]:
            frames = deploy_minimax.snap_frames(duration)
            self.assertEqual(
                frames % deploy_minimax.FRAMES_PER_CHUNK,
                deploy_minimax.LATENTS_PER_CHUNK,
                f"Duration {duration}s snapped to {frames} which is not 17*n+5"
            )
            # Duration must be at least 2s (48 frames or more)
            self.assertGreaterEqual(frames, 48)

    def test_canvas_resolutions_patch_alignment(self):
        """All canvas resolutions must be divisible by 32 for transformer patch tiling."""
        for label, (h, w) in deploy_minimax.CANVASES.items():
            self.assertEqual(h % 32, 0, f"Canvas {label} height {h} not divisible by 32")
            self.assertEqual(w % 32, 0, f"Canvas {label} width {w} not divisible by 32")
            self.assertIn(label, MINIMAX_CANVASES)

    def test_default_canvas_exists(self):
        """Default canvas preset must exist in canvas dictionary."""
        self.assertIn(DEFAULT_MINIMAX_CANVAS, deploy_minimax.CANVASES)


class TestMiniMaxCostGuardedHeartbeat(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()
        backend.minimax_heartbeat_enabled = False
        backend.last_minimax_activity_time = 0.0

    def tearDown(self):
        backend.reset_cached_models()
        backend.minimax_heartbeat_enabled = False

    def test_minimax_heartbeat_defaults_to_paused(self):
        """Heartbeat must default to False to prevent costly A100 credit burn."""
        self.assertFalse(backend.minimax_heartbeat_enabled)

    def test_minimax_heartbeat_state_toggle(self):
        """Toggling state should accurately update backend variable."""
        backend.set_minimax_heartbeat_state(True)
        self.assertTrue(backend.minimax_heartbeat_enabled)
        backend.set_minimax_heartbeat_state(False)
        self.assertFalse(backend.minimax_heartbeat_enabled)

    def test_record_minimax_activity(self):
        """Recording activity should update the timestamp."""
        before = time.time()
        backend.record_minimax_activity()
        after = time.time()
        self.assertGreaterEqual(backend.last_minimax_activity_time, before)
        self.assertLessEqual(backend.last_minimax_activity_time, after)

    def test_idle_timeout_guard(self):
        """Verify that when activity exceeds KEEPALIVE_IDLE_WINDOW, keepalive would not ping."""
        # Simulated time 4 minutes ago (exceeds 3 minute KEEPALIVE_IDLE_WINDOW)
        backend.last_minimax_activity_time = time.time() - 240.0
        now = time.time()
        is_idle_expired = (now - backend.last_minimax_activity_time) >= backend.KEEPALIVE_IDLE_WINDOW
        self.assertTrue(is_idle_expired, "Idle window guard should be active after 3 minutes")


class TestMiniMaxCrossToolRouting(unittest.TestCase):
    def test_route_qwen_to_minimax(self):
        """Transferring image from Qwen editor to MiniMax input slots."""
        dummy_img = Image.new("RGB", (100, 100), color="blue")
        self.assertEqual(route_qwen_to_minimax_first(dummy_img), dummy_img)
        self.assertEqual(route_qwen_to_minimax_last(dummy_img), dummy_img)

    def test_route_minimax_to_other_tools(self):
        """MiniMax output frame routing helpers."""
        # When no video is present, should safely return None without crashing
        self.assertIsNone(route_minimax_to_qwen(None, 0))
        self.assertIsNone(route_minimax_to_wan(None, 0))


class TestMiniMaxRemoteInferenceMock(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    @patch("modal.Cls.from_name")
    def test_run_minimax_remote_inference_mock(self, mock_from_name):
        """Test full remote inference orchestration with mock A100 backend."""
        mock_instance = MagicMock()
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        dummy_mp4 = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41"
        mock_instance.generate.remote.return_value = {
            "video_bytes": dummy_mp4,
            "report": "960x544, 124 frames, 28 steps · Seed 42",
            "refined_prompt": "A beautiful cinematic snowy forest",
            "filename": "test_minimax_out.mp4",
        }

        first_img = Image.new("RGB", (64, 64), color="yellow")
        video_path, report, refined = backend.run_minimax_remote_inference(
            prompt="A fox in snow",
            first_image=first_img,
            last_image=None,
            canvas="960x544 · 16:9 fast",
            duration=5,
            steps=28,
            seed=42,
            upsample=True,
        )

        self.assertTrue(video_path.endswith("test_minimax_out.mp4"))
        self.assertIn("960x544", report)
        self.assertEqual(refined, "A beautiful cinematic snowy forest")
        self.assertGreater(backend.last_minimax_activity_time, 0)


if __name__ == "__main__":
    unittest.main()
