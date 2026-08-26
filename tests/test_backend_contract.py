import io
import inspect
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

import backend
import app


class TestQwenAndWanContracts(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    def test_qwen_image_edit_remote_signature(self):
        """Verify qwen_image_edit_remote signature does not contain any LoRA parameters."""
        sig = inspect.signature(backend.qwen_image_edit_remote)
        params = list(sig.parameters.keys())
        self.assertIn("base_image", params)
        self.assertIn("ref_images", params)
        self.assertIn("prompt", params)
        self.assertIn("true_cfg_scale", params)
        self.assertNotIn("loras", params)
        self.assertNotIn("lora_name", params)
        self.assertNotIn("lora_weight", params)

    def test_app_has_minimax_and_qwen_symbols(self):
        """Verify app module defines and exports Qwen and MiniMax symbols."""
        self.assertFalse(hasattr(app, "run_flux_generate"))
        self.assertTrue(hasattr(app, "run_qwen_generate"))
        self.assertTrue(hasattr(app, "run_minimax_generate"))
        self.assertTrue(hasattr(app, "route_qwen_to_minimax_first"))
        self.assertTrue(hasattr(app, "route_minimax_to_qwen"))

    def test_minimax_remote_inference_signature(self):
        """Verify run_minimax_remote_inference has all required parameters."""
        sig = inspect.signature(backend.run_minimax_remote_inference)
        params = list(sig.parameters.keys())
        self.assertIn("prompt", params)
        self.assertIn("first_image", params)
        self.assertIn("last_image", params)
        self.assertIn("canvas", params)
        self.assertIn("duration", params)
        self.assertIn("steps", params)
        self.assertIn("seed", params)
        self.assertIn("upsample", params)

    def test_no_wan_t2v_in_backend(self):
        """Verify Wan 2.1 T2V functions are removed from backend."""
        self.assertFalse(hasattr(backend, "run_remote_inference_t2v"))
        self.assertFalse(hasattr(backend, "get_available_loras"))

    @patch("modal.Cls.from_name")
    def test_qwen_image_edit_remote_mock_call(self, mock_from_name):
        """Verify qwen_image_edit_remote serializes images and dispatches call to Modal cleanly."""
        mock_instance = MagicMock()
        fake_jpeg = b"\xff\xd8\xff\xe0fake_jpeg_bytes"
        mock_instance.generate.remote.return_value = [fake_jpeg]
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        base_img = Image.new("RGB", (256, 256), color="red")
        ref_img = Image.new("RGB", (256, 256), color="blue")

        result = backend.qwen_image_edit_remote(
            base_image=base_img,
            ref_images=[ref_img],
            prompt="a futuristic red bus",
            negative_prompt="blurry",
            seed=12345,
            steps=20,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images=1,
            custom_width=1024,
            custom_height=1024,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(result[0], fake_jpeg)
        mock_from_name.assert_called_once_with("qwen-image-backend", "QwenImageEditModel")
        mock_instance.generate.remote.assert_called_once()
        call_kwargs = mock_instance.generate.remote.call_args[1]

        # Verify arguments passed
        self.assertEqual(call_kwargs["prompt"], "a futuristic red bus")
        self.assertEqual(call_kwargs["negative_prompt"], "blurry")
        self.assertEqual(call_kwargs["seed"], 12345)
        self.assertEqual(call_kwargs["steps"], 20)
        self.assertEqual(call_kwargs["true_cfg_scale"], 4.0)
        self.assertEqual(call_kwargs["num_images"], 1)

    @patch("modal.Cls.from_name")
    def test_qwen_image_edit_multi_image_batch(self, mock_from_name):
        """Verify qwen_image_edit_remote dispatches num_images=4 and returns 4 images."""
        mock_instance = MagicMock()
        fake_jpegs = [f"fake_jpeg_{i}".encode("utf-8") for i in range(4)]
        mock_instance.generate.remote.return_value = fake_jpegs
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        base_img = Image.new("RGB", (256, 256), color="green")

        result = backend.qwen_image_edit_remote(
            base_image=base_img,
            prompt="cyberpunk cityscape",
            negative_prompt="dark, murky",
            seed=5000,
            steps=15,
            num_images=4,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        mock_from_name.assert_called_once_with("qwen-image-backend", "QwenImageEditModel")
        call_kwargs = mock_instance.generate.remote.call_args[1]
        self.assertEqual(call_kwargs["num_images"], 4)
        self.assertEqual(call_kwargs["seed"], 5000)

    @patch("modal.Cls.from_name")
    def test_wan_remote_inference_mock_call(self, mock_from_name):
        """Verify run_remote_inference serializes images and parses streaming events correctly."""
        mock_instance = MagicMock()
        fake_video_bytes = b"fake_mp4_video_data"
        # Simulate generator yielding progress then output
        mock_instance.generate.remote_gen.return_value = [
            {"type": "progress", "step": 1, "total_steps": 6},
            {"type": "progress", "step": 6, "total_steps": 6},
            {"type": "output", "filename": "test_output.mp4", "video_bytes": fake_video_bytes},
        ]
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        resized_img = Image.new("RGB", (832, 480), color="cyan")
        progress_calls = []

        def progress_cb(pct, desc=""):
            progress_calls.append((pct, desc))

        video_path, task_name = backend.run_remote_inference(
            resized_image=resized_img,
            processed_last_image=None,
            prompt="bus driving smoothly on road",
            steps=6,
            negative_prompt="low quality",
            num_frames=57,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            current_seed=42,
            scheduler_name="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=16,
            quality=5,
            duration_seconds=3.5,
            progress=progress_cb,
        )

        self.assertTrue(video_path.endswith("test_output.mp4"))
        self.assertEqual(task_name, "task_42")
        mock_from_name.assert_called_once_with("wan-backend", "WanModel")
        self.assertGreater(len(progress_calls), 0)


if __name__ == "__main__":
    unittest.main()
