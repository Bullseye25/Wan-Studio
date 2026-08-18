import io
import inspect
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

import backend
import app


class TestLoRAFreeContracts(unittest.TestCase):
    def test_flux_klein_remote_signature(self):
        """Verify flux_klein_remote signature does not contain any LoRA parameters."""
        sig = inspect.signature(backend.flux_klein_remote)
        params = list(sig.parameters.keys())
        self.assertNotIn("loras", params)
        self.assertNotIn("lora_name", params)
        self.assertNotIn("lora_weight", params)

    def test_app_no_lora_symbols(self):
        """Verify app module does not define or export LoRA symbols."""
        self.assertFalse(hasattr(app, "LORA_STYLES"))
        self.assertFalse(hasattr(app, "LORA_TITLES"))
        self.assertFalse(hasattr(app, "build_lora_list"))
        self.assertFalse(hasattr(app, "update_lora_sliders"))
        self.assertFalse(hasattr(app, "update_lora_prompt"))

    def test_no_wan_t2v_in_backend(self):
        """Verify Wan 2.1 T2V functions are removed from backend."""
        self.assertFalse(hasattr(backend, "run_remote_inference_t2v"))
        self.assertFalse(hasattr(backend, "get_available_loras"))

    @patch("modal.Cls.from_name")
    def test_flux_klein_remote_mock_call(self, mock_from_name):
        """Verify flux_klein_remote serializes images and dispatches call to Modal cleanly."""
        # Create a mock modal model instance
        mock_instance = MagicMock()
        fake_jpeg = b"\xff\xd8\xff\xe0fake_jpeg_bytes"
        mock_instance.generate.remote.return_value = fake_jpeg
        mock_from_name.return_value = MagicMock(return_value=mock_instance)

        base_img = Image.new("RGB", (256, 256), color="red")
        ref_img = Image.new("RGB", (256, 256), color="blue")

        result = backend.flux_klein_remote(
            base_image=base_img,
            ref_images=[ref_img],
            prompt="a futuristic red bus",
            negative_prompt="blurry",
            seed=12345,
            steps=20,
            guidance_scale=4.0,
            canvas_mode="Square (1024x1024)",
            canvas_fit_mode="Cover",
            pad_color="#000000",
            custom_width=1024,
            custom_height=1024,
        )

        self.assertEqual(result, fake_jpeg)
        mock_from_name.assert_called_once_with("flux-klein-backend", "FluxKleinModel")
        mock_instance.generate.remote.assert_called_once()
        call_kwargs = mock_instance.generate.remote.call_args[1]

        # Verify no LoRA argument was passed
        self.assertNotIn("loras", call_kwargs)
        self.assertEqual(call_kwargs["prompt"], "a futuristic red bus")
        self.assertEqual(call_kwargs["seed"], 12345)
        self.assertEqual(call_kwargs["steps"], 20)

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
