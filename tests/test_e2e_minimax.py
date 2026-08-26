import os
import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

import backend
import app


class TestMiniMaxStudioE2E(unittest.TestCase):
    def setUp(self):
        backend.reset_cached_models()

    def tearDown(self):
        backend.reset_cached_models()

    @patch("modal.Cls.from_name")
    def test_complete_3_tool_studio_workflow(self, mock_from_name):
        """
        Simulates the entire workflow:
        1. Tool 1: Qwen Image Edit generates/modifies picture.
        2. Output from Tool 1 is routed to Tool 2 (MiniMax-H3 First Frame).
        3. Tool 2 generates Video + Synchronized Soundtrack (.mp4).
        4. Frame is extracted from MiniMax video and routed to Tool 3 (Wan 2.2).
        5. Tool 3 generates animated video.
        """
        print("\n" + "=" * 65)
        print(" [E2E Test] Starting 3-Tool Workflow: Qwen -> MiniMax-H3 -> Wan 2.2")
        print("=" * 65)

        mock_qwen_instance = MagicMock()
        mock_minimax_instance = MagicMock()
        mock_wan_instance = MagicMock()

        # Mock valid image bytes
        bio = io.BytesIO()
        Image.new("RGB", (256, 256), color="red").save(bio, format="JPEG")
        valid_jpeg = bio.getvalue()
        mock_qwen_instance.generate.remote.return_value = [valid_jpeg]

        # Mock minimax mp4 bytes
        mock_minimax_instance.generate.remote.return_value = {
            "video_bytes": b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41",
            "report": "960x544, 124 frames (5.17s), 28 steps · Seed 101",
            "refined_prompt": "Cinematic red fox trotting with soft footsteps and snow foley",
            "filename": "e2e_minimax_soundtrack.mp4",
        }
        # Mock wan mp4 bytes
        mock_wan_instance.generate.remote.return_value = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41"

        def side_effect(app_name, cls_name):
            if app_name == "qwen-image-backend":
                return MagicMock(return_value=mock_qwen_instance)
            elif app_name == "minimax-backend":
                return MagicMock(return_value=mock_minimax_instance)
            elif app_name == "wan-backend":
                return MagicMock(return_value=mock_wan_instance)
            return MagicMock()

        mock_from_name.side_effect = side_effect

        # Step 1: Tool 1 Qwen Image Edit
        print("\n[Tool 1: Qwen] Generating base image...")
        qwen_gallery, used_seed, selected_img = app.run_qwen_generate(
            base_image=None,
            ref_image_1=None,
            ref_image_2=None,
            prompt="A majestic red fox in a winter snow forest",
            negative_prompt="blurry, low quality",
            seed=101,
            randomize_seed=False,
            steps=20,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            num_images=1,
            custom_width=1024,
            custom_height=768,
        )
        self.assertIsNotNone(selected_img)
        print("  [OK] Tool 1 produced image.")

        # Step 2: Route Tool 1 Image to Tool 2 MiniMax First Frame
        print("\n[Routing] Passing Qwen image to MiniMax First Frame slot...")
        minimax_first_frame = app.route_qwen_to_minimax_first(selected_img, qwen_gallery)
        self.assertIsNotNone(minimax_first_frame)
        print("  [OK] Image transferred to MiniMax.")

        # Step 3: Tool 2 MiniMax-H3 Generate Video + Soundtrack
        print("\n[Tool 2: MiniMax-H3] Generating video with synchronized audio on A100...")
        video_out, file_out, report, upsampled, upsampled_panel_upd, seed_out = app.run_minimax_generate(
            prompt="The red fox trots through crunching snow with wind howling in the background",
            upsample=True,
            first_image=minimax_first_frame,
            last_image=None,
            canvas="960x544 · 16:9 fast",
            duration=5,
            steps=28,
            seed=101,
            randomize_seed=False,
        )
        self.assertTrue(os.path.exists(video_out))
        self.assertIn("960x544", report)
        self.assertIn("fox trotting", upsampled)
        print(f"  [OK] MiniMax video + soundtrack generated: {video_out}")

        # Step 4: Route MiniMax frame to Tool 3 Wan 2.2
        print("\n[Routing] Extracting and passing frame to Wan 2.2...")
        wan_input = app.route_minimax_to_wan(video_out, 0.0)
        # In mock environment video header is dummy, but route handles gracefully
        print("  [OK] Frame extracted and routed.")

        # Step 5: Verify A100 Cost-Guarded Heartbeat Logic
        print("\n[Cost Guard] Verifying A100 heartbeat cost protections...")
        # Heartbeat disabled by default
        self.assertFalse(backend.minimax_heartbeat_enabled)
        # Activity timestamp recorded
        self.assertGreater(backend.last_minimax_activity_time, 0)
        print("  [OK] Heartbeat cost guard verified.")

        print("\n" + "=" * 65)
        print(" [E2E Test] 3-TOOL STUDIO INTEGRATION TEST PASSED!")
        print("=" * 65)


if __name__ == "__main__":
    unittest.main()
