import os
import io
import json
import tempfile
import pathlib
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import cv2
import numpy as np

import studio_pipeline
from studio_pipeline import StudioPipeline, ASPECT_RATIO_PRESETS


class TestStudioPipeline(unittest.TestCase):
    """
    Comprehensive test suite for the Studio Multi-Agent Video Production Pipeline.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pipeline = StudioPipeline(
            api_key="test_key_dummy",
            model="gemma4:31b",
            output_base_dir=self.temp_dir.name,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: JSON Parsing & Schema Robustness
    # ─────────────────────────────────────────────────────────────────────────

    def test_json_parsing_clean(self):
        sample = '{"title": "Sharing Bear", "art_style": "2D", "characters": [], "scenes": []}'
        data = StudioPipeline._parse_json_safely(sample)
        self.assertEqual(data["title"], "Sharing Bear")

    def test_json_parsing_markdown_wrapped(self):
        sample = '```json\n{"title": "Wrapped Story", "art_style": "Cartoon", "characters": [], "scenes": []}\n```'
        data = StudioPipeline._parse_json_safely(sample)
        self.assertEqual(data["title"], "Wrapped Story")

    def test_json_parsing_trailing_comma_recovery(self):
        sample = '{"title": "Trailing", "characters": [{"name": "Leo",}], "scenes": []}'
        data = StudioPipeline._parse_json_safely(sample)
        self.assertEqual(data["title"], "Trailing")
        self.assertEqual(len(data["characters"]), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Gemma 4: 31B Story Generation Integration Contract
    # ─────────────────────────────────────────────────────────────────────────

    @patch("urllib.request.urlopen")
    def test_generate_story_and_characters_contract(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "title": "The Golden Apple",
                            "art_style": "2D cartoon",
                            "theme": "Sharing is caring",
                            "characters": [
                                {
                                    "id": "char_1",
                                    "name": "Barnaby",
                                    "face": "Round friendly bear face",
                                    "hair": "Brown fur",
                                    "clothing": "Red bowtie",
                                    "signature_details": "Honey pot",
                                    "z_image_prompt": "Portrait of Barnaby the bear",
                                },
                                {
                                    "id": "char_2",
                                    "name": "Pippin",
                                    "face": "Tiny pink nose",
                                    "hair": "White fluffy fur",
                                    "clothing": "Blue scarf",
                                    "signature_details": "Twitching whiskers",
                                    "z_image_prompt": "Portrait of Pippin the rabbit",
                                },
                            ],
                            "scenes": [
                                {
                                    "scene_number": 1,
                                    "title": "Finding the Treat",
                                    "duration_seconds": 6.0,
                                    "scenario": "Barnaby finds honey",
                                    "qwen_scene_prompt": "Barnaby sitting near flower patch",
                                    "wan_motion_prompt": "Barnaby happily inspecting honey",
                                    "camera_movement": "Slow Cinematic Push-In",
                                    "character_ids": ["char_1"],
                                }
                            ],
                        })
                    }
                }
            ]
        }
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = self.pipeline.generate_story_and_characters(
            prompt="A story about sharing honey",
            num_scenes=1,
            target_duration=6.0,
        )

        self.assertEqual(result["title"], "The Golden Apple")
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(result["characters"][0]["name"], "Barnaby")
        self.assertEqual(result["characters"][1]["name"], "Pippin")
        self.assertEqual(len(result["scenes"]), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Character Generation with Locked Identical Seed
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.generate_z_image_turbo_remote")
    def test_character_generation_locked_seed(self, mock_z_image):
        dummy_img = Image.new("RGB", (1024, 1024), color="blue")
        mock_z_image.return_value = (dummy_img, "/dummy/path/char.png")

        characters = [
            {"id": "char_1", "name": "Barnaby", "z_image_prompt": "Bear portrait"},
            {"id": "char_2", "name": "Pippin", "z_image_prompt": "Rabbit portrait"},
        ]

        locked_seed = 98765
        updated_chars = self.pipeline.generate_character_portraits(
            characters=characters,
            shared_seed=locked_seed,
        )

        self.assertEqual(len(updated_chars), 2)
        # Check both called with the EXACT SAME locked seed
        self.assertEqual(mock_z_image.call_count, 2)
        call_args_1 = mock_z_image.call_args_list[0][1]
        call_args_2 = mock_z_image.call_args_list[1][1]
        self.assertEqual(call_args_1["seed"], locked_seed)
        self.assertEqual(call_args_2["seed"], locked_seed)
        self.assertEqual(updated_chars[0]["portrait_seed"], locked_seed)
        self.assertEqual(updated_chars[1]["portrait_seed"], locked_seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Scene Keyframe Generation with Multi-Reference Transfer
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.qwen_image_edit_remote")
    def test_scene_keyframes_multi_ref_transfer(self, mock_qwen):
        img_bytes_io = io.BytesIO()
        Image.new("RGB", (576, 1024), color="green").save(img_bytes_io, format="JPEG")
        mock_qwen.return_value = [img_bytes_io.getvalue()]

        char_1_img = Image.new("RGB", (1024, 1024), color="red")
        char_2_img = Image.new("RGB", (1024, 1024), color="yellow")

        characters = [
            {"id": "char_1", "name": "Barnaby", "portrait_pil": char_1_img},
            {"id": "char_2", "name": "Pippin", "portrait_pil": char_2_img},
        ]
        scenes = [
            {
                "scene_number": 1,
                "title": "Meeting at the Meadow",
                "character_ids": ["char_1", "char_2"],
                "qwen_scene_prompt": "Barnaby and Pippin meeting in meadow",
            }
        ]

        updated_scenes = self.pipeline.generate_scene_keyframes(
            scenes=scenes,
            characters=characters,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            shared_seed=42,
        )

        self.assertEqual(len(updated_scenes), 1)
        self.assertTrue(os.path.exists(updated_scenes[0]["keyframe_path"]))

        # Check Qwen call received base image and reference image
        mock_qwen.assert_called_once()
        qwen_kwargs = mock_qwen.call_args[1]
        self.assertEqual(qwen_kwargs["base_image"], char_1_img)
        self.assertEqual(qwen_kwargs["ref_images"], [char_2_img])
        self.assertEqual(qwen_kwargs["custom_width"], 576)
        self.assertEqual(qwen_kwargs["custom_height"], 1024)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Scene Video Animation (Wan 2.2 with 4k+1 rule and CFG)
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    def test_scene_videos_wan_constraints(self, mock_wan):
        # Create a dummy video file for mock return
        dummy_vid = pathlib.Path(self.temp_dir.name) / "temp_dummy.mp4"
        with open(dummy_vid, "wb") as f:
            f.write(b"\x00\x00\x00 dummy mp4")
        mock_wan.return_value = str(dummy_vid)

        keyframe = Image.new("RGB", (480, 832), color="purple")
        scenes = [
            {
                "scene_number": 1,
                "title": "Scene 1",
                "keyframe_pil": keyframe,
                "duration_seconds": 3.5,
                "wan_motion_prompt": "smooth camera push-in",
                "camera_movement": "Slow Push-In",
            }
        ]

        clips = self.pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
        )

        self.assertEqual(len(clips), 1)
        self.assertTrue(os.path.exists(clips[0]))

        # Verify 4k+1 constraint (3.5s -> 57 frames) and CFG 2.0
        wan_kwargs = mock_wan.call_args[1]
        self.assertEqual(wan_kwargs["num_frames"], 57)  # 4 * 14 + 1
        self.assertEqual(wan_kwargs["guidance_scale"], 2.0)
        self.assertEqual(wan_kwargs["resized_image"].size, (480, 832))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Video Assembly & Stitching with FFmpeg
    # ─────────────────────────────────────────────────────────────────────────

    def test_stitch_clips_ffmpeg(self):
        """Creates 2 real minimal MP4 video files with OpenCV and stitches them with FFmpeg."""
        clip1_path = os.path.join(self.temp_dir.name, "clip1.mp4")
        clip2_path = os.path.join(self.temp_dir.name, "clip2.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for p, color in [(clip1_path, (255, 0, 0)), (clip2_path, (0, 255, 0))]:
            out = cv2.VideoWriter(p, fourcc, 16.0, (160, 120))
            frame = np.full((120, 160, 3), color, dtype=np.uint8)
            for _ in range(16):  # 1 second each
                out.write(frame)
            out.release()

        final_movie = self.pipeline.stitch_clips(
            clip_paths=[clip1_path, clip2_path],
            movie_title="Test_Kids_Short",
        )

        self.assertTrue(os.path.exists(final_movie))
        self.assertGreater(os.path.getsize(final_movie), 1000)

        # Verify combined duration with OpenCV
        cap = cv2.VideoCapture(final_movie)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertGreaterEqual(total_frames, 30)  # ~32 frames total


if __name__ == "__main__":
    unittest.main()
