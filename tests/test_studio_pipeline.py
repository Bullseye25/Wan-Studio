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
from studio_pipeline import (
    StudioPipeline,
    ASPECT_RATIO_PRESETS,
    frames_for_duration,
    plan_scene_count,
    WAN_DEFAULTS,
    restore_extracted_video_frame,
)
import app


class TestStudioPipeline(unittest.TestCase):
    """
    Comprehensive test suite for the 6-stage Studio Multi-Agent Video Production Pipeline:
      1. Story & Character Breakdown (Gemma 4: 31B)
      2. Character Portrait Synthesis (Z-Image Turbo, locked seed)
      3. Scene Keyframe Synthesis (Qwen Image Edit, multi-reference)
      4. Scene Animation Direction (Gemma 4: 31B Vision Director per 5s clip)
      5. Video Clip Animation (Wan 2.2, 4k+1 frames & CFG stability)
      6. Video Assembly & Stitching (FFmpeg + JSON manifest)
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
    # Test 1: JSON Parsing, Frame Math & Scene Planning
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

    def test_frames_for_duration_4k_plus_1(self):
        """Every returned frame count must satisfy (n - 1) % 4 == 0 and stay within 17..81."""
        self.assertEqual(frames_for_duration(0.2), 17)
        self.assertEqual(frames_for_duration(1.0), 17)
        self.assertEqual(frames_for_duration(3.5), 57)
        self.assertEqual(frames_for_duration(5.0), 81)
        self.assertEqual(frames_for_duration(9.0), 81)
        for dur in [1.0, 1.8, 2.5, 3.5, 4.2, 5.0]:
            f = frames_for_duration(dur)
            self.assertEqual((f - 1) % 4, 0)
            self.assertGreaterEqual(f, 17)
            self.assertLessEqual(f, 81)

    def test_plan_scene_count_for_30s(self):
        """A 30s movie must use at least 6 scenes so no clip exceeds Wan's 5.0s native limit."""
        self.assertEqual(plan_scene_count(30.0, requested_scenes=5), 6)
        self.assertEqual(plan_scene_count(30.0, requested_scenes=6), 6)
        self.assertEqual(plan_scene_count(30.0, requested_scenes=8), 8)
        self.assertEqual(plan_scene_count(15.0, requested_scenes=3), 3)

    def test_normalize_story_repairs_broken_llm_output(self):
        raw = {
            "title": "Test Film",
            "characters": [{"name": "Barnaby"}, {"id": "char_1", "name": "DuplicateId"}],
            "scenes": [
                {"title": "S1", "duration_seconds": 12.0, "character_ids": ["nonexistent"]},
                {"title": "S2", "duration_seconds": 12.0, "character_ids": "char_1"},
            ],
        }
        norm = StudioPipeline.normalize_story(
            raw, prompt="Sharing honey", num_scenes=2, target_duration=10.0, art_style="2D cartoon"
        )
        ids = [c["id"] for c in norm["characters"]]
        self.assertEqual(len(ids), len(set(ids)), "Character IDs must be deduplicated")
        for s in norm["scenes"]:
            self.assertLessEqual(s["duration_seconds"], 5.0)
            self.assertTrue(all(cid in ids for cid in s["character_ids"]))

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
                                    "duration_seconds": 5.0,
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
            target_duration=5.0,
        )

        self.assertEqual(result["title"], "The Golden Apple")
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(result["characters"][0]["name"], "Barnaby")
        self.assertEqual(result["characters"][1]["name"], "Pippin")
        self.assertEqual(len(result["scenes"]), 1)
        self.assertLessEqual(result["scenes"][0]["duration_seconds"], 5.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Character Generation with Locked Identical Seed
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.generate_z_image_turbo_remote")
    def test_character_generation_locked_seed(self, mock_z_image):
        dummy_img = Image.new("RGB", (1024, 1024), color="blue")
        mock_z_image.return_value = (dummy_img, "/dummy/path/char.png")

        characters = [
            {"id": "char_1", "name": "Barnaby", "face": "round face", "z_image_prompt": "Bear portrait"},
            {"id": "char_2", "name": "Pippin", "face": "pink nose", "z_image_prompt": "Rabbit portrait"},
        ]

        locked_seed = 98765
        updated_chars = self.pipeline.generate_character_portraits(
            characters=characters,
            shared_seed=locked_seed,
            art_style="2D vibrant cartoon animation",
        )

        self.assertEqual(len(updated_chars), 2)
        self.assertEqual(mock_z_image.call_count, 2)
        call_args_1 = mock_z_image.call_args_list[0][1]
        call_args_2 = mock_z_image.call_args_list[1][1]
        self.assertEqual(call_args_1["seed"], locked_seed)
        self.assertEqual(call_args_2["seed"], locked_seed)
        self.assertIn("2D vibrant cartoon animation", call_args_1["prompt"])
        self.assertEqual(call_args_1["width"], 576)
        self.assertEqual(call_args_1["height"], 1024)
        self.assertIn("9:16 vertical full-body", call_args_1["prompt"])
        self.assertEqual(updated_chars[0]["portrait_seed"], locked_seed)
        self.assertEqual(updated_chars[1]["portrait_seed"], locked_seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Scene Keyframe Generation with Multi-Reference Transfer & Caps
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

        mock_qwen.assert_called_once()
        qwen_kwargs = mock_qwen.call_args[1]
        self.assertEqual(qwen_kwargs["base_image"], char_1_img)
        self.assertEqual(qwen_kwargs["ref_images"], [char_2_img])
        self.assertIn("Barnaby is the character in image 1", qwen_kwargs["prompt"])
        self.assertIn("Pippin is the character in image 2", qwen_kwargs["prompt"])
        self.assertEqual(qwen_kwargs["custom_width"], 576)
        self.assertEqual(qwen_kwargs["custom_height"], 1024)

    @patch("studio_pipeline.qwen_image_edit_remote")
    def test_scene_keyframes_caps_refs_and_handles_missing_portraits(self, mock_qwen):
        img_bytes_io = io.BytesIO()
        Image.new("RGB", (576, 1024), color="green").save(img_bytes_io, format="JPEG")
        mock_qwen.return_value = [img_bytes_io.getvalue()]

        # 4 characters in scene -> Qwen must receive at most base + 2 ref_images
        chars = [
            {"id": f"char_{i}", "name": f"C{i}", "portrait_pil": Image.new("RGB", (64, 64))}
            for i in range(1, 5)
        ]
        scenes = [{"scene_number": 1, "title": "Crowd", "character_ids": ["char_1", "char_2", "char_3", "char_4"]}]
        self.pipeline.generate_scene_keyframes(scenes=scenes, characters=chars)
        qwen_kwargs = mock_qwen.call_args[1]
        self.assertEqual(len(qwen_kwargs["ref_images"]), 2)

        # No portraits available -> blank canvas fallback
        mock_qwen.reset_mock()
        self.pipeline.generate_scene_keyframes(scenes=scenes, characters=[{"id": "char_1", "name": "NoPic"}])
        qwen_kwargs2 = mock_qwen.call_args[1]
        self.assertIsInstance(qwen_kwargs2["base_image"], Image.Image)
        self.assertEqual(qwen_kwargs2["ref_images"], [])

    @patch("studio_pipeline.qwen_image_edit_remote")
    def test_scene_keyframes_locked_same_seed_across_all_scenes(self, mock_qwen):
        """All scene keyframes must receive the exact same locked qwen_seed (not shared_seed + idx)."""
        img_bytes_io = io.BytesIO()
        Image.new("RGB", (576, 1024), color="green").save(img_bytes_io, format="JPEG")
        mock_qwen.return_value = [img_bytes_io.getvalue()]

        chars = [{"id": "char_1", "name": "Barnaby", "portrait_pil": Image.new("RGB", (64, 64))}]
        scenes = [
            {"scene_number": 1, "title": "S1", "character_ids": ["char_1"]},
            {"scene_number": 2, "title": "S2", "character_ids": ["char_1"]},
            {"scene_number": 3, "title": "S3", "character_ids": ["char_1"]},
        ]
        locked_qwen_seed = 888888
        updated = self.pipeline.generate_scene_keyframes(
            scenes=scenes,
            characters=chars,
            shared_seed=locked_qwen_seed,
        )
        self.assertEqual(mock_qwen.call_count, 3)
        for idx, call in enumerate(mock_qwen.call_args_list):
            self.assertEqual(call[1]["seed"], locked_qwen_seed)
            self.assertEqual(updated[idx]["keyframe_seed"], locked_qwen_seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Stage 4 — Gemma 4: 31B Animation Director (Vision + Fallbacks)
    # ─────────────────────────────────────────────────────────────────────────

    def test_compile_director_plan_clamps_and_builds_timeline(self):
        raw = {
            "shot_type": "medium two-shot",
            "camera_movement": "slow push-in",
            "motion_intensity": 99,       # out of range -> clamp to 5
            "subject_action": "Barnaby hands Pippin half of the berry",
            "timeline": ["Barnaby looks at berry", "Barnaby splits berry and offers half"],  # only 2 -> padded to 3
            "environment_motion": "fireflies drifting softly",
            "identity_anchors": "Barnaby red bowtie, Pippin blue scarf",
            "wan_prompt": "Barnaby gently offers half a berry to Pippin",
            "negative_additions": "extra paws, floating objects",
            "guidance_scale": 9.5,        # clamp to 2.5
            "steps": 50,                  # clamp to 12
            "flow_shift": 0.1,            # clamp to 2.0
        }
        scene = {"scene_number": 1, "title": "Sharing", "duration_seconds": 5.0}
        cast = [{"name": "Barnaby"}, {"name": "Pippin"}]
        plan = StudioPipeline.compile_director_plan(raw, scene, cast, {"art_style": "2D vibrant cartoon"})

        self.assertEqual(plan["motion_intensity"], 5)
        self.assertEqual(plan["guidance_scale"], 2.5)
        self.assertEqual(plan["steps"], 12)
        self.assertEqual(plan["flow_shift"], 2.0)
        self.assertEqual(len(plan["timeline"]), 3)
        self.assertEqual(plan["num_frames"], 81)
        self.assertIn("[Timeline]: Phase 1", plan["wan_prompt"])
        self.assertIn("[Identity Anchors]: Barnaby red bowtie, Pippin blue scarf", plan["wan_prompt"])
        self.assertIn("extra paws", plan["negative_prompt"])
        self.assertIn("face morphing", plan["negative_prompt"])

    @patch.object(StudioPipeline, "_chat_completion")
    def test_direct_scene_animations_vision_payload(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "shot_type": "close-up",
            "camera_movement": "gentle dolly in",
            "motion_intensity": 2,
            "subject_action": "Barnaby smiles warmly and waves his paw",
            "timeline": [
                "Barnaby stands still holding the honey jar",
                "Barnaby raises his right paw in a friendly wave",
                "Barnaby settles with a warm smile",
            ],
            "environment_motion": "wildflowers swaying in light breeze",
            "identity_anchors": "plump brown bear, red bowtie",
            "wan_prompt": "Barnaby waves his paw warmly while holding his honey jar",
            "negative_additions": "paw distortion",
            "guidance_scale": 2.0,
            "steps": 8,
            "flow_shift": 3.0,
        })

        kf = Image.new("RGB", (576, 1024), color="orange")
        scenes = [{
            "scene_number": 1,
            "title": "Greeting",
            "duration_seconds": 5.0,
            "scenario": "Barnaby greets Pippin",
            "character_ids": ["char_1"],
            "keyframe_pil": kf,
        }]
        chars = [{"id": "char_1", "name": "Barnaby", "face": "round bear face", "clothing": "red bowtie"}]

        directed = self.pipeline.direct_scene_animations(
            scenes=scenes, characters=chars, story_meta={"title": "Test", "art_style": "2D cartoon"}, use_vision=True
        )

        self.assertEqual(len(directed), 1)
        plan = directed[0]["director"]
        self.assertEqual(plan["source"], "gemma-vision")
        self.assertEqual(plan["camera_movement"], "gentle dolly in")
        self.assertEqual(plan["num_frames"], 81)

        # Verify multimodal image_url payload was sent to Gemma
        sent_messages = mock_chat.call_args[0][0]
        user_content = sent_messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    @patch.object(StudioPipeline, "_chat_completion")
    def test_direct_scene_animations_fallback_chain(self, mock_chat):
        """If vision fails, falls back to text; if text also fails, falls back to deterministic plan."""
        # Case A: Vision raises, text succeeds
        mock_chat.side_effect = [
            RuntimeError("Vision endpoint busy"),
            json.dumps({"camera_movement": "static tripod shot", "motion_intensity": 2, "steps": 8}),
        ]
        kf = Image.new("RGB", (256, 256), color="blue")
        scenes = [{"scene_number": 1, "title": "S1", "duration_seconds": 3.5, "keyframe_pil": kf}]
        directed = self.pipeline.direct_scene_animations(scenes=scenes, characters=[], use_vision=True)
        self.assertEqual(directed[0]["director"]["source"], "gemma-text")

        # Case B: Both vision and text raise -> deterministic fallback still produces a valid plan
        mock_chat.side_effect = RuntimeError("Network down")
        directed2 = self.pipeline.direct_scene_animations(scenes=scenes, characters=[], use_vision=True)
        self.assertEqual(directed2[0]["director"]["source"], "fallback")
        self.assertEqual(directed2[0]["director"]["num_frames"], 57)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Stage 5 — Scene Video Animation (Wan 2.2 constraints & locked seed)
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    def test_scene_videos_wan_constraints_and_tuple_return(self, mock_wan):
        dummy_vid = pathlib.Path(self.temp_dir.name) / "temp_dummy.mp4"
        with open(dummy_vid, "wb") as f:
            f.write(b"\x00\x00\x00 dummy mp4")
        # Real run_remote_inference returns (video_path, task_name) tuple!
        mock_wan.return_value = (str(dummy_vid), "task_42")

        keyframe = Image.new("RGB", (480, 832), color="purple")
        scenes = [
            {
                "scene_number": 1,
                "title": "Scene 1",
                "keyframe_pil": keyframe,
                "duration_seconds": 3.5,
                "wan_motion_prompt": "smooth camera push-in",
                "camera_movement": "Slow Push-In",
            },
            {
                "scene_number": 2,
                "title": "Scene 2",
                "keyframe_pil": keyframe,
                "duration_seconds": 3.5,
                "wan_motion_prompt": "gentle smile",
                "camera_movement": "Static Tripod Shot",
            },
        ]

        locked_wan_seed = 999999
        clips = self.pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            base_seed=locked_wan_seed,
        )

        self.assertEqual(len(clips), 2)
        self.assertTrue(os.path.exists(clips[0]))
        self.assertTrue(os.path.exists(clips[1]))

        # Both clips must use the exact same locked wan_seed (not base_seed + idx)
        for idx, call in enumerate(mock_wan.call_args_list):
            wan_kwargs = call[1]
            self.assertEqual(wan_kwargs["current_seed"], locked_wan_seed)
            self.assertEqual(scenes[idx]["video_seed"], locked_wan_seed)
            self.assertEqual(wan_kwargs["num_frames"], 57)  # 4 * 14 + 1
            self.assertEqual(wan_kwargs["guidance_scale"], 2.0)
            self.assertEqual(wan_kwargs["guidance_scale_2"], WAN_DEFAULTS["guidance_scale_2"])
            self.assertEqual(wan_kwargs["steps"], WAN_DEFAULTS["steps"])
            self.assertIsInstance(wan_kwargs["quality"], int)
            self.assertEqual(wan_kwargs["quality"], 6)
            self.assertEqual(wan_kwargs["resized_image"].size, (480, 832))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 7: Video Assembly & Stitching with FFmpeg
    # ─────────────────────────────────────────────────────────────────────────

    def test_stitch_clips_ffmpeg(self):
        """Creates 2 real minimal MP4 video files with OpenCV and stitches them with FFmpeg."""
        clip1_path = os.path.join(self.temp_dir.name, "clip1.mp4")
        clip2_path = os.path.join(self.temp_dir.name, "clip2.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for p, color in [(clip1_path, (255, 0, 0)), (clip2_path, (0, 255, 0))]:
            out = cv2.VideoWriter(p, fourcc, 16.0, (160, 120))
            frame = np.full((120, 160, 3), color, dtype=np.uint8)
            for _ in range(16):
                out.write(frame)
            out.release()

        final_movie = self.pipeline.stitch_clips(
            clip_paths=[clip1_path, clip2_path],
            movie_title="Test_Kids_Short",
        )

        self.assertTrue(os.path.exists(final_movie))
        self.assertGreater(os.path.getsize(final_movie), 1000)

        cap = cv2.VideoCapture(final_movie)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertGreaterEqual(total_frames, 30)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 8: Full 6-Stage End-to-End Orchestrator + Manifest Persistence
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    @patch("studio_pipeline.qwen_image_edit_remote")
    @patch("studio_pipeline.generate_z_image_turbo_remote")
    @patch.object(StudioPipeline, "_chat_completion")
    def test_run_full_pipeline_6_stages_and_manifest(
        self, mock_chat, mock_z_img, mock_qwen, mock_wan
    ):
        # 1st chat call: Story Architect; 2nd chat call: Animation Director for scene 1
        mock_chat.side_effect = [
            json.dumps({
                "title": "Sharing Berry",
                "art_style": "2D cartoon",
                "theme": "Sharing is caring",
                "characters": [{"id": "char_1", "name": "Barnaby", "face": "round bear"}],
                "scenes": [{
                    "scene_number": 1,
                    "title": "The Berry",
                    "duration_seconds": 5.0,
                    "scenario": "Barnaby shares a berry",
                    "qwen_scene_prompt": "Barnaby in meadow with berry",
                    "wan_motion_prompt": "Barnaby offers berry",
                    "character_ids": ["char_1"],
                }],
            }),
            json.dumps({
                "action_summary": "Barnaby extends his paw with the berry",
                "qwen_end_frame_prompt": "Modify the characters in Image 1 into their dynamic ending pose at t=5.0s: Barnaby extends his paw offering the berry with a warm smile.",
            }),
            json.dumps({
                "shot_type": "medium shot",
                "camera_movement": "slow push-in",
                "motion_intensity": 2,
                "subject_action": "Barnaby offers berry to camera",
                "timeline": ["holds berry", "extends paw", "smiles warmly"],
                "wan_prompt": "Barnaby extends his paw offering a berry with a warm smile",
                "guidance_scale": 2.0,
                "steps": 8,
                "flow_shift": 3.0,
            }),
        ]

        char_pil = Image.new("RGB", (512, 512), color="brown")
        char_path = os.path.join(self.temp_dir.name, "barnaby.png")
        char_pil.save(char_path)
        mock_z_img.return_value = (char_pil, char_path)

        buf = io.BytesIO()
        Image.new("RGB", (576, 1024), color="green").save(buf, format="JPEG")
        mock_qwen.return_value = [buf.getvalue()]

        # Create real 1-second MP4 clip so FFmpeg stitch succeeds
        clip_path = os.path.join(self.temp_dir.name, "wan_clip.mp4")
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (160, 120))
        for _ in range(16):
            out.write(np.zeros((120, 160, 3), dtype=np.uint8))
        out.release()
        mock_wan.return_value = (clip_path, "task_1")

        story_data = self.pipeline.run_full_pipeline(
            user_prompt="Barnaby shares a berry",
            z_image_seed=111,
            qwen_seed=222,
            wan_seed=333,
            num_scenes=1,
            target_duration=5.0,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            art_style="2D cartoon",
            use_vision_director=True,
        )

        self.assertTrue(os.path.exists(story_data["final_movie_path"]))
        self.assertTrue(os.path.exists(story_data["manifest_path"]))
        self.assertEqual(story_data["z_image_seed"], 111)
        self.assertEqual(story_data["qwen_seed"], 222)
        self.assertEqual(story_data["wan_seed"], 333)
        self.assertEqual(story_data["scenes"][0]["director"]["source"], "gemma-vision")
        with open(story_data["manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["title"], "Sharing Berry")
        self.assertIn("director", manifest["scenes"][0])

    # ─────────────────────────────────────────────────────────────────────────
    # Test 9: 3-Seed Helper & Single-Item Redo Methods (Character, Scene, Clip)
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    @patch("studio_pipeline.qwen_image_edit_remote")
    @patch("studio_pipeline.generate_z_image_turbo_remote")
    def test_single_item_redos_and_three_seeds(self, mock_z_img, mock_qwen, mock_wan):
        from studio_pipeline import generate_three_studio_seeds

        seeds = generate_three_studio_seeds()
        self.assertIn("z_image_seed", seeds)
        self.assertIn("qwen_seed", seeds)
        self.assertIn("wan_seed", seeds)
        self.assertGreater(seeds["z_image_seed"], 0)
        self.assertGreater(seeds["qwen_seed"], 0)
        self.assertGreater(seeds["wan_seed"], 0)

        # Setup story_data with 2 characters and 2 scenes
        char_pil = Image.new("RGB", (256, 256), color="pink")
        char_path = os.path.join(self.temp_dir.name, "redo_char.png")
        char_pil.save(char_path)
        mock_z_img.return_value = (char_pil, char_path)

        buf = io.BytesIO()
        Image.new("RGB", (576, 1024), color="teal").save(buf, format="JPEG")
        mock_qwen.return_value = [buf.getvalue()]

        clip_path = os.path.join(self.temp_dir.name, "redo_clip.mp4")
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (160, 120))
        for _ in range(16):
            out.write(np.zeros((120, 160, 3), dtype=np.uint8))
        out.release()
        mock_wan.return_value = (clip_path, "task_redo")

        story_data = {
            "title": "Redo Story",
            "art_style": "2D cartoon",
            "characters": [
                {"id": "char_1", "name": "Luna", "face": "old face", "portrait_pil": char_pil, "portrait_path": char_path, "portrait_seed": 100},
                {"id": "char_2", "name": "Oliver", "face": "owl face", "portrait_pil": char_pil, "portrait_path": char_path, "portrait_seed": 100},
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "Scene 1",
                    "duration_seconds": 5.0,
                    "scenario": "old scenario",
                    "qwen_scene_prompt": "old qwen prompt",
                    "wan_motion_prompt": "old motion",
                    "camera_movement": "Static Tripod Shot",
                    "character_ids": ["char_1"],
                    "keyframe_pil": char_pil,
                    "keyframe_path": char_path,
                    "video_path": clip_path,
                },
                {
                    "scene_number": 2,
                    "title": "Scene 2",
                    "duration_seconds": 5.0,
                    "scenario": "scene 2",
                    "character_ids": ["char_2"],
                    "keyframe_pil": char_pil,
                    "keyframe_path": char_path,
                    "video_path": clip_path,
                },
            ],
        }

        # 1. Redo single character (index 1 -> Oliver) with new seed & clothing
        updated_char = self.pipeline.regenerate_single_character(
            story_data=story_data,
            char_index=1,
            updated_fields={"clothing": "emerald wizard cape", "z_image_prompt": ""},
            seed=55555,
        )
        self.assertEqual(updated_char["clothing"], "emerald wizard cape")
        self.assertEqual(updated_char["portrait_seed"], 55555)
        self.assertEqual(mock_z_img.call_args[1]["width"], 576)
        self.assertEqual(mock_z_img.call_args[1]["height"], 1024)
        self.assertIn("9:16 vertical full-body", mock_z_img.call_args[1]["prompt"])
        self.assertEqual(story_data["characters"][0]["portrait_seed"], 100)  # char 0 untouched!

        # 2. Redo single scene keyframe (index 0 -> Scene 1) with new prompt & seed
        updated_scene = self.pipeline.regenerate_single_scene_keyframe(
            story_data=story_data,
            scene_index=0,
            updated_fields={"scenario": "Luna finds a glowing crystal", "qwen_scene_prompt": "Luna holding crystal"},
            seed=66666,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
        )
        self.assertEqual(updated_scene["scenario"], "Luna finds a glowing crystal")
        self.assertEqual(updated_scene["keyframe_seed"], 66666)
        self.assertEqual(mock_qwen.call_args[1]["seed"], 66666)

        # 2b. Redo single scene with shot chaining from Scene 1
        updated_scene_2 = self.pipeline.regenerate_single_scene_keyframe(
            story_data=story_data,
            scene_index=1,
            seed=88888,
            chain_previous_end_frame=True,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
        )
        prev_kf = updated_scene.get("last_keyframe_path") or updated_scene.get("keyframe_path")
        self.assertEqual(updated_scene_2["keyframe_path"], prev_kf)

        # 3. Redo single 5s video clip (index 0 -> Scene 1) with new motion description & seed
        updated_clip_scene = self.pipeline.regenerate_single_scene_video(
            story_data=story_data,
            scene_index=0,
            updated_fields={"wan_motion_prompt": "Luna lifts the glowing crystal high into the air"},
            seed=77777,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            re_run_director=False,
            auto_stitch=True,
        )
        self.assertEqual(updated_clip_scene["video_seed"], 77777)
        self.assertEqual(mock_wan.call_args[1]["current_seed"], 77777)
        self.assertTrue(os.path.exists(story_data["final_movie_path"]))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 10: Gradio UI 4-Phase Wizard & Individual Redo Handlers in app.py
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    @patch("studio_pipeline.qwen_image_edit_remote")
    @patch("studio_pipeline.generate_z_image_turbo_remote")
    @patch.object(StudioPipeline, "_chat_completion")
    def test_app_studio_step_handlers(self, mock_chat, mock_z_img, mock_qwen, mock_wan):
        mock_chat.side_effect = [
            json.dumps({
                "title": "UI Step Test",
                "art_style": "2D cartoon",
                "characters": [{"id": "char_1", "name": "Pippin", "face": "bunny face"}],
                "scenes": [{"scene_number": 1, "title": "Hop", "duration_seconds": 3.5, "character_ids": ["char_1"]}],
            }),
            json.dumps({
                "action_summary": "Pippin finishes the cheerful hop",
                "qwen_end_frame_prompt": "Modify the characters in Image 1 into their dynamic ending pose at t=5.0s: Pippin lands from the cheerful hop with arms outstretched.",
            }),
            json.dumps({
                "shot_type": "wide shot",
                "camera_movement": "static tripod shot",
                "motion_intensity": 2,
                "wan_prompt": "Pippin hops cheerfully",
                "guidance_scale": 2.0,
                "steps": 8,
            }),
        ]

        char_pil = Image.new("RGB", (256, 256), color="white")
        char_path = os.path.join(self.temp_dir.name, "pippin.png")
        char_pil.save(char_path)
        mock_z_img.return_value = (char_pil, char_path)

        buf = io.BytesIO()
        Image.new("RGB", (576, 1024), color="cyan").save(buf, format="JPEG")
        mock_qwen.return_value = [buf.getvalue()]

        clip_path = os.path.join(self.temp_dir.name, "ui_clip.mp4")
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (160, 120))
        for _ in range(16):
            out.write(np.zeros((120, 160, 3), dtype=np.uint8))
        out.release()
        mock_wan.return_value = (clip_path, "task_ui")

        orig_init = StudioPipeline.__init__
        temp_base = self.temp_dir.name

        def patched_init(inst, api_key=None, model=None, endpoint_url=None, output_base_dir=None):
            orig_init(inst, api_key=api_key, model=model, endpoint_url=endpoint_url, output_base_dir=output_base_dir or temp_base)

        with patch.object(StudioPipeline, "__init__", patched_init):
            # Phase 1: Generate Story
            step1_res = app.studio_run_step1(
                "Pippin hops", "2D cartoon", 1, 3.5, "dummy_key", "gemma4:31b"
            )
            json_out, status1, state = step1_res[0], step1_res[1], step1_res[2]
            self.assertIn("Step 1 Complete", status1)
            self.assertEqual(json_out["title"], "UI Step Test")

            # Phase 2: Generate All Characters
            step2_res = app.studio_run_step2(101, "dummy_key", "gemma4:31b", state)
            char_gallery, status2, state = step2_res[0], step2_res[1], step2_res[2]
            self.assertIn("Step 2 Complete", status2)
            self.assertEqual(len(char_gallery), 1)

            # Phase 2 Individual Redo: Redo Character #1 with new seed 102
            redo_char_res = app.studio_redo_single_character(
                "1. Pippin (char_1)",
                "Pippin",
                "fluffy white bunny face",
                "long ears",
                "blue vest",
                "carrot badge",
                "",
                102,
                "dummy_key",
                "gemma4:31b",
                state,
            )
            char_gallery, char_preview, _, status_redo_char, state = redo_char_res
            self.assertIn("Character Redo Complete", status_redo_char)
            self.assertEqual(state["characters"][0]["portrait_seed"], 102)
            self.assertEqual(state["characters"][0]["clothing"], "blue vest")

            # Phase 3: Generate All Scene Images (Dual-Keyframe: Start + End Frame per scene)
            step3_res = app.studio_run_step3(
                "9:16 Vertical (Shorts/Reels)", 201, "dummy_key", "gemma4:31b", state
            )
            scene_gallery, status3, state = step3_res[0], step3_res[1], step3_res[2]
            self.assertIn("Step 3 Complete", status3)
            self.assertEqual(len(scene_gallery), 2)
            self.assertIsNotNone(state["scenes"][0].get("keyframe_path"))
            self.assertIsNotNone(state["scenes"][0].get("last_keyframe_path"))

            # Phase 3 Individual Redo: Redo Scene #1 with new seed 202
            redo_scene_res = app.studio_redo_single_scene(
                "Scene 1: Hop",
                "Hop in Meadow",
                "Pippin leaps over a sparkling brook",
                "Pippin jumping over brook in sunlit meadow",
                "char_1",
                202,
                "9:16 Vertical (Shorts/Reels)",
                "dummy_key",
                "gemma4:31b",
                state,
            )
            scene_gallery, scene_preview, scene_end_preview, _, status_redo_scene, state = redo_scene_res
            self.assertIn("Keyframe Redo Complete", status_redo_scene)
            self.assertEqual(state["scenes"][0]["keyframe_seed"], 202)
            self.assertIn(state["scenes"][0]["last_keyframe_seed"], [(202 + 10007) % (2**31 - 1), (202 + 10007 + 54321) % (2**31 - 1)])

            # Phase 4: Generate All Video Clips & Final Movie
            step4_res = app.studio_run_step4(
                "9:16 Vertical (Shorts/Reels)", 301, "dummy_key", "gemma4:31b", True, state
            )
            final_video, status5, state = step4_res[0], step4_res[1], step4_res[2]
            self.assertIn("Step 5 Complete", status5)
            self.assertTrue(os.path.exists(final_video))
            self.assertIn("director", state["scenes"][0])

            # Phase 4 Individual Redo: Redo Clip #1 with new seed 302
            redo_clip_res = app.studio_redo_single_clip(
                "Scene 1: Hop in Meadow",
                "Pippin does a joyful backflip and lands softly",
                "Slow Push-In",
                "Pippin backflip prompt",
                302,
                False,
                True,
                "9:16 Vertical (Shorts/Reels)",
                "dummy_key",
                "gemma4:31b",
                state,
            )
            clip_video, final_video2, _, status_redo_clip, state = redo_clip_res
            self.assertIn("Video Clip Redo Complete", status_redo_clip)
            self.assertEqual(state["scenes"][0]["video_seed"], 302)
            self.assertTrue(os.path.exists(final_video2))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 11: Dual-Keyframe (First + Last Frame) Anchor-Chained Synthesis & Selective Redo
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.qwen_image_edit_remote")
    def test_dual_keyframe_anchor_chained_generation_and_selective_redo(self, mock_qwen):
        """End Frame must use Start Frame as base_image + character portraits as ref_images."""
        start_buf = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(120, 160, 200)).save(start_buf, format="JPEG")
        end_buf = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(180, 110, 90)).save(end_buf, format="JPEG")
        mock_qwen.side_effect = [
            [start_buf.getvalue()],
            [end_buf.getvalue()],
            [end_buf.getvalue()],
            [start_buf.getvalue()],
        ]

        p1 = Image.new("RGB", (256, 256), color="red")
        chars = [{"id": "char_1", "name": "Barnaby", "face": "round bear face", "clothing": "red bowtie", "portrait_pil": p1}]
        scenes = [{
            "scene_number": 1,
            "title": "Wave",
            "scenario": "Barnaby waves hello",
            "qwen_scene_prompt": "Barnaby standing with paws at sides",
            "qwen_end_scene_prompt": "Barnaby raising right paw high in cheerful wave",
            "character_ids": ["char_1"],
        }]

        self.pipeline.generate_scene_keyframes(
            scenes=scenes,
            characters=chars,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            shared_seed=777,
            generate_end_frame=True,
        )

        # Call 1 (Start Frame): base_image=p1 (character portrait), ref_images=[]
        self.assertEqual(mock_qwen.call_count, 2)
        call1_kwargs = mock_qwen.call_args_list[0][1]
        self.assertIs(call1_kwargs["base_image"], p1)
        self.assertEqual(call1_kwargs["ref_images"], [])
        self.assertEqual(call1_kwargs["seed"], 777)

        # Call 2 (End Frame): base_image=Start Frame PIL (576x1024), ref_images=[p1]
        call2_kwargs = mock_qwen.call_args_list[1][1]
        self.assertEqual(call2_kwargs["base_image"].size, (576, 1024))
        self.assertEqual(len(call2_kwargs["ref_images"]), 1)
        self.assertIs(call2_kwargs["ref_images"][0], p1)
        self.assertEqual(call2_kwargs["seed"], (777 + 10007) % (2**31 - 1))
        self.assertTrue(os.path.exists(scenes[0]["keyframe_path"]))
        self.assertTrue(os.path.exists(scenes[0]["last_keyframe_path"]))

        # Selective Redo: which_frame="end" should make only 1 Qwen call (leaving keyframe_seed=777)
        story_data = {"art_style": "2D cartoon", "characters": chars, "scenes": scenes}
        self.pipeline.regenerate_single_scene_keyframe(
            story_data=story_data,
            scene_index=0,
            updated_fields={"qwen_end_scene_prompt": "Barnaby bowing politely"},
            seed=888,
            which_frame="end",
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
        )
        self.assertEqual(mock_qwen.call_count, 3)
        self.assertEqual(scenes[0]["keyframe_seed"], 777)
        self.assertEqual(scenes[0]["last_keyframe_seed"], 888)

        # Selective Redo: which_frame="start" should make only 1 Qwen call (updating keyframe_seed=999)
        self.pipeline.regenerate_single_scene_keyframe(
            story_data=story_data,
            scene_index=0,
            updated_fields={"qwen_scene_prompt": "Barnaby standing by oak tree"},
            seed=999,
            which_frame="start",
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
        )
        self.assertEqual(mock_qwen.call_count, 4)
        self.assertEqual(scenes[0]["keyframe_seed"], 999)
        self.assertEqual(scenes[0]["last_keyframe_seed"], 888)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 12: CIE-Lab Color & Luminance Matching (match_color_histogram_lab)
    # ─────────────────────────────────────────────────────────────────────────

    def test_match_color_histogram_lab_normalizes_exposure_and_tint(self):
        from studio_pipeline import match_color_histogram_lab

        # Create a warm bright reference Start Frame and a cool dark drifted End Frame
        rng = np.random.default_rng(42)
        ref_arr = np.clip(rng.normal(loc=[200, 160, 110], scale=20, size=(128, 128, 3)), 0, 255).astype(np.uint8)
        src_arr = np.clip(rng.normal(loc=[80, 100, 190], scale=25, size=(128, 128, 3)), 0, 255).astype(np.uint8)
        ref_pil = Image.fromarray(ref_arr, mode="RGB")
        src_pil = Image.fromarray(src_arr, mode="RGB")

        matched_pil = match_color_histogram_lab(src_pil, ref_pil, strength=0.9)
        self.assertEqual(matched_pil.size, ref_pil.size)
        self.assertEqual(matched_pil.mode, "RGB")

        matched_arr = np.array(matched_pil, dtype=np.float32)
        # Distance between channel means must be dramatically reduced after CIE-Lab matching
        orig_mean_diff = np.abs(src_arr.mean(axis=(0, 1)) - ref_arr.mean(axis=(0, 1))).sum()
        new_mean_diff = np.abs(matched_arr.mean(axis=(0, 1)) - ref_arr.mean(axis=(0, 1))).sum()
        self.assertLess(new_mean_diff, orig_mean_diff * 0.35)

    # ─────────────────────────────────────────────────────────────────────────
    # Test 13: Gemma 4: 31B Vision QA Critic & Auto-Retry Loop
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.qwen_image_edit_remote")
    @patch.object(StudioPipeline, "_chat_completion")
    def test_vlm_qa_critic_and_auto_self_correction_retry(self, mock_chat, mock_qwen):
        buf_start = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(50, 200, 50)).save(buf_start, format="JPEG")
        buf_end = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(200, 50, 50)).save(buf_end, format="JPEG")
        mock_qwen.side_effect = [
            [buf_start.getvalue()],
            [buf_end.getvalue()],
            [buf_start.getvalue()],
            [buf_end.getvalue()],
        ]

        # First QA check scores 5/10 (fails < 7) and suggests refined prompts; second QA check scores 9/10
        mock_chat.side_effect = [
            json.dumps({
                "score": 5,
                "passed": False,
                "issues": "Character missing red bowtie in end frame",
                "suggested_start_prompt": "Barnaby clearly wearing bright red bowtie at start",
                "suggested_end_prompt": "Barnaby waving while wearing bright red bowtie at end",
            }),
            json.dumps({
                "score": 9,
                "passed": True,
                "issues": "none",
                "suggested_start_prompt": "",
                "suggested_end_prompt": "",
            }),
        ]

        p1 = Image.new("RGB", (256, 256), color="brown")
        chars = [{"id": "char_1", "name": "Barnaby", "face": "round bear", "clothing": "red bowtie", "portrait_pil": p1}]
        scenes = [{
            "scene_number": 1,
            "title": "Bowtie Check",
            "scenario": "Barnaby waves",
            "qwen_scene_prompt": "Barnaby standing",
            "qwen_end_scene_prompt": "Barnaby waving",
            "character_ids": ["char_1"],
        }]

        self.pipeline.generate_scene_keyframes(
            scenes=scenes,
            characters=chars,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            shared_seed=500,
            generate_end_frame=True,
            auto_qa_retry=True,
            direct_end_frame=False,
        )

        # Initial attempt (2 Qwen calls with seed 500) + 1 retry (2 Qwen calls with seed 501) = 4 Qwen calls
        self.assertEqual(mock_qwen.call_count, 4)
        self.assertEqual(scenes[0]["keyframe_seed"], 501)
        self.assertEqual(scenes[0]["last_keyframe_seed"], (501 + 10007) % (2**31 - 1))
        self.assertEqual(scenes[0]["qa_report"]["score"], 9)
        self.assertTrue(scenes[0]["qa_report"]["passed"])
        self.assertIn("bright red bowtie", scenes[0]["qwen_scene_prompt"])
        self.assertIn("bright red bowtie", scenes[0]["qwen_end_scene_prompt"])

    # ─────────────────────────────────────────────────────────────────────────
    # Test 14: Dual-Frame Vision Director + Wan 2.2 FLF2V & 32fps RIFE 2x
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    @patch.object(StudioPipeline, "_chat_completion")
    def test_dual_frame_vision_director_and_wan_flf2v_with_rife_32fps(self, mock_chat, mock_wan):
        mock_chat.return_value = json.dumps({
            "shot_type": "medium shot",
            "camera_movement": "slow push-in",
            "motion_intensity": 2,
            "subject_action": "Barnaby transitions smoothly from standing to waving",
            "timeline": ["stands ready", "lifts right arm", "waves warmly"],
            "wan_prompt": "Barnaby lifts his right arm and waves warmly",
            "guidance_scale": 2.0,
            "steps": 8,
            "flow_shift": 3.0,
        })

        dummy_vid = pathlib.Path(self.temp_dir.name) / "flf2v_dummy.mp4"
        with open(dummy_vid, "wb") as f:
            f.write(b"\x00\x00\x00 flf2v mp4")
        mock_wan.return_value = (str(dummy_vid), "task_flf2v")

        start_kf = Image.new("RGB", (576, 1024), color="orange")
        end_kf = Image.new("RGB", (576, 1024), color="gold")
        scenes = [{
            "scene_number": 1,
            "title": "Dual Frame Scene",
            "duration_seconds": 5.0,
            "scenario": "Barnaby waves",
            "character_ids": ["char_1"],
            "keyframe_pil": start_kf,
            "last_keyframe_pil": end_kf,
        }]
        chars = [{"id": "char_1", "name": "Barnaby", "face": "bear face"}]

        # 1. Verify Gemma Vision receives BOTH Start Frame and End Frame images
        self.pipeline.direct_scene_animations(scenes=scenes, characters=chars, use_vision=True)
        sent_messages = mock_chat.call_args[0][0]
        user_content = sent_messages[1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 2)

        # 2. Verify Wan 2.2 receives processed_last_image (480x832) and frame_multiplier=32 (RIFE 2x)
        self.pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            base_seed=444444,
            use_last_frame=True,
            enable_rife_2x=True,
        )
        wan_kwargs = mock_wan.call_args[1]
        self.assertIsNotNone(wan_kwargs["processed_last_image"])
        self.assertEqual(wan_kwargs["processed_last_image"].size, (480, 832))
        self.assertEqual(wan_kwargs["frame_multiplier"], 32)

        # 3. Verify disabling use_last_frame and enable_rife_2x falls back to I2V (None) and 16fps
        self.pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            base_seed=444444,
            use_last_frame=False,
            enable_rife_2x=False,
        )
        wan_kwargs_i2v = mock_wan.call_args[1]
        self.assertIsNone(wan_kwargs_i2v["processed_last_image"])
        self.assertEqual(wan_kwargs_i2v["frame_multiplier"], 16)

    def test_restore_extracted_video_frame(self):
        """Verifies restore_extracted_video_frame edge deblocking, sharpening, and color matching."""
        # 1. Null handling
        self.assertIsNone(restore_extracted_video_frame(None))

        # 2. Synthetic frame restoration
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[25:75, 25:75] = [200, 50, 50]
        img = Image.fromarray(arr)

        restored = restore_extracted_video_frame(img, sharpen_strength=0.5)
        self.assertIsInstance(restored, Image.Image)
        self.assertEqual(restored.size, (100, 100))

        # 3. Restoration with reference image color histogram matching
        ref_arr = np.full((100, 100, 3), [50, 200, 50], dtype=np.uint8)
        ref_img = Image.fromarray(ref_arr)
        restored_with_ref = restore_extracted_video_frame(img, reference_pil=ref_img, color_match_strength=0.8)
        self.assertIsInstance(restored_with_ref, Image.Image)
        res_arr = np.array(restored_with_ref)
        # Green channel should have shifted towards reference
        self.assertGreater(res_arr[50, 50, 1], arr[50, 50, 1])

    def test_anti_noise_and_rotation_guard(self):
        """Verifies anti-noise parameter clamping for long clips and rotation identity anchors."""
        # 1. Long clip (>= 4.5s) without rotation: steps clamped >= 10, flow_shift >= 4.5, CFG <= 2.0
        cfg_long = {
            "prompt": "Barnaby walks gently forward",
            "negative_prompt": "blurry, low quality",
            "steps": 6,
            "flow_shift": 3.0,
            "guidance_scale": 3.5,
            "guidance_scale_2": 2.5,
        }
        guarded_long = StudioPipeline.apply_anti_noise_and_rotation_guard(
            cfg_long, duration_seconds=5.0, anti_noise_guard=True
        )
        self.assertEqual(guarded_long["steps"], 10)
        self.assertEqual(guarded_long["flow_shift"], 4.5)
        self.assertEqual(guarded_long["guidance_scale"], 2.0)
        self.assertEqual(guarded_long["guidance_scale_2"], 1.5)
        self.assertNotIn("[Rotation Identity Lock]", guarded_long["prompt"])

        # 2. Rotation prompt: injects geometric anchor and negative prompt suppression
        cfg_turn = {
            "prompt": "Luna turns around to face the camera",
            "negative_prompt": "blurry",
            "steps": 8,
            "flow_shift": 3.0,
            "guidance_scale": 1.8,
            "guidance_scale_2": 1.4,
        }
        guarded_turn = StudioPipeline.apply_anti_noise_and_rotation_guard(
            cfg_turn, duration_seconds=3.0, anti_noise_guard=True
        )
        self.assertIn("[Rotation Identity Lock]", guarded_turn["prompt"])
        self.assertIn("identity loss during rotation", guarded_turn["negative_prompt"])
        # For 3.0s, steps and flow_shift remain untouched if below 4.5s
        self.assertEqual(guarded_turn["steps"], 8)
        self.assertEqual(guarded_turn["flow_shift"], 3.0)

        # 3. Guard disabled: returns config unchanged
        unchanged = StudioPipeline.apply_anti_noise_and_rotation_guard(
            cfg_long, duration_seconds=5.0, anti_noise_guard=False
        )
        self.assertEqual(unchanged["steps"], 6)

    @patch.object(StudioPipeline, "_synthesize_single_scene_frame")
    def test_seamless_shot_chaining_scene_keyframes(self, mock_synth):
        """Verifies chain_previous_end_frame reuses previous scene end frame without extra generation."""
        img_s1_start = Image.new("RGB", (100, 100), color="blue")
        img_s1_end = Image.new("RGB", (100, 100), color="cyan")
        img_s2_end = Image.new("RGB", (100, 100), color="magenta")

        def side_effect(scene, characters, aspect_ratio, seed, generate_end_frame=True, which_frame="both", end_seed=None, **kwargs):
            num = scene.get("scene_number", 1)
            entry = dict(scene)
            if num == 1:
                entry["keyframe_pil"] = img_s1_start
                entry["last_keyframe_pil"] = img_s1_end
                entry["last_keyframe_path"] = "scene_1_end.jpg"
            elif num == 2:
                # When chaining, only 'end' frame should be synthesized
                self.assertEqual(which_frame, "end")
                entry["last_keyframe_pil"] = img_s2_end
                entry["last_keyframe_path"] = "scene_2_end.jpg"
            return entry

        mock_synth.side_effect = side_effect

        scenes = [
            {"scene_number": 1, "title": "Scene 1", "scenario": "Intro", "duration_seconds": 5.0, "character_ids": ["c1"]},
            {"scene_number": 2, "title": "Scene 2", "scenario": "Next action", "duration_seconds": 5.0, "character_ids": ["c1"]},
        ]
        chars = [{"id": "c1", "name": "Barnaby", "face": "bear"}]

        chained = self.pipeline.generate_scene_keyframes(
            scenes=scenes,
            characters=chars,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            generate_end_frame=True,
            chain_previous_end_frame=True,
        )

        self.assertEqual(len(chained), 2)
        # Scene 2's start frame must exactly be Scene 1's end frame
        self.assertEqual(chained[1]["keyframe_pil"], img_s1_end)
        self.assertEqual(chained[1]["last_keyframe_pil"], img_s2_end)
        # Only 2 synthesis calls were made (Scene 1 both + Scene 2 end), saving 50% on Scene 2
        self.assertEqual(mock_synth.call_count, 2)

    @patch.object(StudioPipeline, "_chat_completion")
    def test_direct_scene_end_frame_with_gemma_vision(self, mock_chat):
        """Verifies Gemma 4: 31B Vision inspects start frame and generates high-delta end pose command."""
        expected_prompt = (
            "Modify the characters in Image 1 into their dynamic ending pose at t=5.0s: "
            "Knight drops to one knee and holds shield forward defensively; Mage raises crystal staff radiating bright azure light. "
            "Keep the exact same background environment, trees, path, lighting, and character outfits from Image 1."
        )
        mock_chat.return_value = json.dumps({
            "action_summary": "Knight defends while mage casts spell",
            "qwen_end_frame_prompt": expected_prompt,
        })

        kf = Image.new("RGB", (256, 256), color="forestgreen")
        scene = {
            "scene_number": 1,
            "title": "Encounter",
            "scenario": "Knight and Mage explore the forest path",
            "wan_motion_prompt": "Knight raises shield, mage casts spell",
            "character_ids": ["c1", "c2"],
        }
        chars = [
            {"id": "c1", "name": "Knight", "clothing": "Silver armor"},
            {"id": "c2", "name": "Mage", "clothing": "Blue robes"},
        ]

        directed = self.pipeline.direct_scene_end_frame_with_gemma(
            scene=scene,
            keyframe_pil=kf,
            characters=chars,
            art_style="Cinematic Anime",
            use_vision=True,
        )

        self.assertEqual(directed, expected_prompt)
        # Verify vision payload with base64 data URL was sent
        sent_messages = mock_chat.call_args[0][0]
        user_content = sent_messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    @patch.object(StudioPipeline, "_chat_completion")
    def test_direct_scene_end_frame_with_gemma_fallback(self, mock_chat):
        """Verifies fallback produces a deterministic high-delta prompt if Gemma LLM call raises."""
        mock_chat.side_effect = RuntimeError("Ollama Vision timeout")
        kf = Image.new("RGB", (256, 256), color="red")
        scene = {
            "scene_number": 2,
            "title": "Confrontation",
            "scenario": "Heroes face the shadowy guardian",
            "wan_motion_prompt": "Heroes ready their weapons in combat stance",
            "character_ids": ["c1"],
        }
        chars = [{"id": "c1", "name": "Knight", "clothing": "Armor"}]

        directed = self.pipeline.direct_scene_end_frame_with_gemma(
            scene=scene,
            keyframe_pil=kf,
            characters=chars,
            art_style="Cinematic Anime",
            use_vision=True,
        )

        self.assertIsNotNone(directed)
        self.assertIn("Ending frame at t=5.0s", directed)
        self.assertIn("Heroes ready their weapons in combat stance", directed)

    def test_measure_frame_pose_delta(self):
        """Verifies pose delta calculation correctly identifies identical vs distinct keyframes."""
        from studio_pipeline import measure_frame_pose_delta
        im1 = Image.new("RGB", (100, 100), color=(100, 100, 100))
        im2 = Image.new("RGB", (100, 100), color=(100, 100, 100))
        # Identical images must yield 0.0 delta
        self.assertEqual(measure_frame_pose_delta(im1, im2), 0.0)

        # Subtle noise / near identical (delta < 18.0)
        im_subtle = Image.new("RGB", (100, 100), color=(105, 105, 105))
        self.assertLess(measure_frame_pose_delta(im1, im_subtle), 18.0)

        # Clear pose transformation (high delta >= 18.0)
        im3 = Image.new("RGB", (100, 100), color=(200, 50, 50))
        delta = measure_frame_pose_delta(im1, im3)
        self.assertGreaterEqual(delta, 18.0)

    def test_ensure_high_delta_end_prompt(self):
        """Verifies prompt validator replaces stagnant end prompts with concrete contrasting postures."""
        from studio_pipeline import ensure_high_delta_end_prompt

        start = "Mercenaries aiming weapons forward into the dark cave"
        # 1. Identical prompt -> auto-enhanced
        fixed1 = ensure_high_delta_end_prompt(start, start, scenario="Blind Fire", motion="firing weapons")
        self.assertNotEqual(fixed1, start)
        self.assertIn("Ending frame at t=5.0s", fixed1)
        self.assertTrue(any(kw in fixed1.lower() for kw in ["duck", "crouch", "kneel", "recoil", "cover"]))

        # 2. Too similar prompt (high overlap, missing posture keyword) -> auto-enhanced
        similar = "Mercenaries continue aiming forward in the cave"
        fixed2 = ensure_high_delta_end_prompt(start, similar, scenario="Blind Fire", motion="firing weapons")
        self.assertIn("Ending frame at t=5.0s", fixed2)

        # 3. Genuinely distinct prompt with posture keyword -> preserved
        distinct = "Mercenaries drop to one knee taking cover behind rock crates with rifles lowered"
        preserved = ensure_high_delta_end_prompt(start, distinct)
        self.assertEqual(preserved, distinct)

    @patch("studio_pipeline.qwen_image_edit_remote")
    def test_pose_delta_gatekeeper_triggers_retry_on_low_delta(self, mock_qwen):
        """Verifies that if Qwen returns an identical end frame (delta < 18.0), the gatekeeper retries with high delta."""
        # Start frame: green
        img_start = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(50, 150, 50)).save(img_start, format="JPEG")

        # First end frame: almost identical green (mean diff ~2 < 18.0) -> triggers gatekeeper retry
        img_low_delta = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(52, 152, 52)).save(img_low_delta, format="JPEG")

        # Second end frame (retry): distinct blue/red pose
        img_retry = io.BytesIO()
        Image.new("RGB", (576, 1024), color=(200, 50, 50)).save(img_retry, format="JPEG")

        # Mock sequence: 1st call = start frame, 2nd call = stagnant end frame, 3rd call = retry end frame
        mock_qwen.side_effect = [
            [img_start.getvalue()],
            [img_low_delta.getvalue()],
            [img_retry.getvalue()],
        ]

        scene = {
            "scene_number": 1,
            "title": "Blind Fire",
            "scenario": "Three soldiers fire rifles",
            "qwen_scene_prompt": "Three soldiers aiming rifles forward",
            "qwen_end_scene_prompt": "Three soldiers continue aiming rifles",
            "character_ids": ["c1"],
        }
        chars = [{"id": "c1", "name": "Soldier", "portrait_pil": Image.new("RGB", (64, 64))}]

        res = self.pipeline._synthesize_single_scene_frame(
            scene=scene,
            characters=chars,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            seed=12345,
            which_frame="both",
            direct_end_frame=False,
        )

        # Verify retry was executed (3 calls total: start, initial end, retry end)
        self.assertEqual(mock_qwen.call_count, 3)
        self.assertGreaterEqual(res["pose_delta"], 18.0)
        # End seed must be different from start seed
        self.assertNotEqual(res["last_keyframe_seed"], 12345)

    def test_cinematography_vault_blueprint_retrieval(self):
        """Verifies semantic & rule-based retrieval of lens optics, rigs, and reactive physics."""
        from cinematography_vault import retrieve_cinematography_blueprint, format_period_sequenced_wan_prompt

        # Test forest environment & Steadicam matching
        bp_forest = retrieve_cinematography_blueprint(
            scenario="Knight rides horse through the pine forest grove",
            camera_movement="Steadicam tracking",
            art_style="Cinematic",
        )
        self.assertEqual(bp_forest["lens"], "35mm wide-angle cinema lens")
        self.assertIn("Leaves rustle", bp_forest["reactive_physics"])

        # Test magic and orbital tracking matching
        bp_magic = retrieve_cinematography_blueprint(
            scenario="Mage casts glowing spell in ancient stone room",
            camera_movement="orbital tracking shot",
        )
        self.assertEqual(bp_magic["lens"], "85mm telephoto portrait lens")
        self.assertIn("Luminous energy", bp_magic["reactive_physics"])

        # Test format_period_sequenced_wan_prompt
        prompt = format_period_sequenced_wan_prompt(
            camera_directive="Slow push-in on 50mm lens",
            timeline_steps=["Hero draws sword", "Hero steps forward", "Hero settles in stance"],
            environment_physics="Pine needles scatter underfoot",
            identity_anchors="Hero silver armor",
            art_style="2D Anime",
        )
        self.assertIn("Slow push-in on 50mm lens.", prompt)
        self.assertIn("Hero draws sword.", prompt)
        self.assertIn("Pine needles scatter underfoot.", prompt)
        self.assertIn("Hero silver armor", prompt)
        self.assertIn("Rigid background geometry", prompt)


if __name__ == "__main__":
    unittest.main()

