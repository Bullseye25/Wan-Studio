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

            # Phase 3: Generate All Scene Images
            step3_res = app.studio_run_step3(
                "9:16 Vertical (Shorts/Reels)", 201, "dummy_key", "gemma4:31b", state
            )
            scene_gallery, status3, state = step3_res[0], step3_res[1], step3_res[2]
            self.assertIn("Step 3 Complete", status3)
            self.assertEqual(len(scene_gallery), 1)

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
            scene_gallery, scene_preview, _, status_redo_scene, state = redo_scene_res
            self.assertIn("Keyframe Redo Complete", status_redo_scene)
            self.assertEqual(state["scenes"][0]["keyframe_seed"], 202)

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


if __name__ == "__main__":
    unittest.main()
