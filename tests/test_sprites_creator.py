import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

import sprite_cutter
from sprite_cutter import (
    remove_background_alpha,
    find_union_bounding_box,
    crop_frames_to_union_bbox,
    pack_sprite_sheet,
    generate_engine_metadata,
    process_video_to_sprite_sheet,
)
from sprites_pipeline import SpritesPipeline, GENRE_PRESETS
import app


class TestSpriteCutter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_remove_background_alpha(self):
        # Create a 64x64 green background (#00FF00) with a red square (character) in the center
        img = Image.new("RGB", (64, 64), (0, 255, 0))
        # Draw a 20x20 red square in the middle
        for x in range(22, 42):
            for y in range(22, 42):
                img.putpixel((x, y), (255, 0, 0))

        rgba = remove_background_alpha(img, bg_color=(0, 255, 0), tolerance=35.0)
        self.assertEqual(rgba.mode, "RGBA")
        
        # Corners should be fully transparent (alpha == 0)
        corner_alpha = rgba.getpixel((2, 2))[3]
        self.assertEqual(corner_alpha, 0)
        
        # Center of red square should be fully opaque (alpha == 255)
        center_alpha = rgba.getpixel((32, 32))[3]
        self.assertEqual(center_alpha, 255)
        self.assertEqual(rgba.getpixel((32, 32))[:3], (255, 0, 0))

    def test_find_union_bounding_box(self):
        # Frame 1: sprite at (10, 10, 30, 30)
        f1 = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        for x in range(10, 30):
            for y in range(10, 30):
                f1.putpixel((x, y), (255, 255, 255, 255))

        # Frame 2: sprite at (20, 20, 50, 50)
        f2 = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        for x in range(20, 50):
            for y in range(20, 50):
                f2.putpixel((x, y), (255, 255, 255, 255))

        bbox = find_union_bounding_box([f1, f2], padding=0)
        self.assertEqual(bbox, (10, 10, 50, 50))

        cropped, cbounds = crop_frames_to_union_bbox([f1, f2], padding=2)
        self.assertEqual(len(cropped), 2)
        # Bounding box should include padding: (8, 8, 52, 52) -> width 44, height 44
        self.assertEqual(cropped[0].size, (44, 44))
        self.assertEqual(cropped[1].size, (44, 44))

    def test_pack_sprite_sheet(self):
        frames = [
            Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 255, 0, 255)),
            Image.new("RGBA", (32, 32), (0, 0, 255, 255)),
            Image.new("RGBA", (32, 32), (255, 255, 0, 255)),
        ]
        sheet, metadata = pack_sprite_sheet(frames, max_cols=2, border_padding=0, shape_padding=0)
        self.assertEqual(sheet.mode, "RGBA")
        self.assertEqual(sheet.size, (64, 64))
        self.assertEqual(len(metadata), 4)
        self.assertEqual(metadata[0]["frame_index"], 0)
        self.assertEqual(metadata[0]["x"], 0)
        self.assertEqual(metadata[0]["y"], 0)
        self.assertEqual(metadata[1]["x"], 32)
        self.assertEqual(metadata[1]["y"], 0)
        self.assertEqual(metadata[2]["x"], 0)
        self.assertEqual(metadata[2]["y"], 32)

    def test_generate_engine_metadata(self):
        frame_coords = [
            {"frame_index": 0, "x": 0, "y": 0, "w": 32, "h": 32},
            {"frame_index": 1, "x": 32, "y": 0, "w": 32, "h": 32},
        ]
        meta = generate_engine_metadata(
            character_name="Pepsiman",
            animation_name="Run Forward",
            genre="Action Runner",
            fps=12,
            is_loop=True,
            sheet_size=(64, 32),
            frame_coords=frame_coords,
        )
        self.assertEqual(meta["character"], "Pepsiman")
        self.assertEqual(meta["animation"], "Run Forward")
        self.assertEqual(meta["fps"], 12)
        self.assertTrue(meta["is_loop"])
        self.assertEqual(meta["total_frames"], 2)
        self.assertEqual(len(meta["frames"]), 2)
        self.assertIn("unity", meta["frames"][0])
        self.assertIn("godot", meta["frames"][0])


class TestSpritesPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = SpritesPipeline()

    def test_genre_presets(self):
        genres = list(GENRE_PRESETS.keys())
        self.assertIn("Action Runner (Pepsiman / Crash Bandicoot)", genres)
        self.assertIn("Action Platformer (Mega Man X / Hollow Knight)", genres)
        self.assertIn("Top-Down RPG (Zelda / Chrono Trigger)", genres)
        self.assertIn("Arcade Shmup (Space Invaders / Gradius)", genres)
        self.assertIn("Beat 'em Up (Streets of Rage / Final Fight)", genres)
        self.assertIn("Custom / Hybrid Genre", genres)

        # Check default moves
        moves = SpritesPipeline.get_default_moves_for_genre("Action Runner (Pepsiman / Crash Bandicoot)")
        self.assertTrue(any(m["name"] == "Full Sprint" and m["type"] == "loop" for m in moves))
        self.assertTrue(any(m["name"] == "Hurdle Slide" and m["type"] == "action" for m in moves))
        self.assertTrue(any(m["name"] == "Can Chug / Victory" and m["type"] == "action" for m in moves))

    @patch.object(SpritesPipeline, "_chat_completion")
    def test_refine_character_with_gemma(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "name": "Spark_Knight",
            "title": "Arcade Defender",
            "appearance": "Metallic azure armor with lightning emblems",
            "signature_gear": "Plasma Buster Gauntlet",
            "z_image_prompt": "2D game sprite of Spark_Knight on chroma key green #00FF00",
            "idle_stance": "Ready bounce posture",
        })
        res = self.pipeline.refine_character_with_gemma(
            user_input="Blue armored hero with buster",
            genre="Action Platformer (Mega Man X / Hollow Knight)",
            art_style="16-bit arcade pixel art style",
        )
        self.assertEqual(res["name"], "Spark_Knight")
        self.assertIn("z_image_prompt", res)
        self.assertIn("chroma key green", res["z_image_prompt"].lower())

    @patch.object(SpritesPipeline, "_chat_completion")
    def test_refine_custom_animation_with_gemma(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "animation_name": "Dragon Uppercut",
            "motion_type": "action",
            "action_summary": "Character crouches, leaps upward with flame fist, and recovers to stance",
            "wan_prompt": "Isolated 2D sprite of Hero executing a Dragon Uppercut, zero camera pan",
        })
        char_data = {"name": "Hero", "genre": "Beat 'em Up"}
        plan = self.pipeline.refine_custom_animation_with_gemma(
            character_data=char_data,
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
            anim_name="Dragon Uppercut",
            raw_description="Leap up with flaming fist",
            anim_type="action",
        )
        self.assertEqual(plan["animation_name"], "Dragon Uppercut")
        self.assertEqual(plan["motion_type"], "action")
        self.assertIn("Dragon Uppercut", plan["wan_prompt"])

    @patch("sprites_pipeline.run_remote_inference")
    def test_synthesize_animation_clip_loop_and_action(self, mock_wan):
        mock_wan.return_value = ("output/test_video.mp4", 1234)
        dummy_img = Image.new("RGB", (64, 64), (0, 255, 0))

        char_data = {
            "name": "TestHero",
            "master_sprite_pil": dummy_img,
            "genre": "Action Runner (Pepsiman / Crash Bandicoot)",
            "art_style": "16-bit arcade pixel art style",
        }

        # 1. Test Loop move
        loop_move = {
            "name": "Sprint Run",
            "type": "loop",
            "refined_prompt": "Hero sprinting continuously forward in infinite seamless loop",
        }
        res_video = self.pipeline.synthesize_animation_clip(char_data, loop_move, wan_seed=999)
        self.assertEqual(res_video, "output/test_video.mp4")
        # In loop mode, last_frame_image should be None
        _, kwargs = mock_wan.call_args
        self.assertIsNone(kwargs["last_frame_image"])

        # 2. Test Action move (must return to idle stance via FLF2V anchor)
        action_move = {
            "name": "Can Chug Quench Recovery",
            "type": "action",
            "refined_prompt": "Hero drinks a cold can and returns to idle",
        }
        res_video2 = self.pipeline.synthesize_animation_clip(char_data, action_move, wan_seed=1001)
        self.assertEqual(res_video2, "output/test_video.mp4")
        # In action mode, last_frame_image should be anchored to master_sprite_pil
        _, kwargs2 = mock_wan.call_args
        self.assertIsNotNone(kwargs2["last_frame_image"])


class TestSpritesUIHandlers(unittest.TestCase):
    @patch("sprites_pipeline.SpritesPipeline.synthesize_master_character_sprite")
    def test_sprites_ui_generate_master_sprite(self, mock_synth):
        dummy_img = Image.new("RGB", (64, 64), (0, 255, 0))
        mock_synth.return_value = (dummy_img, "output/master_hero.png")

        state = {}
        img_path, sel1, sel2, sel3, moves, status, out_state = app.sprites_ui_generate_master_sprite(
            name="PepsiHero",
            title="Runner",
            appearance="Metallic silver and cobalt blue suit",
            gear="Refreshing Soda Can",
            z_prompt="2D game sprite of PepsiHero on solid chroma key green",
            seed=42,
            genre="Action Runner (Pepsiman / Crash Bandicoot)",
            art_style="16-bit arcade pixel art style",
            api_key=None,
            model=None,
            current_state=state,
        )
        self.assertEqual(img_path, "output/master_hero.png")
        self.assertIn("PepsiHero", status)
        self.assertIn("character", out_state)
        self.assertEqual(len(moves), len(SpritesPipeline.get_default_moves_for_genre("Action Runner (Pepsiman / Crash Bandicoot)")))

    def test_sprites_ui_add_custom_anim_and_inspect(self):
        char_data = {
            "name": "PepsiHero",
            "genre": "Action Runner (Pepsiman / Crash Bandicoot)",
            "moves": [
                {"id": "m1", "name": "Idle Stance", "type": "loop", "description": "Resting bounce", "refined_prompt": "Idle"},
            ],
        }
        state = {"character": char_data}

        # Add custom animation
        s1, s2, s3, moves, status, out_state = app.sprites_ui_add_custom_anim(
            char_state=state,
            custom_name="Mega Can Wave",
            custom_desc="Throws huge soda wave forward and resets to stance",
            custom_type="🎯 Action (Return to Idle Stance)",
            refined_prompt="Action sprite executing mega can wave returning to idle",
        )
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[1]["name"], "Mega Can Wave")
        self.assertEqual(moves[1]["type"], "action")
        self.assertIn("Mega Can Wave", status)

        # Inspect details of custom move
        info, vpath = app.sprites_ui_load_move_details("Mega Can Wave (ACTION)", out_state)
        self.assertIn("Mega Can Wave", info)
        self.assertIn("Action & Combo", info)


if __name__ == "__main__":
    unittest.main()
