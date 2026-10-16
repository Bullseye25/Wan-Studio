import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from pathlib import Path

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

    def test_remove_white_background_alpha_with_vfx_and_interior_white(self):
        """
        Tests pure white studio background (#FFFFFF) keying:
        1. Exterior white background is made transparent (alpha == 0).
        2. Interior white features (e.g. white eyes/shirt/teeth) inside the character are NOT hollowed out (alpha == 255).
        3. Semi-transparent glowing VFX (e.g. energy slash / magic trail) preserves smooth alpha (0 < alpha < 255).
        4. White defringing removes white background bleed from the VFX.
        """
        # 64x64 solid pure white studio background
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        
        # Draw character body (dark armor/costume) from (20, 20) to (44, 44)
        for x in range(20, 45):
            for y in range(20, 45):
                img.putpixel((x, y), (40, 40, 60))
                
        # Draw interior pure white shirt/chest emblem inside the character from (28, 28) to (36, 36)
        for x in range(28, 37):
            for y in range(28, 37):
                img.putpixel((x, y), (255, 255, 255))

        # Draw semi-transparent cyan magic slash VFX extending to the right from (45, 28) to (55, 36)
        # composited over white (e.g. 50% opacity cyan (0, 240, 255) over (255, 255, 255) = (128, 248, 255))
        for x in range(45, 56):
            for y in range(28, 37):
                img.putpixel((x, y), (128, 248, 255))

        rgba = remove_background_alpha(img, bg_color=(255, 255, 255), tolerance=38.0, defringe=True, preserve_vfx=True)
        self.assertEqual(rgba.mode, "RGBA")

        # 1. Exterior canvas corners must be 100% transparent (alpha == 0)
        self.assertEqual(rgba.getpixel((2, 2))[3], 0)
        self.assertEqual(rgba.getpixel((60, 2))[3], 0)
        self.assertEqual(rgba.getpixel((2, 60))[3], 0)
        self.assertEqual(rgba.getpixel((60, 60))[3], 0)

        # 2. Character body must be fully opaque (alpha == 255)
        self.assertEqual(rgba.getpixel((22, 22))[3], 255)

        # 3. INTERIOR WHITE FEATURE (shirt/emblem) must be PROTECTED and fully opaque (alpha == 255)
        # (Must NOT be hollowed out or transparent!)
        self.assertEqual(rgba.getpixel((32, 32))[3], 255)
        self.assertEqual(rgba.getpixel((32, 32))[:3], (255, 255, 255))

        # 4. VFX REGION: glowing cyan slash must preserve semi-transparency (0 < alpha < 255)
        vfx_alpha = rgba.getpixel((50, 32))[3]
        self.assertGreater(vfx_alpha, 0)
        self.assertLess(vfx_alpha, 255)

        # 5. Anti-halo Defringing: the cyan color should have white un-premultiplied
        vfx_rgb = rgba.getpixel((50, 32))[:3]
        # Red channel should be lower than green/blue (defringed back towards pure cyan)
        self.assertLess(vfx_rgb[0], vfx_rgb[1])
        self.assertGreater(vfx_rgb[1], 200)

    def test_auto_detect_white_background(self):
        img_np = np.full((64, 64, 3), 255, dtype=np.uint8)
        img_np[20:40, 20:40] = [100, 50, 20]
        detected = sprite_cutter.auto_detect_background_color(img_np)
        np.testing.assert_array_equal(detected, [255, 255, 255])

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
            "z_image_prompt": "2D game sprite of Spark_Knight on pure flat solid white studio background #FFFFFF",
            "idle_stance": "Ready bounce posture",
        })
        res = self.pipeline.refine_character_with_gemma(
            user_input="Blue armored hero with buster",
            genre="Action Platformer (Mega Man X / Hollow Knight)",
            art_style="16-bit arcade pixel art style",
        )
        self.assertEqual(res["name"], "Spark_Knight")
        self.assertIn("z_image_prompt", res)
        self.assertIn("white", res["z_image_prompt"].lower())

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

        # 1. Test Loop move (dual-keyframe closed-loop anchoring for zero frame skipping)
        loop_move = {
            "name": "Sprint Run",
            "type": "loop",
            "refined_prompt": "Hero sprinting continuously forward in infinite seamless loop",
        }
        with patch.object(self.pipeline, "synthesize_stride_keyframe_with_qwen", return_value=dummy_img):
            res_video = self.pipeline.synthesize_animation_clip(char_data, loop_move, wan_seed=999)
        self.assertEqual(res_video, "output/test_video.mp4")
        # In loop mode, last_frame_image is anchored to the starting stride keyframe for zero frame skipping
        _, kwargs = mock_wan.call_args
        self.assertIsNotNone(kwargs["last_frame_image"])
        self.assertEqual(kwargs["resized_image"], kwargs["last_frame_image"])

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

    def test_sword_character_never_gets_boxing_moves(self):
        """Ensure a character wielding a sword in Beat 'em Up or Platformer gets sword moves, never boxing jabs."""
        sword_char = {
            "name": "Kensei_Valen",
            "title": "Crimson Ronin",
            "appearance": "Samurai warrior in obsidian armor holding a glowing katana sword",
            "signature_gear": "Crimson Katana Sword",
        }
        arch = SpritesPipeline.detect_character_combat_archetype(
            sword_char, genre="Beat 'em Up (Streets of Rage / Final Fight)"
        )
        self.assertEqual(arch["archetype"], "blade")
        self.assertIn("katana", arch["weapon_label"].lower())

        moves = SpritesPipeline.get_smart_contextual_moves(
            sword_char, genre="Beat 'em Up (Streets of Rage / Final Fight)"
        )
        self.assertGreaterEqual(len(moves), 5)
        move_names_blob = " ".join(m["name"].lower() for m in moves)
        move_descs_blob = " ".join(m["description"].lower() for m in moves)

        # Must have sword/blade choreography
        self.assertTrue("sword" in move_names_blob or "slash" in move_names_blob or "blade" in move_names_blob)
        # Must NOT have boxing jab or arm cannon in move names or descriptions
        self.assertNotIn("jab-cross-hook", move_names_blob)
        self.assertNotIn("boxing", move_descs_blob)
        self.assertNotIn("arm cannon", move_descs_blob)

        # Every move must have pre-animation Wan 2.2 instructions with timeline & weapon invariants
        for m in moves:
            self.assertIn("[Character & Weapon Invariants]:", m["refined_prompt"])
            self.assertIn("[Pre-Animation Timeline]:", m["refined_prompt"])
            self.assertIn("Crimson Katana Sword", m["refined_prompt"])

    @patch.object(SpritesPipeline, "_chat_completion")
    def test_generate_smart_moveset_with_gemma(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "combat_archetype_summary": "Master swordsman using rapid katana iaijutsu slashes",
            "moves": [
                {
                    "id": "sword_idle",
                    "name": "Katana Guard Idle",
                    "type": "loop",
                    "description": "[Start] Poised two-handed katana guard -> [Peak] Subtle breathing and blade glint -> [Cycle] Seamless loop.",
                    "wan_prompt": "2D sprite of Ronin in katana guard idle on chroma key green #00FF00",
                },
                {
                    "id": "sword_sprint",
                    "name": "Blade-Low Dash",
                    "type": "loop",
                    "description": "[Start] Low forward sprint with katana held at hip -> [Peak] Full stride extension -> [Cycle] Seamless loop.",
                    "wan_prompt": "2D sprite of Ronin sprinting with katana on chroma key green #00FF00",
                },
                {
                    "id": "iaijutsu_slash",
                    "name": "Iaijutsu Cross Slash",
                    "type": "action",
                    "description": "[Anticipation] Crouch hand on hilt -> [Strike] Lightning cross slash -> [Recovery] Sheathe and return to exact Frame 1 guard.",
                    "wan_prompt": "2D sprite of Ronin executing Iaijutsu Cross Slash returning to idle on chroma key green #00FF00",
                },
            ],
        })
        pipeline = SpritesPipeline(api_key="test_key")
        sword_char = {
            "name": "Ronin",
            "appearance": "Samurai with katana sword",
            "signature_gear": "Katana Sword",
        }
        moves = pipeline.generate_smart_moveset_with_gemma(
            character_data=sword_char,
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
            combat_notes="Fast iaijutsu sword slashes",
            use_llm=True,
        )
        self.assertGreaterEqual(len(moves), 3)
        self.assertEqual(moves[0]["name"], "Katana Guard Idle")
        self.assertEqual(moves[0]["type"], "loop")
        self.assertEqual(moves[2]["name"], "Iaijutsu Cross Slash")
        self.assertEqual(moves[2]["type"], "action")
        self.assertIn("Iaijutsu Cross Slash", moves[2]["refined_prompt"])

    def test_phase2_smart_choreograph_and_move_editor_handlers(self):
        state = {
            "character": {
                "name": "Arthur",
                "title": "Paladin",
                "appearance": "Knight in silver plate armor",
                "signature_gear": "Excalibur Broadsword & Kite Shield",
                "genre": "Beat 'em Up (Streets of Rage / Final Fight)",
            }
        }
        # 1. Trigger Smart Choreograph Move-Set
        (
            sel1, sel2, sel3, moves,
            edit_name, edit_type, edit_desc, edit_prompt,
            status, state
        ) = app.sprites_ui_smart_choreograph_moveset(
            char_state=state,
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
            combat_notes="Heavy broadsword slashes and shield bash",
            name="Arthur",
            title="Paladin",
            appearance="Knight in silver plate armor",
            gear="Excalibur Broadsword & Kite Shield",
            api_key=None,
            model=None,
        )
        self.assertGreaterEqual(len(moves), 5)
        self.assertIn("Sword", edit_name)
        self.assertIn("Excalibur Broadsword", edit_prompt)
        self.assertIn("Smart Choreography Complete", status)

        # 2. Save manual edits to a move's pre-animation instructions
        target_label = f"{moves[1]['name']} ({moves[1]['type'].upper()})"
        s1, s2, s3, updated_moves, save_status, state = app.sprites_ui_save_selected_move(
            choice_label=target_label,
            edit_name="Excalibur Sky Cleave",
            edit_type="🎯 Action (Return to Idle Stance)",
            edit_desc="[Anticipation] Raise broadsword overhead -> [Apex] Cleave downward with golden arc -> [Recovery] Return to Frame 1 sword guard.",
            edit_prompt="",
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
            char_state=state,
        )
        self.assertEqual(updated_moves[1]["name"], "Excalibur Sky Cleave")
        self.assertIn("[Pre-Animation Timeline]:", updated_moves[1]["refined_prompt"])
        self.assertIn("Saved Pre-Animation Instructions", save_status)

    @patch("sprites_pipeline.run_remote_inference")
    def test_phase3_animate_move_and_wizard_transitions(self, mock_wan):
        mock_wan.return_value = ("output/wan_move_clip.mp4", 9999)
        dummy_img = Image.new("RGB", (64, 64), (0, 255, 0))

        state = {
            "character": {
                "name": "Arthur",
                "master_sprite_pil": dummy_img,
                "master_sprite_path": "output/arthur_master.png",
                "genre": "Beat 'em Up (Streets of Rage / Final Fight)",
                "moves": [
                    {
                        "id": "move_1",
                        "name": "Sword Slash",
                        "type": "action",
                        "description": "Clean sword slash",
                        "refined_prompt": "Arthur executes sword slash on chroma key green",
                    }
                ],
            }
        }

        # 1. Test Phase 2 -> Phase 3 transition
        tab_update, move_drop, info, vpath = app.sprites_ui_transition_phase2_to_phase3(
            char_state=state, current_choice="Sword Slash (ACTION)"
        )
        self.assertEqual(tab_update["selected"], "sprites_phase3")
        self.assertEqual(move_drop["value"], "Sword Slash (ACTION)")
        self.assertIn("Sword Slash", info)

        # 2. Test Phase 3: Animate Selected Move
        v_out, cut_drop, anim_info, status, out_state = app.sprites_ui_animate_move(
            choice_label="Sword Slash (ACTION)",
            anim_seed=42,
            char_state=state,
        )
        self.assertEqual(v_out, "output/wan_move_clip.mp4")
        self.assertIn("wan_move_clip.mp4", anim_info)
        self.assertIn("Synthesized Move Animation", status)
        self.assertEqual(out_state["character"]["moves"][0]["video_path"], "output/wan_move_clip.mp4")

        # 3. Test Phase 3 -> Phase 4 transition
        tab_update4, cut_drop4 = app.sprites_ui_transition_phase3_to_phase4(
            char_state=out_state, current_choice="Sword Slash (ACTION)"
        )
        self.assertEqual(tab_update4["selected"], "sprites_phase4")
        self.assertEqual(cut_drop4["value"], "Sword Slash (ACTION)")

    def test_multi_weapon_and_damage_tier_parsing(self):
        """Tests parsing complex multi-weapon characters with asymmetric damage tiers."""
        hero_data = {
            "name": "Eric",
            "title": "Heartbroken Warrior",
            "appearance": "Young hero in his 20s, brown hair, scar on eye, heartbroken and sad, broken copper armor red and black outfit",
            "signature_gear": "dual edge sword and a small magic Red dagger",
            "user_concept": "Young hero in his 20s, brown hair, scar on eye, heartbroken and sad, broken copper armor red and black outfit, fights with dual edge sword and a small magic Red dagger.",
        }
        tiers = SpritesPipeline.parse_weapons_and_combat_tiers(hero_data)
        self.assertEqual(tiers["primary_heavy_weapon"], "dual edge sword")
        self.assertEqual(tiers["secondary_light_weapon"], "small magic red dagger")
        self.assertTrue(tiers["is_dual_wield"])
        self.assertTrue(tiers["has_magic_vfx"])
        self.assertIn("heartbroken", tiers["emotional_tone"])

        arch = SpritesPipeline.detect_character_combat_archetype(hero_data, genre="Beat 'em Up (Streets of Rage / Final Fight)")
        self.assertEqual(arch["archetype"], "dual_wield")
        self.assertIn("dual edge sword", arch["weapon_label"])
        self.assertIn("small magic red dagger", arch["weapon_label"])
        self.assertIn("dual edge sword in dominant hand", arch["idle_posture"])
        self.assertIn("small magic red dagger in reverse off-hand", arch["idle_posture"])

    def test_streets_of_rage_4_moveset_generation(self):
        """Tests generation of the 10 codified Streets of Rage 4 beat 'em up moves."""
        hero_data = {
            "name": "Eric",
            "signature_gear": "dual edge sword and a small magic Red dagger",
            "user_concept": "Young hero in his 20s, brown hair, scar on eye, broken copper armor red and black outfit, fights with dual edge sword and a small magic Red dagger.",
        }
        pipeline = SpritesPipeline()
        moves = pipeline.get_smart_contextual_moves(hero_data, genre="Beat 'em Up (Streets of Rage / Final Fight)")

        # Verify exactly 10 moves
        self.assertEqual(len(moves), 10)
        move_names = [m["name"] for m in moves]
        self.assertIn("Combat Idle Stance", move_names)
        self.assertIn("Walk Forward", move_names)
        self.assertIn("Sprint Dash", move_names)
        self.assertIn("3-Hit Light Attack Combo", move_names)
        self.assertIn("3-Hit Heavy Attack Combo", move_names)
        self.assertIn("Heavy Double-Damage Charge Hit", move_names)
        self.assertIn("Sprint Dash Light Strike", move_names)
        self.assertIn("Stationary Dodge Counter", move_names)
        self.assertIn("Hit Reaction (Hurt Recoil)", move_names)
        self.assertIn("Knockdown & Ground Recovery (Drop Down & Get Up)", move_names)

        # Verify weapon distribution
        light_combo = next(m for m in moves if m["name"] == "3-Hit Light Attack Combo")
        self.assertIn("small magic red dagger", light_combo["description"])
        self.assertIn("magic sparks", light_combo["description"])

        heavy_combo = next(m for m in moves if m["name"] == "3-Hit Heavy Attack Combo")
        self.assertIn("dual edge sword", heavy_combo["description"])
        self.assertIn("knockdown", heavy_combo["description"])

        knockdown = next(m for m in moves if m["name"] == "Knockdown & Ground Recovery (Drop Down & Get Up)")
        kd_prompt = pipeline.build_pre_animation_wan_prompt(
            character_data=hero_data,
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
            move_name=knockdown["name"],
            move_type=knockdown["type"],
            move_description=knockdown["description"],
        )
        self.assertIn("4-Phase Staged Knockdown & Recovery", kd_prompt)
        self.assertIn("Phase 1", kd_prompt)
        self.assertIn("Phase 2", kd_prompt)
        self.assertIn("Phase 3", kd_prompt)
        self.assertIn("Phase 4", kd_prompt)

    def test_ai_neural_matting_cavity_removal(self):
        """Tests that remove_background_alpha completely excises enclosed negative space cavities."""
        # Create an image with an enclosed white background cavity between two dark vertical bars (like legs)
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        # Left leg (x: 20-35, y: 10-90) in black (0, 0, 0)
        for x in range(20, 36):
            for y in range(10, 91):
                img.putpixel((x, y), (15, 15, 15))
        # Right leg (x: 65-80, y: 10-90) in black (0, 0, 0)
        for x in range(65, 81):
            for y in range(10, 91):
                img.putpixel((x, y), (15, 15, 15))
        # Top waist connecting them (x: 20-80, y: 10-25) in black (0, 0, 0)
        for x in range(20, 81):
            for y in range(10, 26):
                img.putpixel((x, y), (15, 15, 15))
        # Bottom floor connector (x: 20-80, y: 85-90) creating a completely enclosed white cavity inside (x: 36-64, y: 26-84)
        for x in range(20, 81):
            for y in range(85, 91):
                img.putpixel((x, y), (15, 15, 15))

        rgba = remove_background_alpha(img, bg_color=(255, 255, 255), tolerance=38.0, defringe=True)
        self.assertEqual(rgba.mode, "RGBA")
        # Exterior corner is transparent
        self.assertEqual(rgba.getpixel((5, 5))[3], 0)
        # Leg character body is opaque
        self.assertEqual(rgba.getpixel((25, 50))[3], 255)
        # The internal enclosed cavity (50, 50) MUST be transparent (alpha == 0), NOT solid white!
        cavity_alpha = rgba.getpixel((50, 50))[3]
        self.assertEqual(cavity_alpha, 0)

    @patch("sprites_pipeline.run_remote_inference")
    def test_qwen_stride_keyframe_and_flf2v_loop_anchoring(self, mock_wan):
        """Tests that synthesize_animation_clip conditions both start and end frames with stride keyframes for locomotion loops."""
        mock_wan.return_value = ("output/walk_loop.mp4", 1234)
        pipeline = SpritesPipeline()
        dummy_img = Image.new("RGB", (64, 64), (200, 50, 50))
        char_data = {
            "name": "Hero",
            "master_sprite_pil": dummy_img,
            "master_sprite_path": "output/hero_master.png",
            "genre": "Beat 'em Up (Streets of Rage / Final Fight)",
        }
        walk_move = {
            "name": "Walk Forward",
            "type": "loop",
            "description": "Treadmill walking stride",
            "refined_prompt": "Hero walks on pure white background #FFFFFF",
        }

        with patch.object(pipeline, "synthesize_stride_keyframe_with_qwen", return_value=dummy_img) as mock_qwen_stride:
            clip_path = pipeline.synthesize_animation_clip(char_data, walk_move, wan_seed=42)
            self.assertEqual(clip_path, "output/walk_loop.mp4")
            mock_qwen_stride.assert_called_once()
            # Verify Wan inference was called with matching start and last frame images
            call_kwargs = mock_wan.call_args[1]
            self.assertIsNotNone(call_kwargs.get("resized_image"))
            self.assertIsNotNone(call_kwargs.get("processed_last_image"))
            self.assertIsNotNone(call_kwargs.get("last_frame_image"))
            self.assertEqual(call_kwargs.get("resized_image"), call_kwargs.get("last_frame_image"))

    def test_combo_advancing_footwork_reasoning_sor4_and_shinobi(self):
        """Tests AI decision-making and advancing step-and-hit footwork for Light (Shinobi) and Heavy (Streets of Rage 4) combos."""
        hero_data = {
            "name": "Eric",
            "signature_gear": "dual edge sword and a small magic Red dagger",
            "user_concept": "Young hero in his 20s, brown hair, scar on eye, broken copper armor red and black outfit, fights with dual edge sword and a small magic Red dagger.",
        }
        pipeline = SpritesPipeline()
        moves = pipeline.get_smart_contextual_moves(hero_data, genre="Beat 'em Up (Streets of Rage / Final Fight)")

        light_combo = next(m for m in moves if m["name"] == "3-Hit Light Attack Combo")
        heavy_combo = next(m for m in moves if m["name"] == "3-Hit Heavy Attack Combo")

        # 1. Verify Light Combo uses Shinobi: Art of Vengeance darting slip-step & cross-pivot reasoning
        self.assertIn("Shinobi: Art of Vengeance", light_combo["description"])
        self.assertIn("slip-step", light_combo["description"])
        self.assertIn("cross-stepping", light_combo["description"])
        self.assertIn("Shinobi: Art of Vengeance", light_combo["footwork_reasoning"])
        self.assertIn("NEVER stand still with glued feet", light_combo["refined_prompt"])
        self.assertIn("Step 1", light_combo["refined_prompt"])
        self.assertIn("Step 2", light_combo["refined_prompt"])
        self.assertIn("Step 3", light_combo["refined_prompt"])
        self.assertIn("Step 4", light_combo["refined_prompt"])

        # 2. Verify Heavy Combo uses Streets of Rage 4 power step-in, torque cross-step, and 360 pivot reasoning
        self.assertIn("Streets of Rage 4", heavy_combo["description"])
        self.assertIn("power step-in", heavy_combo["description"])
        self.assertIn("cross-step", heavy_combo["description"])
        self.assertIn("360-degree", heavy_combo["description"])
        self.assertIn("Streets of Rage 4", heavy_combo["footwork_reasoning"])
        self.assertIn("NEVER stand still with glued feet", heavy_combo["refined_prompt"])
        self.assertIn("Advancing Power Step", heavy_combo["refined_prompt"])

    def test_delete_move_in_phase2(self):
        """Tests deleting actions/moves/behaviors in Phase 2 with full UI synchronization and last-move protection."""
        state = {
            "character": {
                "name": "Eric",
                "moves": [
                    {"id": "m1", "name": "Combat Idle Stance", "type": "loop", "description": "Resting guard"},
                    {"id": "m2", "name": "Walk Forward", "type": "loop", "description": "Walking stride"},
                    {"id": "m3", "name": "3-Hit Light Attack Combo", "type": "action", "description": "Dagger flurry"},
                ],
            }
        }
        # 1. Delete the 3rd move (3-Hit Light Attack Combo)
        s1, s2, s3, moves, edit_name, edit_type, edit_desc, edit_prompt, status, state = app.sprites_ui_delete_selected_move(
            choice_label="3-Hit Light Attack Combo (ACTION)",
            char_state=state,
        )
        self.assertEqual(len(moves), 2)
        self.assertEqual(len(state["character"]["moves"]), 2)
        self.assertIn("Deleted Move: `3-Hit Light Attack Combo`", status)
        self.assertEqual(len(s1["choices"]), 2)
        self.assertEqual(len(s2["choices"]), 2)
        self.assertEqual(len(s3["choices"]), 2)
        self.assertNotIn("3-Hit Light Attack Combo", [m["name"] for m in moves])

        # 2. Delete down to 1 move
        s1, s2, s3, moves, edit_name, edit_type, edit_desc, edit_prompt, status, state = app.sprites_ui_delete_selected_move(
            choice_label="Walk Forward (LOOP)",
            char_state=state,
        )
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["name"], "Combat Idle Stance")

        # 3. Attempt to delete the last remaining move (must be rejected)
        s1, s2, s3, moves, edit_name, edit_type, edit_desc, edit_prompt, status, state = app.sprites_ui_delete_selected_move(
            choice_label="Combat Idle Stance (LOOP)",
            char_state=state,
        )
        self.assertEqual(len(moves), 1)
        self.assertIn("Cannot delete the only remaining move", status)

    def test_ai_smart_mandatory_idle_and_walk_locomotion_loops(self):
        """Tests that the AI pipeline always guarantees both the Idle loop and the Walking animation loop alongside action behaviors."""
        char_data = {
            "name": "Valen",
            "signature_gear": "twin daggers",
            "appearance": "Agile rogue in black hood",
        }
        # Given a moveset with ONLY attack actions (no Idle, no Walk)
        action_only_moves = [
            {"id": "m1", "name": "Dagger Stab", "type": "action", "description": "Quick forward stab"},
            {"id": "m2", "name": "Shadow Finisher", "type": "action", "description": "Spinning somersault slash"},
        ]
        guaranteed = SpritesPipeline.ensure_mandatory_locomotion_loops(
            moves=action_only_moves,
            character_data=char_data,
            genre="Beat 'em Up (Streets of Rage / Final Fight)",
        )
        # Must now contain at least 4 moves (Idle + Walk + 2 actions)
        self.assertGreaterEqual(len(guaranteed), 4)
        names = [m["name"] for m in guaranteed]
        types = [m["type"] for m in guaranteed]

        # Verify Idle Loop is present
        self.assertTrue(any("idle" in n.lower() or "stance" in n.lower() for n in names))
        # Verify Walking Loop is present
        self.assertTrue(any("walk" in n.lower() for n in names))
        # Verify Action behaviors remain present
        self.assertIn("Dagger Stab", names)
        self.assertIn("Shadow Finisher", names)

        # Verify loop types are 'loop'
        idle_move = next(m for m in guaranteed if "idle" in m["name"].lower() or "stance" in m["name"].lower())
        walk_move = next(m for m in guaranteed if "walk" in m["name"].lower())
        self.assertEqual(idle_move["type"], "loop")
        self.assertEqual(walk_move["type"], "loop")
        self.assertIn("twin daggers", walk_move["refined_prompt"])

    @patch("sprites_pipeline.run_remote_inference")
    def test_animation_clips_saved_to_clips_folder(self, mock_wan):
        """Verifies synthesized video animations are explicitly routed and saved into outputs/sprites/clips/."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SpritesPipeline(base_dir=tmp_dir)
            dummy_img = Image.new("RGB", (64, 64), (255, 255, 255))
            char_data = {
                "name": "Axel Stone",
                "master_sprite_pil": dummy_img,
                "genre": "Beat 'em Up (Streets of Rage / Final Fight)",
            }
            move_data = {
                "name": "Grand Upper Charge Hit",
                "type": "action",
                "refined_prompt": "Axel slides forward with flame dragon uppercut",
            }

            # Simulate run_remote_inference writing the video to save_dir/output_filename
            expected_filename = "axel_stone_grand_upper_charge_hit_clip.mp4"
            expected_clip_path = pipeline.clips_dir / expected_filename

            def fake_remote_inference(**kwargs):
                save_dir = Path(kwargs.get("save_dir", pipeline.clips_dir))
                save_dir.mkdir(parents=True, exist_ok=True)
                out_name = kwargs.get("output_filename", expected_filename)
                target = save_dir / out_name
                target.write_bytes(b"FAKE_MP4_BYTES")
                return str(target), 42

            mock_wan.side_effect = fake_remote_inference

            result_path = pipeline.synthesize_animation_clip(char_data, move_data, wan_seed=42)

            # Verify kwargs passed to run_remote_inference
            call_kwargs = mock_wan.call_args[1]
            self.assertEqual(call_kwargs["save_dir"], str(pipeline.clips_dir))
            self.assertEqual(call_kwargs["output_filename"], expected_filename)

            # Verify output location
            self.assertTrue(Path(result_path).exists())
            self.assertEqual(Path(result_path).resolve(), expected_clip_path.resolve())
            self.assertEqual(move_data["video_path"], str(expected_clip_path))
            self.assertTrue(str(result_path).endswith(expected_filename))
            self.assertIn("clips", Path(result_path).parts)


if __name__ == "__main__":
    unittest.main()


