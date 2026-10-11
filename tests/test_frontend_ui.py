import asyncio
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import gradio as gr

import app
import backend
from studio_pipeline import StudioPipeline


class TestGradioFrontendStructure(unittest.TestCase):
    """Verifies the Gradio Blocks layout, 4-Phase Wizard tabs, 3-seed bar, and auto-heartbeat controls."""

    def test_demo_blocks_and_4_phase_wizard_tabs_exist(self):
        """app.demo should be a gr.Blocks instance containing the 4 Phase TabItems."""
        self.assertIsInstance(app.demo, gr.Blocks)
        self.assertIn("Filmmaking Studio", app.demo.title)

        tab_items = [
            b for b in app.demo.blocks.values()
            if isinstance(b, gr.TabItem)
        ]
        tab_ids = { getattr(t, "id", None) for t in tab_items }
        for expected_id in ("phase1", "phase2", "phase3", "phase4"):
            self.assertIn(expected_id, tab_ids, f"Missing wizard tab id '{expected_id}'")

    def test_three_seed_inputs_initialized_and_synced(self):
        """The 3 global seed inputs and 3 inspector seed inputs should be initialized from INITIAL_STUDIO_SEEDS."""
        z_seed = app.INITIAL_STUDIO_SEEDS["z_image_seed"]
        q_seed = app.INITIAL_STUDIO_SEEDS["qwen_seed"]
        w_seed = app.INITIAL_STUDIO_SEEDS["wan_seed"]

        self.assertEqual(app.studio_seed.value, z_seed)
        self.assertEqual(app.studio_qwen_seed.value, q_seed)
        self.assertEqual(app.studio_wan_seed.value, w_seed)

        self.assertEqual(app.studio_char_seed_edit.value, z_seed)
        self.assertEqual(app.studio_scene_seed_edit.value, q_seed)
        self.assertEqual(app.studio_clip_seed_edit.value, w_seed)

    def test_auto_heartbeat_buttons_and_timer_configured(self):
        """All 3 model heartbeat buttons should start in Active/primary state and the 25s timer should be active."""
        self.assertIn("Active", app.z_image_hb_btn.value)
        self.assertIn("Active", app.qwen_hb_btn.value)
        self.assertIn("Active", app.wan_hb_btn.value)

        self.assertEqual(app.z_image_hb_btn.variant, "primary")
        self.assertEqual(app.qwen_hb_btn.variant, "primary")
        self.assertEqual(app.wan_hb_btn.variant, "primary")

        self.assertIsInstance(app.heartbeat_timer, gr.Timer)
        self.assertEqual(app.heartbeat_timer.value, 25)
        self.assertTrue(app.heartbeat_timer.active)

        # Backend flags should all be True on startup
        self.assertTrue(backend.z_image_heartbeat_enabled)
        self.assertTrue(backend.qwen_heartbeat_enabled)
        self.assertTrue(backend.wan_heartbeat_enabled)

    def test_registered_event_functions_in_demo_graph(self):
        """Verify key frontend handlers are bound into app.demo.fns."""
        registered_fn_names = {
            getattr(fn_obj.fn, "__name__", "")
            for fn_obj in app.demo.fns.values()
            if getattr(fn_obj, "fn", None) is not None
        }
        expected_handlers = {
            "_roll_and_sync_seeds",
            "_ui_step1",
            "_validate_phase1",
            "_ui_step2",
            "_on_char_gallery_select",
            "studio_load_character_fields",
            "studio_save_character_text_edits",
            "studio_redo_single_character",
            "_ui_step3",
            "_on_scene_gallery_select",
            "studio_load_scene_fields",
            "studio_save_scene_text_edits",
            "studio_redo_single_scene",
            "_ui_director",
            "_ui_step4",
            "studio_load_clip_fields",
            "studio_redirect_single_clip",
            "studio_redo_single_clip",
            "studio_run_full",
            "toggle_z_image_heartbeat",
            "toggle_wan_heartbeat",
            "toggle_qwen_heartbeat",
            "handle_client_heartbeat",
        }
        for name in expected_handlers:
            self.assertIn(name, registered_fn_names, f"Handler '{name}' is not bound in app.demo.fns")


class TestFrontendThreeSeedRolling(unittest.TestCase):
    """Tests rolling and syncing the 3 studio seeds across global and inspector inputs."""

    def test_roll_three_studio_seeds(self):
        z_s, q_s, w_s = app.roll_three_studio_seeds()
        for s in (z_s, q_s, w_s):
            self.assertIsInstance(s, int)
            self.assertGreaterEqual(s, 100000)
            self.assertLessEqual(s, 999999999)

    def test_roll_and_sync_seeds_updates_all_six_boxes(self):
        z_s, q_s, w_s, char_s, scene_s, clip_s = app._roll_and_sync_seeds()
        self.assertEqual(z_s, char_s)
        self.assertEqual(q_s, scene_s)
        self.assertEqual(w_s, clip_s)


class TestFrontendSelectorsAndGallery(unittest.TestCase):
    """Tests dropdown choice generation, selector index parsing, gallery formatting, and director summary table."""

    def test_character_and_scene_choices(self):
        self.assertEqual(app._character_choices({}), [])
        self.assertEqual(app._scene_choices(None), [])

        story = {
            "characters": [
                {"id": "char_1", "name": "Luna"},
                {"id": "char_2", "name": "Oliver"},
            ],
            "scenes": [
                {"scene_number": 1, "title": "The Discovery"},
                {"scene_number": 2, "title": "The Great Lift"},
            ],
        }
        c_choices = app._character_choices(story)
        s_choices = app._scene_choices(story)

        self.assertEqual(c_choices, ["1. Luna (char_1)", "2. Oliver (char_2)"])
        self.assertEqual(s_choices, ["Scene 1: The Discovery", "Scene 2: The Great Lift"])

    def test_parse_selector_index_modes(self):
        choices = ["1. Luna (char_1)", "2. Oliver (char_2)", "3. Pippin (char_3)"]

        # Empty choices
        self.assertEqual(app._parse_selector_index("anything", []), 0)
        # Exact match
        self.assertEqual(app._parse_selector_index("2. Oliver (char_2)", choices), 1)
        # Substring / prefix match
        self.assertEqual(app._parse_selector_index("Oliver", choices), 1)
        self.assertEqual(app._parse_selector_index("3.", choices), 2)
        # Integer index (with clamping)
        self.assertEqual(app._parse_selector_index(2, choices), 2)
        self.assertEqual(app._parse_selector_index(99, choices), 2)
        self.assertEqual(app._parse_selector_index(-5, choices), 0)
        # Unknown string fallback
        self.assertEqual(app._parse_selector_index("Unknown", choices), 0)

    def test_studio_gallery_items_and_director_summary(self):
        story = {
            "characters": [
                {"name": "Luna", "portrait_path": "/tmp/luna.png", "portrait_seed": 111},
                {"name": "NoPic"},  # should be skipped since portrait_path is missing
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "Meadow",
                    "keyframe_path": "/tmp/s1.jpg",
                    "keyframe_seed": 222,
                    "director": {
                        "camera_movement": "slow push-in",
                        "motion_intensity": 2,
                        "num_frames": 81,
                        "source": "gemma-vision",
                    },
                }
            ],
        }
        char_items, scene_items = app._studio_gallery_items(story)
        self.assertEqual(len(char_items), 1)
        self.assertEqual(char_items[0], ("/tmp/luna.png", "Luna (Seed: 111)"))
        self.assertEqual(len(scene_items), 1)
        self.assertEqual(scene_items[0], ("/tmp/s1.jpg", "Scene 1: Meadow (Seed: 222)"))

        table_md = app._studio_director_summary(story["scenes"])
        self.assertIn("| 1 | Meadow | slow push-in | 2 | 81 | gemma-vision |", table_md)
        self.assertEqual(app._studio_director_summary([]), "")


class TestFrontendPhaseHandlersAndEdgeCases(unittest.TestCase):
    """Tests Phase 1-4 inspector load/save handlers, gallery selection events, validation, and error guards."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.char_img_path = os.path.join(self.temp_dir.name, "char1.png")
        self.scene_img_path = os.path.join(self.temp_dir.name, "scene1.jpg")
        self.clip_vid_path = os.path.join(self.temp_dir.name, "clip1.mp4")

        Image.new("RGB", (64, 64), color="red").save(self.char_img_path)
        Image.new("RGB", (64, 64), color="blue").save(self.scene_img_path)
        with open(self.clip_vid_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42")

        self.sample_state = {
            "title": "Starry Rescue",
            "art_style": "2D vibrant cartoon animation",
            "characters": [
                {
                    "id": "char_1",
                    "name": "Luna",
                    "face": "amber eyes",
                    "hair": "russet fur",
                    "clothing": "indigo scarf",
                    "signature_details": "brass lantern",
                    "z_image_prompt": "Luna portrait prompt",
                    "portrait_seed": 1111,
                    "portrait_path": self.char_img_path,
                },
                {
                    "id": "char_2",
                    "name": "Oliver",
                    "face": "round spectacles",
                    "hair": "white feathers",
                    "clothing": "emerald vest",
                    "signature_details": "pocket watch",
                    "z_image_prompt": "Oliver portrait prompt",
                    "portrait_seed": 1111,
                    "portrait_path": self.char_img_path,
                },
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "The Discovery",
                    "scenario": "Luna finds a fallen star",
                    "qwen_scene_prompt": "Luna in forest with glowing star",
                    "character_ids": ["char_1", "char_2"],
                    "keyframe_seed": 2222,
                    "keyframe_path": self.scene_img_path,
                    "wan_motion_prompt": "Luna kneels beside the star",
                    "camera_movement": "Slow Push-In",
                    "video_seed": 3333,
                    "video_path": self.clip_vid_path,
                    "director": {
                        "camera_movement": "gentle dolly in",
                        "motion_intensity": 2,
                        "wan_prompt": "Compiled director prompt for scene 1",
                        "source": "gemma-vision",
                    },
                }
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_phase1_validation_updates_title_and_switches_tab(self):
        state = {"title": "Old Title", "characters": [], "scenes": []}
        tab_update, status, updated_state = app._validate_phase1("New Edited Title", state)
        self.assertEqual(updated_state["title"], "New Edited Title")
        self.assertEqual(tab_update["selected"], "phase2")
        self.assertIn("Phase 1 Validated", status)

    def test_phase2_load_save_and_gallery_select_character(self):
        # Empty state guard
        empty_fields = app.studio_load_character_fields(None, {}, default_seed=777)
        self.assertEqual(empty_fields, ("", "", "", "", "", "", 777, None))

        # Load character 2 ("Oliver")
        fields = app.studio_load_character_fields("2. Oliver (char_2)", self.sample_state, default_seed=999)
        self.assertEqual(fields[0], "Oliver")
        self.assertEqual(fields[1], "round spectacles")
        self.assertEqual(fields[3], "emerald vest")
        self.assertEqual(fields[6], 1111)
        self.assertEqual(fields[7], self.char_img_path)

        # Save text edits for character 2
        json_out, status, new_state = app.studio_save_character_text_edits(
            "2. Oliver (char_2)",
            "Professor Oliver",
            "golden eyes with spectacles",
            "snowy feathers",
            "velvet academic cloak",
            "silver compass",
            "Updated Oliver Z-Image prompt",
            self.sample_state,
        )
        self.assertIn("Saved description updates for Character #2", status)
        self.assertEqual(new_state["characters"][1]["name"], "Professor Oliver")
        self.assertEqual(new_state["characters"][1]["clothing"], "velvet academic cloak")
        self.assertEqual(json_out["characters"][1]["signature_details"], "silver compass")

        # Simulate clicking Character #2 in the Gradio gallery
        mock_evt = MagicMock()
        mock_evt.index = 1
        sel_res = app._on_char_gallery_select(mock_evt, new_state, default_seed=999)
        self.assertEqual(sel_res[0]["value"], "2. Professor Oliver (char_2)")
        self.assertEqual(sel_res[1], "Professor Oliver")

    def test_phase3_load_save_and_gallery_select_scene(self):
        # Empty state guard
        empty_scene = app.studio_load_scene_fields(None, {}, default_seed=888)
        self.assertEqual(empty_scene, ("", "", "", "", 888, None))

        # Load scene 1
        s_fields = app.studio_load_scene_fields("Scene 1: The Discovery", self.sample_state, default_seed=888)
        self.assertEqual(s_fields[0], "The Discovery")
        self.assertEqual(s_fields[1], "Luna finds a fallen star")
        self.assertEqual(s_fields[3], "char_1, char_2")
        self.assertEqual(s_fields[4], 2222)
        self.assertEqual(s_fields[5], self.scene_img_path)

        # Save text edits for scene 1
        json_out, status, new_state = app.studio_save_scene_text_edits(
            "Scene 1: The Discovery",
            "The Glowing Crater",
            "Luna discovers a warm baby star",
            "Wide shot of Luna and glowing star",
            "char_1",
            self.sample_state,
        )
        self.assertIn("Saved description updates for Scene #1", status)
        self.assertEqual(new_state["scenes"][0]["title"], "The Glowing Crater")
        self.assertEqual(new_state["scenes"][0]["character_ids"], ["char_1"])

        # Simulate clicking Scene #1 in the Gradio gallery
        mock_evt = MagicMock()
        mock_evt.index = 0
        sel_res = app._on_scene_gallery_select(mock_evt, new_state, default_seed=888)
        self.assertEqual(sel_res[0]["value"], "Scene 1: The Glowing Crater")
        self.assertEqual(sel_res[1], "The Glowing Crater")

    @patch.object(StudioPipeline, "_chat_completion")
    def test_phase4_load_and_redirect_single_clip(self, mock_chat):
        # Empty state guard
        empty_clip = app.studio_load_clip_fields(None, {}, default_seed=555)
        self.assertEqual(empty_clip, ("", "", "", 555, None))

        # Load clip 1
        c_fields = app.studio_load_clip_fields("Scene 1: The Discovery", self.sample_state, default_seed=555)
        self.assertEqual(c_fields[0], "Luna kneels beside the star")
        self.assertEqual(c_fields[1], "gentle dolly in")
        self.assertEqual(c_fields[2], "Compiled director prompt for scene 1")
        self.assertEqual(c_fields[3], 3333)
        self.assertEqual(c_fields[4], self.clip_vid_path)

        # Re-direct single clip via Gemma Vision
        mock_chat.return_value = json.dumps({
            "shot_type": "close-up",
            "camera_movement": "orbital tracking shot",
            "motion_intensity": 3,
            "subject_action": "Luna cradles the glowing star",
            "timeline": ["kneels", "lifts star", "smiles"],
            "wan_prompt": "Luna cradles the glowing star as camera orbits",
            "guidance_scale": 2.0,
            "steps": 8,
            "flow_shift": 3.0,
        })
        cam, wan_p, json_out, status, new_state = app.studio_redirect_single_clip(
            "Scene 1: The Discovery",
            "Luna cradles the glowing star",
            "orbital tracking shot",
            True,
            "dummy_key",
            "gemma4:31b",
            self.sample_state,
        )
        self.assertEqual(cam, "orbital tracking shot")
        self.assertIn("Luna cradles the glowing star", wan_p)
        self.assertIn("Gemma Vision Re-Directed Scene #1", status)

    def test_empty_state_and_prerequisite_guards(self):
        """All phase execution & redo handlers should return friendly warnings if prerequisites are missing."""
        # Step 2 without Step 1
        chars, msg2, _ = app.studio_run_step2(42, "key", "model", {})
        self.assertEqual(chars, [])
        self.assertIn("Please run Step 1", msg2)

        # Step 3 without Step 2 portraits
        scenes, msg3, _ = app.studio_run_step3(
            "9:16 Vertical (Shorts/Reels)", 42, "key", "model",
            {"characters": [{"id": "char_1"}], "scenes": [{"scene_number": 1}]},
        )
        self.assertEqual(scenes, [])
        self.assertIn("Please run Step 1 & Step 2", msg3)

        # Director without Step 3 keyframes
        _, msg_dir, _ = app.studio_run_director("key", "model", True, {"scenes": [{"scene_number": 1}]})
        self.assertIn("Please run Step 3", msg_dir)

        # Step 4 without Step 3 keyframes
        vid, msg4, _ = app.studio_run_step4(
            "9:16 Vertical (Shorts/Reels)", 42, "key", "model", True, {"scenes": [{"scene_number": 1}]}
        )
        self.assertIsNone(vid)
        self.assertIn("Please run Step 3", msg4)

        # Individual Redo handlers on empty state
        r_char = app.studio_redo_single_character("1", "N", "F", "H", "C", "S", "P", 42, "k", "m", {})
        self.assertIn("Please run Phase 1", r_char[3])

        r_scene = app.studio_redo_single_scene("1", "T", "S", "P", "char_1", 42, "9:16 Vertical (Shorts/Reels)", "k", "m", {})
        self.assertIn("Please run Phase 1 & Phase 2", r_scene[3])

        r_clip = app.studio_redo_single_clip("1", "M", "C", "P", 42, False, True, "9:16 Vertical (Shorts/Reels)", "k", "m", {})
        self.assertIn("Please generate scene keyframes first", r_clip[3])


class TestFrontendHeartbeatAsyncButtons(unittest.TestCase):
    """Tests the async heartbeat toggle callbacks and client heartbeat timer handler."""

    def tearDown(self):
        # Restore all 3 heartbeats to True (default active state)
        backend.set_z_image_heartbeat_state(True)
        backend.set_qwen_heartbeat_state(True)
        backend.set_wan_heartbeat_state(True)

    def test_async_heartbeat_toggle_buttons(self):
        async def _run():
            backend.set_z_image_heartbeat_state(True)
            res_paused = await app.toggle_z_image_heartbeat()
            self.assertFalse(backend.z_image_heartbeat_enabled)
            self.assertIn("Paused", res_paused["value"])
            self.assertEqual(res_paused["variant"], "secondary")

            res_active = await app.toggle_z_image_heartbeat()
            self.assertTrue(backend.z_image_heartbeat_enabled)
            self.assertIn("Active", res_active["value"])
            self.assertEqual(res_active["variant"], "primary")

            # Also verify Wan and Qwen toggles
            res_wan = await app.toggle_wan_heartbeat()
            self.assertFalse(backend.wan_heartbeat_enabled)
            self.assertIn("Paused", res_wan["value"])

            res_qwen = await app.toggle_qwen_heartbeat()
            self.assertFalse(backend.qwen_heartbeat_enabled)
            self.assertIn("Paused", res_qwen["value"])

            # Client heartbeat timer tick
            before = backend.last_frontend_heartbeat_time
            await app.handle_client_heartbeat()
            self.assertGreaterEqual(backend.last_frontend_heartbeat_time, before)
            self.assertGreater(backend.last_frontend_heartbeat_time, 0)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
