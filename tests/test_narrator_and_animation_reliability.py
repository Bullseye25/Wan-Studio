import os
import json
import wave
import tempfile
import unittest
import pathlib
from unittest.mock import patch
import numpy as np
import cv2
from PIL import Image, ImageDraw

from narrator_engine import (
    NARRATOR_VOICE_PRESETS,
    DEFAULT_NARRATOR_VOICE,
    enforce_narration_word_budget,
    extract_sync_trigger_word,
    OpenSourceNarratorTTS,
    AudioVisualSyncEngine,
)
from cinematography_vault import CAUSAL_NARRATIVE_BLUEPRINTS, StoryMemoryBank
from studio_pipeline import (
    StudioPipeline,
    evaluate_flf2v_compatibility,
    stabilize_end_frame_background,
    compute_motion_adaptive_wan_params,
    verify_video_clip_quality,
)


class TestNarratorAndAnimationReliability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = pathlib.Path(self.temp_dir.name)
        self.pipeline = StudioPipeline(
            api_key="test_key",
            output_base_dir=str(self.base_path),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Narrator Script Word-Budget & Trigger Word Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def test_narration_word_budget_and_trigger_extraction(self):
        long_line = (
            "Barnaby slowly walks across the enchanted forest meadow to give the basket of sweet wild berries to Pippin today."
        )
        trimmed = enforce_narration_word_budget(long_line, max_words=11)
        words = trimmed.split()
        self.assertLessEqual(len(words), 11)
        self.assertTrue(trimmed.endswith((".", "!", "?")))

        narration = "Barnaby kneels down and offers the glowing berry to Pippin."
        trigger = extract_sync_trigger_word(narration, explicit_trigger="offers")
        self.assertEqual(trigger, "offers")

        auto_trigger = extract_sync_trigger_word("Kael leaps forward and slashes through the heavy drone.")
        self.assertIn(auto_trigger, ["leaps", "slashes", "forward"])

    def test_causal_blueprints_include_narrator_fields(self):
        for bp in CAUSAL_NARRATIVE_BLUEPRINTS:
            for s in bp.get("scenes", []):
                self.assertTrue(s.get("narration_text"), f"Missing narration_text in {bp.get('title')}")
                self.assertTrue(s.get("sync_trigger_word"), f"Missing sync_trigger_word in {bp.get('title')}")
                self.assertTrue(s.get("narration_voice_style"), f"Missing narration_voice_style in {bp.get('title')}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Open-Source Narrator TTS Synthesis & Word Timestamps
    # ─────────────────────────────────────────────────────────────────────────

    def test_open_source_narrator_tts_wav_and_word_timestamps(self):
        tts = OpenSourceNarratorTTS(output_dir=str(self.base_path / "audio"))
        wav_path = str(self.base_path / "audio" / "scene_01.wav")
        res = tts.synthesize_scene_narration(
            text="Barnaby kneels down and offers the sweet berry to Pippin.",
            output_wav_path=wav_path,
            voice_preset="warm_storyteller",
            sync_trigger_word="offers",
            target_duration_sec=5.0,
        )
        self.assertTrue(os.path.exists(res["audio_path"]))
        self.assertGreater(os.path.getsize(res["audio_path"]), 1000)
        self.assertEqual(res["sample_rate"], 24000)
        self.assertGreater(len(res["word_timestamps"]), 5)
        self.assertEqual(res["sync_trigger_word"], "offers")
        self.assertGreater(res["trigger_timestamp_sec"], 0.2)
        self.assertLess(res["trigger_timestamp_sec"], 4.5)

        with wave.open(res["audio_path"], "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 24000)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Audio-Visual Synchronization Algorithm (4k+1 Quantization & Phase Lock)
    # ─────────────────────────────────────────────────────────────────────────

    def test_audio_visual_sync_engine_quantize_and_phase_lock(self):
        q = AudioVisualSyncEngine.quantize_audio_to_wan_frames(3.2, fps=16)
        self.assertEqual((q["num_frames"] - 1) % 4, 0)
        self.assertGreaterEqual(q["num_frames"], 17)
        self.assertLessEqual(q["num_frames"], 81)

        word_ts = [
            {"word": "Barnaby", "start": 0.15, "end": 0.55},
            {"word": "kneels", "start": 0.60, "end": 1.05},
            {"word": "and", "start": 1.10, "end": 1.35},
            {"word": "offers", "start": 1.42, "end": 1.95},
            {"word": "the", "start": 2.00, "end": 2.25},
            {"word": "berry.", "start": 2.30, "end": 2.85},
        ]
        synced = AudioVisualSyncEngine.compute_synced_phase_boundaries(
            word_timestamps=word_ts,
            sync_trigger_word="offers",
            clip_duration_sec=5.0,
        )
        self.assertAlmostEqual(synced["trigger_onset_sec"], 1.42, places=2)
        # p1 should be slightly before trigger onset (clamped >= 0.5s)
        self.assertAlmostEqual(synced["p1"], 1.27, places=2)
        self.assertGreater(synced["p2"], synced["p1"] + 1.0)
        self.assertLessEqual(synced["p2"], 4.4)

    def test_srt_export_and_ffmpeg_audio_muxing(self):
        tts = OpenSourceNarratorTTS(output_dir=str(self.base_path / "audio"))
        wav_path = str(self.base_path / "audio" / "mux_test.wav")
        tts_res = tts.synthesize_scene_narration(
            text="Barnaby shares the basket with Pippin.",
            output_wav_path=wav_path,
            voice_preset="warm_storyteller",
            sync_trigger_word="shares",
            target_duration_sec=2.0,
        )

        # Create a 2-second test MP4 video (32 frames @ 16fps)
        vid_path = str(self.base_path / "video_silent.mp4")
        out_mux_path = str(self.base_path / "video_muxed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(vid_path, fourcc, 16.0, (160, 120))
        for i in range(32):
            frame = np.full((120, 160, 3), (i * 5 % 255, 100, 180), dtype=np.uint8)
            vw.write(frame)
        vw.release()

        muxed = AudioVisualSyncEngine.mux_video_and_audio(
            video_path=vid_path,
            audio_wav_path=tts_res["audio_path"],
            output_video_path=out_mux_path,
            target_duration_sec=2.0,
        )
        self.assertTrue(os.path.exists(muxed))
        self.assertGreater(os.path.getsize(muxed), 1000)

        # Test SRT subtitle generation
        srt_path = str(self.base_path / "test_subs.srt")
        scenes = [
            {
                "scene_number": 1,
                "duration_seconds": 5.0,
                "narration_text": "Barnaby shares the basket with Pippin.",
                "narration_word_timestamps": tts_res["word_timestamps"],
            }
        ]
        AudioVisualSyncEngine.write_srt_subtitles(scenes, srt_path)
        self.assertTrue(os.path.exists(srt_path))
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        self.assertIn("-->", srt_content)
        self.assertIn("Barnaby shares the basket with Pippin.", srt_content)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FLF2V Compatibility Gate, Background Stabilization & Motion-Adaptive CFG
    # ─────────────────────────────────────────────────────────────────────────

    def test_flf2v_compatibility_and_background_stabilization(self):
        # 1. Identical frames -> TOO_STATIC
        img_base = Image.new("RGB", (320, 320), color=(40, 80, 120))
        draw_base = ImageDraw.Draw(img_base)
        draw_base.rectangle([100, 80, 200, 260], fill=(220, 140, 60))

        comp_static = evaluate_flf2v_compatibility(img_base, img_base)
        self.assertEqual(comp_static["status"], "TOO_STATIC")
        self.assertTrue(comp_static["use_flf2v"])

        # 2. Moderate subject movement with stable background -> OPTIMAL_FLF2V
        img_opt = img_base.copy()
        draw_opt = ImageDraw.Draw(img_opt)
        draw_opt.rectangle([100, 80, 200, 260], fill=(40, 80, 120))
        draw_opt.rectangle([135, 75, 245, 255], fill=(235, 160, 80))
        comp_opt = evaluate_flf2v_compatibility(img_base, img_opt)
        self.assertEqual(comp_opt["status"], "OPTIMAL_FLF2V")
        self.assertTrue(comp_opt["use_flf2v"])

        # 3. Completely different image -> EXTREME_DIVERGENCE (triggers Guided I2V fallback)
        rng = np.random.RandomState(42)
        extreme_arr = rng.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        img_extreme = Image.fromarray(extreme_arr)
        comp_ext = evaluate_flf2v_compatibility(img_base, img_extreme)
        self.assertEqual(comp_ext["status"], "EXTREME_DIVERGENCE")
        self.assertFalse(comp_ext["use_flf2v"])

        # 4. Background border stabilization increases border SSIM while keeping central action
        stabilized = stabilize_end_frame_background(img_base, img_extreme, border_ratio=0.15)
        comp_after = evaluate_flf2v_compatibility(img_base, stabilized)
        self.assertGreater(comp_after["bg_ssim"], comp_ext["bg_ssim"])

    def test_motion_adaptive_wan_params_and_high_velocity_guard(self):
        p_low = compute_motion_adaptive_wan_params(motion_intensity=2, duration_seconds=3.5)
        self.assertEqual(p_low["guidance_scale"], 2.0)

        p_high = compute_motion_adaptive_wan_params(motion_intensity=5, duration_seconds=5.0)
        self.assertLessEqual(p_high["guidance_scale"], 1.75)
        self.assertLessEqual(p_high["guidance_scale_2"], 1.25)
        self.assertGreaterEqual(p_high["flow_shift"], 4.8)
        self.assertGreaterEqual(p_high["steps"], 12)

        guarded = StudioPipeline.apply_anti_noise_and_rotation_guard(
            {"motion_intensity": 5, "guidance_scale": 2.2, "guidance_scale_2": 1.8, "steps": 8, "flow_shift": 3.0},
            duration_seconds=5.0,
        )
        self.assertEqual(guarded["guidance_scale"], 1.75)
        self.assertEqual(guarded["guidance_scale_2"], 1.25)
        self.assertEqual(guarded["steps"], 12)
        self.assertEqual(guarded["flow_shift"], 4.8)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Post-Render Optical Flow & Audio Sync Telemetry Gate
    # ─────────────────────────────────────────────────────────────────────────

    def test_verify_video_clip_quality_and_telemetry_persistence(self):
        # Create a moving video clip where a bright square moves across the frame peaking around t=1.0s
        moving_vid = str(self.base_path / "moving_clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(moving_vid, fourcc, 16.0, (160, 160))
        x_pos = 20
        for i in range(32):
            frame = np.full((160, 160, 3), 30, dtype=np.uint8)
            step = 8 if 12 <= i <= 18 else 2
            x_pos = min(120, x_pos + step)
            cv2.rectangle(frame, (x_pos, 50), (x_pos + 30, 110), (240, 200, 80), -1)
            vw.write(frame)
        vw.release()

        telemetry = verify_video_clip_quality(moving_vid, expected_peak_sec=1.0)
        self.assertGreater(telemetry["mean_flow"], 0.05)
        self.assertIsNotNone(telemetry["audio_sync_delta_sec"])

        # Persist telemetry to StoryMemoryBank
        bank = StoryMemoryBank(storage_dir=str(self.base_path / "memory"))
        ok = bank.record_clip_telemetry(scene_number=1, title="Test Scene", telemetry=telemetry)
        self.assertTrue(ok)
        self.assertTrue(bank.telemetry_file.exists())

    # ─────────────────────────────────────────────────────────────────────────
    # 6. StudioPipeline Integration: Narrator Phase Lock & Guided I2V Fallback
    # ─────────────────────────────────────────────────────────────────────────

    @patch("studio_pipeline.run_remote_inference")
    def test_studio_pipeline_narrator_phase_lock_and_guided_i2v_fallback(self, mock_wan):
        # Create a valid dummy MP4 for mock_wan to return
        dummy_mp4 = str(self.base_path / "wan_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(dummy_mp4, fourcc, 16.0, (160, 160))
        for i in range(32):
            frame = np.full((160, 160, 3), (40 + i, 80, 120), dtype=np.uint8)
            vw.write(frame)
        vw.release()
        mock_wan.return_value = (dummy_mp4, "task_test")

        start_pil = Image.new("RGB", (480, 832), color=(50, 90, 130))
        end_pil = Image.new("RGB", (480, 832), color=(60, 100, 140))

        scenes = [
            {
                "scene_number": 1,
                "title": "The Offering",
                "duration_seconds": 5.0,
                "scenario": "Barnaby kneels and offers the berry basket to Pippin.",
                "narration_text": "Barnaby kneels down and offers the glowing berry to Pippin.",
                "narration_voice_style": "warm_storyteller",
                "sync_trigger_word": "offers",
                "keyframe_pil": start_pil,
                "last_keyframe_pil": end_pil,
                "flf2v_status": "EXTREME_DIVERGENCE",  # Should trigger Guided I2V fallback (last_image=None)
            }
        ]

        # 1. Generate narration & word-level timestamps
        self.pipeline.generate_story_narration(scenes, voice_preset="warm_storyteller")
        self.assertTrue(os.path.exists(scenes[0]["narration_audio_path"]))
        self.assertIn("synced_phases", scenes[0])
        synced_p1 = scenes[0]["synced_phases"]["p1"]

        # 2. Compile director plan and verify Phase 1 -> Phase 2 timestamp matches synced_p1
        plan = self.pipeline.compile_director_plan({}, scenes[0], [], {"art_style": "2D cartoon"})
        self.assertIn(f"Phase 1 (0.0s-{synced_p1}s):", plan["wan_prompt"])
        scenes[0]["director"] = plan

        # 3. Animate scene clip: verify Guided I2V fallback (last_image is None due to EXTREME_DIVERGENCE)
        #    and verify audio muxing + telemetry
        clips = self.pipeline.generate_scene_videos(
            scenes=scenes,
            aspect_ratio="9:16 Vertical (Shorts/Reels)",
            base_seed=777,
            use_last_frame=True,
        )
        self.assertEqual(len(clips), 1)
        self.assertTrue(os.path.exists(clips[0]))
        self.assertFalse(scenes[0]["used_last_frame"])
        self.assertIsNone(mock_wan.call_args[1]["processed_last_image"])
        self.assertTrue(scenes[0].get("has_narrator_audio"))
        self.assertIn("telemetry", scenes[0])


if __name__ == "__main__":
    unittest.main()
