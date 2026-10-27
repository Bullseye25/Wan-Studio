"""
Unit Test Suite for ProductionTelemetryCollector (production_telemetry.py).
Verifies stage latency tracking, character sheet isolation metrics, scene keyframe/narration/optical-flow
telemetry compilation, recommendation engine rules, and JSON/Markdown report persistence.
"""

import os
import json
import tempfile
import unittest
from PIL import Image
import numpy as np
import cv2

from production_telemetry import ProductionTelemetryCollector


class TestProductionTelemetry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.collector = ProductionTelemetryCollector(base_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stage_timing_lifecycle(self):
        self.collector.start_stage("1_screenplay_generation")
        dur = self.collector.end_stage("1_screenplay_generation", metadata={"scenes": 6})
        self.assertGreaterEqual(dur, 0.0)
        self.assertEqual(self.collector.stage_timings["1_screenplay_generation"]["status"], "completed")
        self.assertEqual(self.collector.stage_timings["1_screenplay_generation"]["scenes"], 6)

    def test_analyze_character_portraits_and_isolation(self):
        char_img = np.full((256, 144, 3), 240, dtype=np.uint8)
        # Draw a crisp dark rectangle in the center so Laplacian sharpness > 25 and border stays uniform
        cv2.rectangle(char_img, (40, 50), (104, 200), (20, 40, 80), -1)
        char_path = os.path.join(self.temp_dir.name, "silas.png")
        Image.fromarray(char_img).save(char_path)

        chars = [
            {
                "id": "char_1",
                "name": "Silas",
                "portrait_path": char_path,
                "portrait_seed": 101,
                "z_image_prompt": "Full-body portrait of Silas on clean neutral studio background",
            }
        ]
        res = self.collector.analyze_character_portraits(chars)
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0]["exists"])
        self.assertEqual(res[0]["width"], 144)
        self.assertEqual(res[0]["height"], 256)
        self.assertTrue(res[0]["clean_studio_backdrop"])
        self.assertEqual(len(res[0]["prompt_violations"]), 0)

    def test_compile_and_save_reports_and_recommendations(self):
        # Create dummy movie and srt
        movie_path = os.path.join(self.temp_dir.name, "final_30s.mp4")
        writer = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (160, 120))
        for _ in range(32):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        srt_path = os.path.join(self.temp_dir.name, "final_30s.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,200 --> 00:00:04,500\nSilas finds a tiny frozen robin.\n\n")

        story_data = {
            "title": "The Winter Robin",
            "art_style": "Storybook 2D",
            "theme": "Compassion heals",
            "seeds": {"z_image_seed": 111, "qwen_seed": 222, "wan_seed": 333},
            "logic_audit": {
                "overall_score": 9.4,
                "subscores": {"causality": 9.5, "feasibility_5s": 9.5, "prop_permanence": 9.0, "spatial_continuity": 9.5, "visual_delta": 9.5},
            },
            "characters": [{"id": "char_1", "name": "Silas"}, {"id": "char_2", "name": "Pip"}],
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "The Frosted Sill",
                    "narrative_beat": "Hook",
                    "causal_connector": "SETUP",
                    "narration_text": "On a frozen morning, Silas finds a shivering robin.",
                    "sync_trigger_word": "finds",
                    "pose_delta": 12.5,  # Intentionally < 18.0 to trigger HIGH recommendation
                    "flf2v_compat": {"pose_delta": 12.5, "bg_ssim": 0.88, "status": "TOO_STATIC"},
                    "director": {"source": "gemma-vision", "camera_movement": "slow push-in", "guidance_scale": 2.0, "guidance_scale_2": 1.5, "flow_shift": 4.5, "steps": 8},
                    "telemetry": {
                        "passed": False,
                        "mean_flow": 0.12,
                        "surge_ratio": 6.2,
                        "sharpness_retention": 0.52,
                        "peak_flow_sec": 2.4,
                        "audio_sync_delta_sec": 0.95,
                        "issues": ["Low flow"],
                    },
                    "video_path": movie_path,
                }
            ],
            "final_movie_path": movie_path,
            "subtitle_srt_path": srt_path,
        }

        self.collector.start_stage("5_video_animation_wan22")
        self.collector.end_stage("5_video_animation_wan22")

        report = self.collector.compile_and_save_reports(story_data, transition_type="crossfade", transition_duration=0.5)
        self.assertTrue(os.path.exists(report["json_report_path"]))
        self.assertTrue(os.path.exists(report["markdown_report_path"]))
        self.assertTrue(os.path.exists(self.collector.telemetry_log_file))

        categories = {r["category"] for r in report["improvement_recommendations"]}
        self.assertIn("Qwen Keyframe Pose Delta", categories)
        self.assertIn("Wan 2.2 Temporal Stability", categories)
        self.assertIn("Wan 2.2 Late-Frame Sharpness", categories)
        self.assertIn("Audio-Visual Narrator Sync", categories)

        md_text = open(report["markdown_report_path"], encoding="utf-8").read()
        self.assertIn("The Winter Robin", md_text)
        self.assertIn("Smart Continuous Improvement Recommendations", md_text)


if __name__ == "__main__":
    unittest.main()
