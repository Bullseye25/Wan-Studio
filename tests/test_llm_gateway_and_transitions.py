"""
Unit and Integration Tests for Multi-Provider LLM Gateway, Wan LoRA Forwarding,
and Narrative-Driven Cinematic Transitions.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from orchestra_rules import (
    TransitionType,
    build_ffmpeg_transition_filter,
    calculate_transition_srt_timestamps,
)
from narrator_engine import AudioVisualSyncEngine
from studio_pipeline import (
    StudioPipeline,
    LLM_PROVIDERS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_LOCAL_OLLAMA_MODEL,
)


class TestLLMGateway(unittest.TestCase):
    """Verifies Multi-Provider LLM Gateway (GroqCloud, Local Ollama, Ollama Cloud)."""

    def test_llm_providers_configuration(self):
        self.assertIn("groq", LLM_PROVIDERS)
        self.assertIn("local_ollama", LLM_PROVIDERS)
        self.assertIn("ollama_cloud", LLM_PROVIDERS)

        for p_name, cfg in LLM_PROVIDERS.items():
            self.assertIn("name", cfg)
            self.assertIn("endpoint", cfg)
            self.assertIn("default_model", cfg)
            self.assertIn("models", cfg)
            self.assertTrue(len(cfg["models"]) > 0)

    def test_studio_pipeline_provider_resolution(self):
        # 1. Groq explicit key
        p_groq = StudioPipeline(api_key="gsk_test123", provider="groq")
        self.assertEqual(p_groq.provider, "groq")
        self.assertEqual(p_groq.api_key, "gsk_test123")
        self.assertEqual(p_groq.endpoint_url, DEFAULT_GROQ_ENDPOINT)

        # 2. Local Ollama (offline)
        p_local = StudioPipeline(provider="local_ollama")
        self.assertEqual(p_local.provider, "local_ollama")
        self.assertEqual(p_local.api_key, "local")
        self.assertIn("11434", p_local.endpoint_url)

        # 3. Ollama Cloud
        p_cloud = StudioPipeline(provider="ollama_cloud")
        self.assertEqual(p_cloud.provider, "ollama_cloud")
        self.assertTrue(len(p_cloud.api_key) > 10)

        # 4. OpenRouter (Uncensored Action & Hermes 4 405B)
        p_or = StudioPipeline(api_key="sk-or-test12345", provider="openrouter")
        self.assertEqual(p_or.provider, "openrouter")
        self.assertEqual(p_or.api_key, "sk-or-test12345")
        self.assertIn("openrouter.ai", p_or.endpoint_url)

    @patch("urllib.request.urlopen")
    def test_openrouter_connection_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "pong", "role": "assistant"}}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = StudioPipeline.test_llm_connection(
            provider="openrouter",
            api_key="sk-or-mock",
            model="nousresearch/hermes-4-405b",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "openrouter")
        self.assertIn("nousresearch/hermes-4-405b", res["model"])

    @patch("urllib.request.urlopen")
    def test_llm_test_connection_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "pong"}}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = StudioPipeline.test_llm_connection(
            provider="groq",
            api_key="gsk_mock_key",
            model="llama-3.3-70b-versatile",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["provider"], "groq")
        self.assertIn("Connected", res["message"])

    @patch("urllib.request.urlopen")
    def test_llm_test_connection_failure(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("Connection refused by host")

        res = StudioPipeline.test_llm_connection(
            provider="local_ollama",
            model="qwen2.5:7b",
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["provider"], "local_ollama")
        self.assertIn("Connection failed", res["message"])

    def test_model_tag_resolution(self):
        p_groq = StudioPipeline(api_key="gsk_test", provider="groq", model="auto")
        tag_story = p_groq._resolve_model_tag(task_role="screenplay")
        self.assertEqual(tag_story, "llama-3.3-70b-versatile")

        tag_quick = p_groq._resolve_model_tag(task_role="quick_edit")
        self.assertEqual(tag_quick, "llama-3.1-8b-instant")

        tag_vis = p_groq._resolve_model_tag(messages=[{"role": "user", "content": [{"type": "image_url"}]}])
        self.assertEqual(tag_vis, "llama-3.2-11b-vision-preview")


class TestNarrativeTransitions(unittest.TestCase):
    """Verifies heterogeneous per-cut transitions and subtitle sync."""

    def test_multi_cut_heterogeneous_transition_filter(self):
        durations = [5.0, 5.0, 5.0, 5.0]
        per_cut_types = ["crossfade", "fade_black", "whip_pan"]
        per_cut_durations = [0.5, 0.6, 0.4]

        v_filt, a_filt, total_dur = build_ffmpeg_transition_filter(
            clip_durations=durations,
            transition_type=per_cut_types,
            transition_duration=per_cut_durations,
            has_audio=True,
        )

        self.assertIn("xfade", v_filt)
        self.assertIn("transition=fade", v_filt)
        self.assertIn("transition=fadeblack", v_filt)
        self.assertIn("acrossfade", a_filt)
        self.assertAlmostEqual(total_dur, 20.0 - 0.5 - 0.6 - 0.4, places=2)

    def test_narrative_transition_srt_timestamps(self):
        scenes = [
            {"title": "Scene 1", "narration_text": "Hero looks around the cave.", "duration_seconds": 5.0},
            {"title": "Scene 2", "narration_text": "Suddenly a shadowy beast attacks.", "duration_seconds": 5.0},
            {"title": "Scene 3", "narration_text": "With courage, he wields his sword.", "duration_seconds": 5.0},
        ]
        # 2 cuts: Cut 1 is crossfade (0.5s), Cut 2 is fade_black (0.5s)
        per_cut = ["crossfade", "fade_black"]
        intervals = calculate_transition_srt_timestamps(
            scene_durations=[s["duration_seconds"] for s in scenes],
            transition_type=per_cut,
            transition_duration=0.5,
        )

        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0][0], 0.0)
        self.assertEqual(intervals[0][1], 5.0)

        # Scene 2 should start shifted earlier by cut 1 duration (5.0 - 0.5 = 4.5)
        self.assertAlmostEqual(intervals[1][0], 4.5, places=2)
        # Scene 2 ends at 4.5 + 5.0 = 9.5
        self.assertAlmostEqual(intervals[1][1], 9.5, places=2)

        # Scene 3 starts shifted by cut 1 (0.5) + cut 2 (0.5) = 1.0 (10.0 - 1.0 = 9.0)
        self.assertAlmostEqual(intervals[2][0], 9.0, places=2)


class TestStudioPipelineEnhancements(unittest.TestCase):
    """Verifies Color Script, LoRA forwarding, and Narrative Transition resolution."""

    def test_normalize_story_color_script_and_transitions(self):
        raw_story = {
            "title": "Heart of Iron",
            "art_style": "2D anime cel",
            "characters": [{"id": "hero", "name": "Aiden"}],
            "scenes": [
                {"scene_number": 1, "scenario": "Aiden awakes"},
                {"scene_number": 2, "scenario": "Aiden finds the spark"},
            ],
        }

        normalized = StudioPipeline.normalize_story(
            raw_story,
            prompt="Heart of Iron premise",
            num_scenes=2,
            target_duration=10.0,
            art_style="2D anime cel",
        )

        # 1. Color script present
        self.assertIn("color_and_lighting_script", normalized)
        self.assertIn("palette", normalized["color_and_lighting_script"])
        self.assertIn("lighting_progression", normalized["color_and_lighting_script"])

        # 2. Per-scene recommended transitions present
        scenes = normalized["scenes"]
        self.assertEqual(len(scenes), 2)
        for s in scenes:
            self.assertIn("recommended_transition", s)

    @patch("studio_pipeline.run_remote_inference")
    def test_lora_passthrough_in_animate_single_clip(self, mock_run_remote):
        mock_run_remote.return_value = ["dummy_video.mp4"]
        pipeline = StudioPipeline(provider="groq", api_key="gsk_test")

        dummy_img = Image.new("RGB", (480, 832), color=(20, 30, 40))
        scene = {
            "scene_number": 1,
            "title": "Scene 1",
            "keyframe_pil": dummy_img,
            "duration_seconds": 3.5,
            "director": {"wan_prompt": "cinematic camera move", "duration_seconds": 3.5},
        }

        with patch("studio_pipeline.verify_video_clip_quality") as mock_verify:
            mock_verify.return_value = {"sharpness": 100.0}
            with patch("builtins.open", unittest.mock.mock_open(read_data=b"video_bytes")):
                pipeline._animate_single_scene_clip(
                    scene=scene,
                    aspect_ratio="9:16 Vertical (Shorts/Reels)",
                    seed=42,
                    use_last_frame=False,
                    lora_repo_or_path="custom_lora_style.safetensors",
                    lora_scale=0.75,
                )

        self.assertTrue(mock_run_remote.called)
        kwargs = mock_run_remote.call_args.kwargs
        self.assertEqual(kwargs.get("lora_repo_or_path"), "custom_lora_style.safetensors")
        self.assertEqual(kwargs.get("lora_scale"), 0.75)


if __name__ == "__main__":
    unittest.main()
