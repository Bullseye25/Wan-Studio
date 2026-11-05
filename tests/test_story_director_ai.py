"""
Unit Test Suite for Story Director AI, Tone-Aware Dynamic Poses,
Multi-Pass Writers' Room, Banned Vocabulary Sanitizer & Performance Scripting.
=============================================================================
"""

import pytest
from typing import Dict, Any

from story_director_ai import (
    detect_story_genre,
    sanitize_genre_prompt,
    get_tone_aware_end_pose,
    build_beat_sheet_prompt,
    build_scene_drafting_prompt,
    build_story_editor_audit_prompt,
    build_performance_script,
    audit_screenplay_heuristics,
    GENRE_CONFIGS,
    TONE_AWARE_POSE_LIBRARY,
)


class TestStoryGenreDetection:
    """Tests intelligent genre taxonomy inference from story premise and art style."""

    def test_detect_kids_genre(self):
        premise = "A little bear and rabbit share a basket of ripe berries in the meadow"
        art_style = "Cute vibrant 3D cartoon animation for kids, Pixar storybook style"
        assert detect_story_genre(premise, art_style) == "kids"

    def test_detect_mystery_genre(self):
        premise = "Batman tracks Joker through the dark rainy alleys of Arkham asylum"
        art_style = "Atmospheric noir cinematic animation, neon shadows"
        assert detect_story_genre(premise, art_style) == "mystery"

    def test_detect_adventure_genre(self):
        premise = "A young samurai journeys across ancient dragon ruins to recover a lost relic"
        art_style = "Dynamic cinematic 3D animation, Makoto Shinkai lighting"
        assert detect_story_genre(premise, art_style) == "adventure"


class TestToneAwarePoseLibrary:
    """Verifies that kids stories never receive violent, combat, or defensive poses."""

    def test_kids_pose_never_contains_combat_vocabulary(self):
        kids_banned = GENRE_CONFIGS["kids"]["banned_words"]
        kids_poses = TONE_AWARE_POSE_LIBRARY["kids"]

        for action_key, pose_desc in kids_poses.items():
            pose_lower = pose_desc.lower()
            for banned in kids_banned:
                assert banned not in pose_lower, f"Action '{action_key}' contains banned word '{banned}' in kids pose: {pose_desc}"

    def test_kids_share_pose_selection(self):
        pose = get_tone_aware_end_pose("Barnaby decides to share his berries", genre="kids")
        assert "basket" in pose or "paws" in pose or "smile" in pose
        assert "shockwave" not in pose
        assert "cover" not in pose

    def test_kids_eat_pose_selection(self):
        pose = get_tone_aware_end_pose("Pip happily eats a sweet berry", genre="kids")
        assert "nibbling" in pose or "chewing" in pose or "delight" in pose

    def test_adventure_pose_selection(self):
        pose = get_tone_aware_end_pose("The hero leaps across the chasm", genre="adventure")
        assert "knee" in pose or "stance" in pose or "heroic" in pose


class TestPromptSanitization:
    """Tests strip-and-replace filtering of genre-inappropriate vocabulary."""

    def test_sanitize_shockwave_in_kids(self):
        dirty = "The character releases a glowing shockwave of energy"
        clean = sanitize_genre_prompt(dirty, genre="kids")
        assert "shockwave" not in clean.lower()
        assert "burst of colorful sparkles" in clean

    def test_sanitize_behind_cover_in_kids(self):
        dirty = "Barnaby ducks down defensively behind cover"
        clean = sanitize_genre_prompt(dirty, genre="kids")
        assert "behind cover" not in clean.lower()
        assert "defensively" not in clean.lower()
        assert "warmly together" in clean

    def test_sanitize_weapon_in_kids(self):
        dirty = "The warrior holds a weapon in hand"
        clean = sanitize_genre_prompt(dirty, genre="kids")
        assert "weapon" not in clean.lower()
        assert "friendly gift" in clean


class TestMultiPassPromptBuilders:
    """Verifies Pass 1, Pass 2, and Pass 3 prompt construction contracts."""

    def test_beat_sheet_prompt_builder(self):
        sys_p, usr_p = build_beat_sheet_prompt(
            premise="Two friends learn to share",
            genre="kids",
            num_scenes=2,
            target_duration=10.0,
        )
        assert "CAUSAL BEAT SHEET" in sys_p
        assert "THEREFORE" in sys_p
        assert "Kids Storybook" in sys_p
        assert "Two friends learn to share" in usr_p

    def test_scene_drafting_prompt_builder(self):
        dummy_beats = {
            "logline": "Two friends share berries",
            "beats": [{"scene_number": 1, "dramatic_role": "Want", "causal_connector": "SETUP"}]
        }
        sys_p, usr_p = build_scene_drafting_prompt(
            beat_sheet=dummy_beats,
            premise="Two friends learn to share",
            genre="kids",
            num_scenes=2,
            target_duration=10.0,
            art_style="Cute 3D cartoon",
        )
        assert "master animation director" in sys_p
        assert "STRICT 5-SECOND ATOMICITY" in sys_p
        assert "CEFR B1" in sys_p
        assert "Approved Beat Sheet" in usr_p

    def test_story_editor_audit_prompt_builder(self):
        dummy_story = {
            "title": "Sharing Friends",
            "characters": [{"id": "char_1", "name": "Barnaby"}],
            "scenes": [{"scene_number": 1, "narration_text": "Barnaby picks ripe berries."}]
        }
        sys_p, usr_p = build_story_editor_audit_prompt(dummy_story, genre="kids")
        assert "Story Doctor and Script Editor" in sys_p
        assert "b1_readability" in sys_p
        assert "Sharing Friends" in usr_p


class TestNarratorPerformanceScripting:
    """Verifies that performance scripting attaches expressive attributes."""

    def test_build_performance_script(self):
        scene = {
            "scene_number": 1,
            "narration_text": "Barnaby picks sweet berries in the sunny morning.",
            "narration_voice_style": "warm_wonder",
            "sync_trigger_word": "picks",
        }
        script = build_performance_script(scene, genre="kids")
        assert script["text"] == "Barnaby picks sweet berries in the sunny morning."
        assert script["emotion"] == "warm_wonder"
        assert script["speed"] == 0.96
        assert script["emphasis_word"] == "picks"
        assert script["pause_after_sec"] >= 0.35


class TestScreenplayHeuristicAuditor:
    """Verifies deterministic heuristic scoring of screenplay causality, tone, and readability."""

    def test_high_quality_storybook_passes_audit(self):
        good_story: Dict[str, Any] = {
            "title": "Barnaby's Sweet Gift",
            "genre": "kids",
            "characters": [
                {"id": "char_1", "name": "Barnaby"},
                {"id": "char_2", "name": "Pip"},
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "causal_connector": "SETUP",
                    "narration_text": "Barnaby finds ripe berries in the warm meadow.",
                    "scenario": "Barnaby smiles warmly holding the basket of berries.",
                    "qwen_scene_prompt": "Barnaby holding berry basket with a happy smile.",
                    "qwen_end_scene_prompt": "Barnaby happily waves to Pip with sparkling eyes.",
                    "wan_motion_prompt": "Barnaby turns and smiles warmly.",
                },
                {
                    "scene_number": 2,
                    "causal_connector": "THEREFORE",
                    "narration_text": "Pip smiles gladly as they share the treat.",
                    "scenario": "Pip hugs Barnaby happily as both friends eat sweet berries together.",
                    "qwen_scene_prompt": "Barnaby gently offers the berry basket to Pip.",
                    "qwen_end_scene_prompt": "Both friends sit together eating berries with joyful smiles.",
                    "wan_motion_prompt": "Barnaby hands the basket to Pip, who jumps with joy.",
                },
            ],
        }
        report = audit_screenplay_heuristics(good_story, genre="kids")
        assert report["overall_score"] >= 8.0
        assert report["passed"] is True
        assert report["critical_fail"] is False

    def test_tone_violation_triggers_audit_failure(self):
        bad_story: Dict[str, Any] = {
            "title": "Clash in the Forest",
            "genre": "kids",
            "characters": [{"id": "char_1", "name": "Barnaby"}],
            "scenes": [
                {
                    "scene_number": 1,
                    "causal_connector": "AND THEN",  # Forbidden lazy connector
                    "narration_text": "Barnaby discharges an incomprehensible antimatter shockwave defensively.",  # Literary jargon + banned
                    "scenario": "Barnaby fires weapons and ducks down defensively behind cover.",  # Banned combat vocabulary
                    "qwen_scene_prompt": "Barnaby shooting guns.",
                    "qwen_end_scene_prompt": "Barnaby shooting guns.",  # Frozen keyframe delta
                    "wan_motion_prompt": "Static shot.",
                }
            ],
        }
        report = audit_screenplay_heuristics(bad_story, genre="kids")
        assert report["overall_score"] < 6.0
        assert report["critical_fail"] is True
        assert report["passed"] is False
        assert any("banned word" in iss for iss in report["issues"])
        assert any("forbidden connector 'AND THEN'" in iss for iss in report["issues"])


class TestQualityGatePipelineProtection:
    """Verifies that an unrepairable low-scoring screenplay halts before GPU allocation."""

    def test_run_full_pipeline_hard_quality_gate(self):
        from studio_pipeline import StudioPipeline

        pipeline = StudioPipeline(api_key="mock", provider="groq")

        # Mock an audit result that scored below 6.0
        bad_story = {
            "title": "Unviable Story",
            "scenes": [{"scene_number": 1, "narration_text": "test"}],
        }

        # Override story generation and audit to return synthetic failure
        pipeline.generate_story_and_characters = lambda *args, **kwargs: bad_story
        pipeline.audit_and_repair_story_logic = lambda s, **kwargs: {
            "passed": False,
            "overall_score": 4.5,
            "issues": ["Incoherent causality", "Multiple tone contradictions"],
            "story": s,
        }

        # Attempt to run full pipeline; must raise ValueError before GPU initialization
        with pytest.raises(ValueError) as excinfo:
            pipeline.run_full_pipeline(
                user_prompt="Bad story prompt",
                num_scenes=1,
                run_story_audit=True,
                enable_narrator=False,
                manage_gpu_phases=False,
            )

        assert "Story Doctor Quality Gate Abort" in str(excinfo.value)
        assert "4.5" in str(excinfo.value)
        assert "Aborting before GPU activation to protect compute budget" in str(excinfo.value)


class TestCharacterScaleAndPropInvariants:
    """Verifies character scale locking and prop synchronization engines."""

    def test_infer_character_scale(self):
        from story_director_ai import infer_character_scale

        bear = infer_character_scale({"name": "Barnaby", "description": "A lovable chocolate-brown grizzly bear"})
        assert bear["species"] == "bear"
        assert bear["scale_factor"] == 3.5
        assert bear["category"] == "giant"

        rabbit = infer_character_scale({"name": "Pip", "description": "A fluffy little white bunny rabbit"})
        assert rabbit["species"] == "rabbit"
        assert rabbit["scale_factor"] == 1.0
        assert rabbit["category"] == "tiny"

        child = infer_character_scale({"name": "Timmy", "description": "A curious little 6-year-old boy"})
        assert child["species"] == "human_child"
        assert child["scale_factor"] == 1.4

    def test_build_character_scale_anchor(self):
        from story_director_ai import build_character_scale_anchor

        chars = [
            {"name": "Barnaby", "description": "brown bear"},
            {"name": "Pip", "description": "white rabbit"},
        ]
        anchor = build_character_scale_anchor(chars)
        assert "SCALE INVARIANT:" in anchor
        assert "Barnaby is a massive bear (3.5x larger and taller than Pip)" in anchor
        assert "reaching only up to Barnaby's knee" in anchor
        assert "NEVER shrink Barnaby" in anchor
        assert "3.5:1" in anchor

    def test_audit_and_harmonize_props(self):
        from story_director_ai import audit_and_harmonize_props

        story = {
            "title": "Barnaby's Sweet Treat",
            "world_bible": {},
            "scenes": [
                {
                    "scene_number": 1,
                    "narration_text": "Barnaby finds sweet blueberries in the meadow.",
                    "scenario": "Barnaby looks down at a basket of ripe red strawberries.",
                    "qwen_scene_prompt": "Barnaby holding a bright red strawberry.",
                    "qwen_end_scene_prompt": "Barnaby eating a red strawberry with Pip.",
                    "wan_motion_prompt": "Barnaby offers the basket of blueberries.",
                }
            ],
        }

        harmonized = audit_and_harmonize_props(story)
        s1 = harmonized["scenes"][0]

        # Narration must be harmonized to strawberries
        assert "strawberries" in s1["narration_text"]
        assert "blueberries" not in s1["narration_text"]

        # Motion prompt must be harmonized to strawberries
        assert "strawberries" in s1["wan_motion_prompt"]
        assert "blueberries" not in s1["wan_motion_prompt"]

        # World bible must have locked_prop declared
        assert "locked_prop" in harmonized["world_bible"]
        assert harmonized["world_bible"]["locked_prop"]["canonical_name"] == "strawberries"

    def test_prop_contradiction_penalizes_heuristic_audit(self):
        from story_director_ai import audit_screenplay_heuristics

        contradictory_story = {
            "title": "Berry Mixup",
            "genre": "kids",
            "characters": [
                {"id": "char_1", "name": "Barnaby", "description": "bear"},
                {"id": "char_2", "name": "Pip", "description": "rabbit"},
            ],
            "scenes": [
                {
                    "scene_number": 1,
                    "causal_connector": "SETUP",
                    "narration_text": "They sit together and eat sweet blueberries happily.",
                    "scenario": "Barnaby and Pip are eating ripe red strawberries.",
                    "qwen_scene_prompt": "Holding bright red strawberries.",
                    "qwen_end_scene_prompt": "Chewing on red strawberries happily.",
                    "wan_motion_prompt": "Eating berries.",
                }
            ],
        }

        report = audit_screenplay_heuristics(contradictory_story, genre="kids")
        assert report["subscores"]["prop_continuity"] <= 6.0
        assert any("Prop contradiction" in iss for iss in report["issues"])

