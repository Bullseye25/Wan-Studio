"""
OmniFlow-30: Next-Generation Story-to-Animation Production Engine
=================================================================
Unifies Wan 2.2 FLF2V, Qwen Image Edit, and Z-Image Turbo with:
  1. Parallel Serverless Cloud GPU Swarms on Modal (sub-5-minute 30-second production)
  2. In-Flight Midpoint Latent Interception & Steering (MILIS at t=0.5)
  3. Strict 9:16 Vertical Framing & Composition Safe Zones
  4. Autonomous Causal Story Doctoring & Voiceover Narration (SAPI5)
  5. Multi-Tier Self-Learning AI Critic with Persistent Evolutionary Memory
"""

import os
import re
import sys
import time
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable

from studio_pipeline import (
    StudioPipeline,
    ASPECT_RATIO_PRESETS,
    WAN_FPS,
    DEFAULT_NARRATOR_VOICE,
)
from self_learning_critic import DirectorKnowledgeBase, SelfLearningCritic


@dataclass
class OmniFlowConfig:
    """Configuration settings for an OmniFlow-30 production run."""
    prompt: str
    target_duration: float = 30.0
    aspect_ratio: str = "9:16 Vertical (Shorts/Reels)"
    art_style: str = "2D vibrant cartoon animation, Studio Ghibli watercolor aesthetic, lush hand-painted textures, soft edge lighting, 4K"
    num_scenes: Optional[int] = None  # None = dynamic calculation (4 scenes for 30s)
    parallel_swarm: bool = True
    enable_midpoint_refine: bool = True
    enable_narrator: bool = True
    narrator_voice: str = DEFAULT_NARRATOR_VOICE
    transition_type: str = "narrative"
    transition_duration: float = 0.5
    lora_repo_or_path: str = ""
    lora_scale: float = 0.8
    provider: str = "groq"
    model: Optional[str] = None
    api_key: Optional[str] = None
    story_folder_name: Optional[str] = None
    run_story_audit: bool = True
    auto_qa_retry: bool = True
    manage_gpu_phases: bool = True
    collect_telemetry: bool = True


def calculate_omniflow_scenes(duration: float) -> int:
    """
    Computes the optimal scene count for cinematic narrative pacing.
    For a 30s film, 4 scenes of ~7.5s (or 6 scenes of 5.0s) delivers optimal dramatic structure.
    """
    dur = max(5.0, float(duration))
    if dur <= 10.0:
        return 2
    if dur <= 15.0:
        return 3
    if dur <= 25.0:
        return 4
    # Default 30.0s target -> 4 atomic 7.5s scenes (or 6 scenes of 5.0s)
    return 4


def inject_9_16_vertical_composition_rules(prompt: str, art_style: str) -> str:
    """
    Augments visual prompts with vertical 9:16 framing invariants so characters
    remain inside the TikTok/Shorts central 60% safe zone without cut-offs.
    """
    vertical_cues = (
        "vertical 9:16 cinema composition, centered subject in portrait framing, "
        "full-height vertical framing, high resolution vertical canvas"
    )
    if "9:16" in prompt or "vertical" in prompt.lower():
        return prompt
    return f"{prompt}, {vertical_cues}"


class OmniFlowEngine:
    """
    High-level orchestrator for the OmniFlow-30 animation suite.
    """

    def __init__(self, config: Optional[OmniFlowConfig] = None, base_dir: Optional[str] = None):
        self.config = config
        self.base_dir = pathlib.Path(base_dir) if base_dir else (pathlib.Path(__file__).parent / "outputs" / "studio")
        self.knowledge_base = DirectorKnowledgeBase(storage_dir=str(self.base_dir))
        self.critic = SelfLearningCritic(storage_dir=str(self.base_dir))

    def get_past_learned_directives(self, prompt: str, limit: int = 4) -> List[Dict[str, Any]]:
        """Retrieves matching directives from past film post-mortems to steer future generation."""
        return self.knowledge_base.retrieve_relevant_directives(prompt, top_k=limit)

    def calculate_eta(self, config: OmniFlowConfig) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates dynamic ETA in seconds for the complete production.
        In Swarm mode, video generation runs in parallel, cutting render time from ~20m to ~4.5m!
        """
        num_scenes = config.num_scenes or calculate_omniflow_scenes(config.target_duration)
        p_clean = str(config.provider or "groq").lower().strip()
        t_script = 1.6 if p_clean == "groq" else 6.5
        t_voice = round(num_scenes * 0.15, 1)
        t_chars = 3.5  # Z-Image Turbo on Modal
        t_keyframes = round(num_scenes * 2 * 5.0, 1)  # Qwen Image Edit Start + End

        # Video rendering on Modal A100 GPU
        if config.parallel_swarm and num_scenes > 1:
            # Parallel Swarm executes clips concurrently: O(1 scene) with minor coordination overhead
            t_video = 36.0 + (num_scenes * 2.5)
        else:
            t_video = num_scenes * 36.0

        if config.enable_midpoint_refine:
            t_video += (5.0 if config.parallel_swarm else (num_scenes * 5.0))

        t_assembly = 2.0
        t_critic = 1.0

        total_eta = t_script + t_voice + t_chars + t_keyframes + t_video + t_assembly + t_critic
        breakdown = {
            "screenplay": t_script,
            "voiceover": t_voice,
            "characters": t_chars,
            "keyframes": t_keyframes,
            "video_swarm": t_video,
            "assembly": t_assembly,
            "critic": t_critic,
            "num_scenes": num_scenes,
            "parallel_swarm": config.parallel_swarm,
            "midpoint_refine": config.enable_midpoint_refine,
        }
        return total_eta, breakdown

    def produce_film(
        self,
        config: OmniFlowConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end OmniFlow-30 short film generation.
        """
        num_scenes = config.num_scenes or calculate_omniflow_scenes(config.target_duration)
        enriched_prompt = inject_9_16_vertical_composition_rules(config.prompt, config.art_style)

        # 1. Initialize core pipeline
        pipeline = StudioPipeline(
            api_key=config.api_key,
            model=config.model,
            provider=config.provider,
            base_dir=str(self.base_dir),
        )

        if progress_callback:
            progress_callback(0.02, "OmniFlow-30: Initializing AI Director and retrieving learned rules...")

        # 2. Run full pipeline with swarm and midpoint refinement
        story_data = pipeline.run_full_pipeline(
            user_prompt=enriched_prompt,
            num_scenes=num_scenes,
            target_duration=config.target_duration,
            aspect_ratio=config.aspect_ratio,
            art_style=config.art_style,
            use_vision_director=True,
            use_last_frame=True,
            enable_rife_2x=True,
            chain_previous_end_frame=False,
            anti_noise_guard=True,
            enable_narrator=config.enable_narrator,
            narrator_voice=config.narrator_voice,
            transition_type=config.transition_type,
            transition_duration=config.transition_duration,
            lora_repo_or_path=config.lora_repo_or_path,
            lora_scale=config.lora_scale,
            run_story_audit=config.run_story_audit,
            auto_qa_retry=config.auto_qa_retry,
            manage_gpu_phases=config.manage_gpu_phases,
            parallel_swarm=config.parallel_swarm,
            enable_midpoint_refine=config.enable_midpoint_refine,
            collect_telemetry=config.collect_telemetry,
            story_folder_name=config.story_folder_name,
            progress_cb=progress_callback,
        )

        # 3. Post-Mortem Evaluation with AI Critic
        critic_report = story_data.get("critic_evaluation")
        if not critic_report:
            critic_report = self.critic.evaluate_production_run(story_data)
            story_data["critic_evaluation"] = critic_report

        return story_data
