"""
sprites_pipeline.py - Universal 2D Game Sprites Creator Engine
Integrates:
  - Gemma 4: 31B as Game Character Architect & Animation Choreographer
  - Z-Image Turbo on Modal for Master Chroma-Key Character Sprite Synthesis
  - Wan 2.2 on Modal for Temporal Motion & Animation Video Synthesis (Looping vs Return-to-Idle)
  - sprite_cutter.py for Alpha Matting, Frame Extraction, and Atlas Sheet Packing
"""

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from backend import (
    generate_z_image_turbo_remote,
    run_remote_inference,
    DEFAULT_NEGATIVE_PROMPT,
)
from sprite_cutter import process_video_to_sprite_sheet


DEFAULT_OLLAMA_API_KEY = "ca8516405cc74a238c9525b490331d3a.HmyaGAbsmLFyYN25P1K8_tG6"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_ENDPOINT = "https://ollama.com/v1/chat/completions"

# Master sprite generation dimensions (9:16 vertical orientation for full head-to-toe overview)
SPRITE_BASE_WIDTH = 576
SPRITE_BASE_HEIGHT = 1024

GENRE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Action Runner (Pepsiman / Crash Bandicoot)": {
        "perspective": "dynamic 3/4 perspective runner, side-scrolling rightward motion",
        "camera": "fixed static arcade camera, lateral side-view plane, zero pan, zero tilt",
        "default_moves": [
            {
                "name": "Idle",
                "type": "loop",
                "description": "High-energy runner stance, rhythmic panting breath with occasional energetic thumbs up to camera",
            },
            {
                "name": "Full Sprint",
                "type": "loop",
                "description": "Frantic forward dash, rapid arm pumping and legs in full stride sprint",
            },
            {
                "name": "Lane Switch Dodge",
                "type": "loop",
                "description": "Rapid lateral dodge slide from side to side while running",
            },
            {
                "name": "Hurdle Slide",
                "type": "action",
                "description": "Dives into a low baseball slide under a barrier, quickly recovering back to run stance",
            },
            {
                "name": "Vault Jump",
                "type": "action",
                "description": "High hurdle leap into the air over an obstacle, landing cleanly in run stance",
            },
            {
                "name": "Stumble Trip",
                "type": "action",
                "description": "Comedic trip and wobble over a small hazard, catching balance and returning to stance",
            },
            {
                "name": "Can Chug / Victory",
                "type": "action",
                "description": "Pulls out a cold soda can, chugs it down in one gulp, flashes a proud thumbs-up flex, returns to stance",
            },
        ],
    },
    "Action Platformer (Mega Man X / Hollow Knight)": {
        "perspective": "2D side-scrolling platformer view, facing right",
        "camera": "fixed static 2D platformer camera, zero camera movement",
        "default_moves": [
            {
                "name": "Idle Stance",
                "type": "loop",
                "description": "Armed combat stance, arm cannon humming with energy glow, gentle breathing",
            },
            {
                "name": "Forward Run",
                "type": "loop",
                "description": "Aggressive forward sprint with arm cannon aimed forward",
            },
            {
                "name": "Ground Dash",
                "type": "loop",
                "description": "Low crouch booster dash with ground friction sparks",
            },
            {
                "name": "Buster Shot",
                "type": "action",
                "description": "Fires a rapid plasma bolt from the arm cannon with recoil kickback, returning to idle stance",
            },
            {
                "name": "Charged Blast",
                "type": "action",
                "description": "Channels a massive glowing energy blast, firing a giant beam with heavy recoil, returning to idle stance",
            },
            {
                "name": "Jump Apex",
                "type": "loop",
                "description": "Airborne hover leap with knees bent and weapon ready",
            },
            {
                "name": "Hurt Knockback",
                "type": "action",
                "description": "Knocked backward by an impact with electric sparks, landing firmly back in stance",
            },
        ],
    },
    "Top-Down RPG (Zelda / Chrono Trigger)": {
        "perspective": "top-down 3/4 overhead bird's-eye RPG perspective",
        "camera": "fixed overhead RPG camera angle, static view",
        "default_moves": [
            {
                "name": "Idle Stance",
                "type": "loop",
                "description": "Standing ready with equipped weapon, subtle breathing bounce and cloak flutter",
            },
            {
                "name": "Walk Down (South)",
                "type": "loop",
                "description": "Walking downward facing toward the camera with rhythmic steps",
            },
            {
                "name": "Walk Up (North)",
                "type": "loop",
                "description": "Walking upward facing away from the camera into the distance",
            },
            {
                "name": "Walk Side (East/West)",
                "type": "loop",
                "description": "Walking sideways along horizontal axis with lateral stride",
            },
            {
                "name": "Sword Slash Combo",
                "type": "action",
                "description": "Swift horizontal sword swing arc with white blade streak, returning to guard pose",
            },
            {
                "name": "Magic Spell Cast",
                "type": "action",
                "description": "Raises staff or hands high, summoning a pulsing magical sigil ring, returning to guard",
            },
            {
                "name": "Item Use / Victory",
                "type": "action",
                "description": "Lifts treasure or potion bottle high above head with celebratory gleam, returning to guard",
            },
        ],
    },
    "Arcade Shmup (Space Invaders / Gradius)": {
        "perspective": "overhead top-down or horizontal shooter perspective",
        "camera": "fixed static arcade screen view, zero camera pan",
        "default_moves": [
            {
                "name": "Hover / Thrusters",
                "type": "loop",
                "description": "Hovering in space with blue ion exhaust plumes firing rhythmically from rear engines",
            },
            {
                "name": "Bank Left Roll",
                "type": "loop",
                "description": "Angled bank and tilt to the left showing ship top armor plating",
            },
            {
                "name": "Bank Right Roll",
                "type": "loop",
                "description": "Angled bank and tilt to the right showing ship top armor plating",
            },
            {
                "name": "Plasma Cannons Volley",
                "type": "action",
                "description": "Dual cannons pulse rapidly with bright laser muzzle flares, returning to cruise hover",
            },
            {
                "name": "Torpedo Launch",
                "type": "action",
                "description": "Bays open and launch a heavy homing photon torpedo with smoke trail, returning to cruise",
            },
            {
                "name": "Energy Shield Pulse",
                "type": "action",
                "description": "Hexagonal energy barrier expands around the hull and absorbs an impact, settling back to cruise",
            },
            {
                "name": "Explosion / Defeat",
                "type": "action",
                "description": "Ship takes fatal damage and detonates into a multi-stage arcade fire explosion",
            },
        ],
    },
    "Beat 'em Up (Streets of Rage / Final Fight)": {
        "perspective": "2.5D belt-scrolling side view, heroic proportions",
        "camera": "fixed static arcade camera, lateral plane, zero camera pan",
        "default_moves": [
            {
                "name": "Combat Idle",
                "type": "loop",
                "description": "Boxer bounce combat stance, hands up guarding chin, rhythmic foot shuffle",
            },
            {
                "name": "Walk Forward",
                "type": "loop",
                "description": "Confident forward combat stride closing distance on opponents",
            },
            {
                "name": "Depth Walk (Up/Down)",
                "type": "loop",
                "description": "Diagonal belt-scroll walk moving up or down the street plane",
            },
            {
                "name": "Sprint Dash",
                "type": "loop",
                "description": "Fast rushing sprint forward closing distance aggressively",
            },
            {
                "name": "3-Hit Combo Strike",
                "type": "action",
                "description": "Rapid left jab, right cross, followed by a fierce roundhouse kick, recovering back to guard",
            },
            {
                "name": "Heavy Knockdown Uppercut",
                "type": "action",
                "description": "Deep lunging rising uppercut with windup and impact freeze, returning to guard",
            },
            {
                "name": "Special Arcade Move",
                "type": "action",
                "description": "Spinning flaming 360-degree whirlwind kick with light trails, settling back to guard",
            },
        ],
    },
    "Custom / Hybrid Genre": {
        "perspective": "user defined 2D game perspective",
        "camera": "fixed static camera, zero camera drift",
        "default_moves": [
            {
                "name": "Idle",
                "type": "loop",
                "description": "Character breathing and resting stance",
            },
            {
                "name": "Move Primary",
                "type": "loop",
                "description": "Main forward locomotion",
            },
            {
                "name": "Primary Action",
                "type": "action",
                "description": "Primary attack or skill returning to idle stance",
            },
        ],
    },
}


class SpritesPipeline:
    """
    Orchestrates the 4-phase 2D game sprites creation workflow.
    """

    def __init__(
        self,
        api_key: str = DEFAULT_OLLAMA_API_KEY,
        model: str = DEFAULT_OLLAMA_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        base_dir: Optional[str] = None,
    ):
        self.api_key = api_key or DEFAULT_OLLAMA_API_KEY
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.endpoint = endpoint or DEFAULT_OLLAMA_ENDPOINT
        self.base_dir = Path(base_dir or os.path.join(os.getcwd(), "outputs", "sprites"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.char_dir = self.base_dir / "characters"
        self.char_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.base_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.sheets_dir = self.base_dir / "sheets"
        self.sheets_dir.mkdir(parents=True, exist_ok=True)

    def _chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.5) -> str:
        """Invokes Gemma 4: 31B via Ollama Cloud API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("message", {}).get("content") or "").strip()
        except Exception as e:
            print(f"[SpritesPipeline] Gemma API call failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Character Genesis & Gemma Refinement
    # ─────────────────────────────────────────────────────────────────────────

    def refine_character_with_gemma(
        self,
        user_input: str,
        genre: str = "Action Platformer (Mega Man X / Hollow Knight)",
        art_style: str = "16-bit arcade pixel art style",
    ) -> Dict[str, Any]:
        """
        Uses Gemma 4: 31B to refine a raw user character idea into a professional 2D game asset concept
        optimized for sprite isolation against a solid chroma key background.
        """
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        system_prompt = (
            "You are a Lead 2D Game Character Designer and Technical Artist specializing in retro and modern arcade games.\n"
            "Your task is to take a rough user character idea and turn it into a production-ready 2D game sprite concept.\n"
            f"Game Genre: {genre} ({genre_info['perspective']})\n"
            f"Art Style: {art_style}\n\n"
            "CRITICAL SPRITE DESIGN REQUIREMENTS:\n"
            "1. Readability: High-contrast color blocking, exaggerated iconic silhouette recognizable at low resolutions.\n"
            "2. Orientation: Full body head-to-toe vertical 9:16 overview, strictly standing on a neutral ground plane.\n"
            "3. Background: The character MUST be placed on a flat, solid, uniform bright lime green chroma key background (#00FF00) "
            "with zero ground shadows, zero gradient, and zero scenery, ensuring flawless alpha cutout.\n\n"
            "Respond ONLY with valid JSON following this exact schema:\n"
            "{\n"
            '  "name": "Character Name",\n'
            '  "title": "Short title or archetype",\n'
            '  "genre": "Selected game genre",\n'
            '  "art_style": "Art style",\n'
            '  "appearance": "Detailed head-to-toe visual description including costume, colors, and footwear",\n'
            '  "signature_gear": "Weapon, prop, cannon, or signature item",\n'
            '  "palette": ["Color 1", "Color 2", "Color 3"],\n'
            '  "z_image_prompt": "Complete prompt for Z-Image Turbo specifying isolated 2D game sprite, 9:16 vertical full-body portrait, standing head-to-toe, complete outfit and footwear visible, front/side game view, on flat solid bright lime green chroma key studio background #00FF00, no floor shadows, clean crisp outline"\n'
            "}"
        )

        user_msg = f"User concept: {user_input}"
        try:
            raw_reply = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
        except Exception:
            # Deterministic fallback
            safe_name = "Game_Hero"
            if user_input and len(user_input.split()) <= 3:
                safe_name = user_input.title().replace(" ", "_")
            data = {
                "name": safe_name,
                "title": "Arcade Protagonist",
                "genre": genre,
                "art_style": art_style,
                "appearance": f"2D game character based on '{user_input}', full body standing head-to-toe, vibrant game asset",
                "signature_gear": "Standard arcade equipment",
                "palette": ["Vibrant Blue", "Bright Yellow", "Pure White"],
                "z_image_prompt": (
                    f"2D game sprite character sheet of {user_input}, 9:16 vertical full-body portrait, standing head-to-toe, "
                    f"complete outfit and footwear visible, front-side game view, {art_style}, "
                    f"on flat solid bright lime green chroma key studio background #00FF00, no floor shadows, clean crisp outline"
                ),
            }

        return data

    def synthesize_master_character_sprite(
        self,
        character_data: Dict[str, Any],
        seed: int = 42,
        output_dir: Optional[str] = None,
    ) -> Tuple[Image.Image, str]:
        """
        Synthesizes the master reference sprite image using Z-Image Turbo on Modal.
        """
        prompt = character_data.get("z_image_prompt") or f"2D game character sheet of {character_data.get('name', 'Hero')}"
        if "#00FF00" not in prompt and "green" not in prompt.lower():
            prompt += ", on flat solid bright lime green chroma key background #00FF00, no shadows, clean outline"

        save_folder = output_dir or str(self.char_dir)
        name_clean = "".join(c for c in character_data.get("name", "hero") if c.isalnum() or c in "_-")

        print(f"[SpritesPipeline] Generating master sprite for '{name_clean}' with seed {seed}...")
        pil_img, local_path = generate_z_image_turbo_remote(
            prompt=prompt,
            seed=int(seed),
            width=SPRITE_BASE_WIDTH,
            height=SPRITE_BASE_HEIGHT,
            steps=9,
            save_dir=save_folder,
            filename_prefix=f"master_sprite_{name_clean}",
        )
        return pil_img, local_path

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Move Choreography & Custom Animation Builder
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_default_moves_for_genre(genre: str) -> List[Dict[str, Any]]:
        """Returns the signature animation template for the specified genre."""
        preset = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        moves = []
        for idx, m in enumerate(preset["default_moves"]):
            moves.append({
                "id": f"move_{idx+1}",
                "name": m["name"],
                "type": m.get("type", "loop"),
                "description": m["description"],
                "refined_prompt": "",
                "wan_seed": 42 + idx * 1007,
                "video_path": None,
                "sheet_path": None,
                "json_path": None,
            })
        return moves

    def refine_custom_animation_with_gemma(
        self,
        character_data: Dict[str, Any],
        genre: str,
        anim_name: str,
        raw_description: str,
        anim_type: str = "action",
    ) -> Dict[str, Any]:
        """
        Uses Gemma 4: 31B to translate the user's custom animation idea into a micro-choreography
        prompt with strict camera lockdown and anchor rules.
        """
        char_name = character_data.get("name", "Character")
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        is_loop = anim_type.lower() == "loop"

        system_prompt = (
            "You are a Lead Gameplay Animator and Technical Director for 2D arcade video games.\n"
            f"Game Genre: {genre} ({genre_info['perspective']})\n"
            f"Character: {char_name} ({character_data.get('appearance', '')})\n"
            f"Animation Name: {anim_name}\n"
            f"Animation Type: {'Seamless Looping Cycle' if is_loop else 'Action / Attack with Return-to-Idle Recovery'}\n\n"
            "ANIMATION LAWS:\n"
            f"1. Camera: {genre_info['camera']}. The camera must NEVER pan, tilt, shake, or zoom.\n"
            "2. Background: Must remain fixed solid flat bright lime green chroma key background #00FF00.\n"
            + (
                "3. LOOPING RULE: This animation is an endless loop. The character's motion at t=5.0s must seamlessly match their starting pose at t=0.0s."
                if is_loop
                else "3. RECOVERY RULE: Frame 0 starts in the character's Idle pose. The middle delivers the strike/action. By t=5.0s, the character MUST cleanly settle back into the exact Idle resting pose."
            )
            + "\n\nRespond ONLY with valid JSON following this exact schema:\n"
            "{\n"
            '  "action_summary": "One sentence summary of the movement",\n'
            '  "choreography_prompt": "Micro-level limb and body action breakdown: start posture, acceleration, peak action, recovery",\n'
            '  "wan_prompt": "Complete prompt for Wan 2.2 specifying 2D sprite animation, fixed static camera, zero camera movement, on flat solid lime green background #00FF00",\n'
            '  "motion_intensity": 3\n'
            "}"
        )

        user_msg = f"User description of movement/behavior: {raw_description}"
        try:
            raw_reply = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
        except Exception:
            # Fallback prompt
            loop_desc = (
                "seamless infinite loop, continuous rhythmic cycle"
                if is_loop
                else "starts in resting idle pose, executes action dynamically, cleanly recovers and finishes in resting idle pose"
            )
            data = {
                "action_summary": f"{char_name} performs {anim_name}: {raw_description}",
                "choreography_prompt": f"{char_name} executes {anim_name}. {raw_description}. {loop_desc}.",
                "wan_prompt": (
                    f"2D game sprite animation of {char_name} performing {anim_name}: {raw_description}. "
                    f"{loop_desc}. Fixed static arcade tripod camera, zero camera panning, zero zoom, "
                    f"on flat solid bright lime green chroma key background #00FF00, sharp outline"
                ),
                "motion_intensity": 3,
            }

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Wan 2.2 Animation Synthesis
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_animation_clip(
        self,
        character_data: Dict[str, Any],
        move_data: Dict[str, Any],
        wan_seed: int = 42,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Synthesizes a 5-second video clip using Wan 2.2 on Modal.
        For Action moves: anchors Frame 0 to the master sprite and instructs clean return-to-idle.
        For Looping moves: instructs cyclic matching.
        """
        anim_name = move_data.get("name", "move")
        is_loop = move_data.get("type", "loop") == "loop"
        char_name = character_data.get("name", "Character")

        master_sprite_path = character_data.get("master_sprite_path")
        master_sprite_pil = character_data.get("master_sprite_pil")
        if master_sprite_pil is None:
            if not master_sprite_path or not os.path.exists(master_sprite_path):
                raise FileNotFoundError(
                    f"Master character sprite is missing at '{master_sprite_path}'. Please run Phase 1 first."
                )
            master_sprite_pil = Image.open(master_sprite_path).convert("RGB")
        elif not master_sprite_path or not os.path.exists(master_sprite_path):
            temp_path = self.char_dir / f"{char_name}_master.png"
            master_sprite_pil.save(str(temp_path))
            master_sprite_path = str(temp_path)
            character_data["master_sprite_path"] = master_sprite_path

        # Compile Wan prompt
        refined_wan = move_data.get("refined_prompt") or move_data.get("description", "")
        loop_clause = (
            "seamless cyclic loop, continuous motion, start pose matches end pose"
            if is_loop
            else "starts in resting idle pose, performs dynamic action, cleanly recovers back to resting idle pose at end"
        )
        full_wan_prompt = (
            f"2D game sprite animation of {char_name} executing {anim_name}: {refined_wan}. "
            f"{loop_clause}. Fixed tripod shot, static arcade 2D camera plane, zero camera pan, zero zoom, "
            f"on flat solid bright lime green chroma key background #00FF00, no floor shadows, clean outline"
        )

        if progress_cb:
            progress_cb(0.15, f"Synthesizing '{anim_name}' with Wan 2.2 (Seed {wan_seed})...")

        print(f"[SpritesPipeline] Synthesizing Wan 2.2 clip for '{anim_name}' (Seed: {wan_seed}, Loop: {is_loop})...")
        local_video_path, _ = run_remote_inference(
            prompt=full_wan_prompt,
            image=master_sprite_pil,
            negative_prompt=(
                "camera motion, camera pan, camera tilt, zoom, dynamic background, complex background, "
                "scenery, floor shadows, ground clutter, perspective shift, rotation, 3d camera orbit"
            ),
            current_seed=int(wan_seed),
            num_frames=81,  # 81 frames @ 16 fps ≈ 5.06s
            width=480,
            height=832,
            steps=8,
            guidance_scale=2.0,
            guidance_scale_2=1.5,
            flow_shift=3.0,
            quality=6,
            scheduler_name="UniPCMultistep",
            frame_multiplier=16,
            last_frame_image=master_sprite_pil if not is_loop else None,  # Return-to-idle anchor for actions!
        )

        move_data["video_path"] = local_video_path
        move_data["wan_seed"] = wan_seed

        if progress_cb:
            progress_cb(1.0, f"Animation '{anim_name}' synthesized successfully.")

        return local_video_path

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Sprite Cutter & Sheet Exporter
    # ─────────────────────────────────────────────────────────────────────────

    def cut_and_export_sprite_sheet(
        self,
        character_data: Dict[str, Any],
        move_data: Dict[str, Any],
        target_fps: int = 12,
        chroma_tolerance: float = 38.0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Slices the generated animation video into transparent RGBA frames and packs a sprite sheet.
        """
        video_path = move_data.get("video_path")
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video for move '{move_data.get('name')}' not found at: {video_path}")

        save_dir = output_dir or str(self.sheets_dir)
        char_name = character_data.get("name", "Hero")
        anim_name = move_data.get("name", "anim")
        genre = character_data.get("genre", "2D Game")
        is_loop = move_data.get("type", "loop") == "loop"

        result = process_video_to_sprite_sheet(
            video_path=video_path,
            character_name=char_name,
            animation_name=anim_name,
            genre=genre,
            target_fps=target_fps,
            is_loop=is_loop,
            tolerance=chroma_tolerance,
            output_dir=save_dir,
        )

        move_data["sheet_path"] = result["sheet_path"]
        move_data["json_path"] = result["json_path"]
        move_data["fps"] = target_fps
        move_data["total_frames"] = result["frame_count"]

        return result
