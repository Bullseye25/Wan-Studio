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
import shutil
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

# Verified active Ollama Cloud models (tested & benchmarked)
VERIFIED_OLLAMA_MODELS = [
    ("⚡ gemma4:31b — Fast Vision & Balanced (31B - Default)", "gemma4:31b"),
    ("🎬 gpt-oss:120b — Hollywood Screenwriter & Master Storyteller (120B)", "gpt-oss:120b"),
    ("⚔️ nemotron-3-ultra — Weapon-Locked Combat Choreographer & Physics", "nemotron-3-ultra"),
    ("🔍 nemotron-3-super — AI Director QA Critic & Keyframe Auditor", "nemotron-3-super"),
    ("🚀 nemotron-3-nano:30b — Lightning-Fast JSON & Timeline Compiler (30B)", "nemotron-3-nano:30b"),
    ("⚡ gpt-oss:20b — Agile Interactive Co-Pilot & Quick Tweaks (20B)", "gpt-oss:20b"),
    ("🤖 Auto-Routing (Smart Multi-Agent Workflow — Best Model per Task)", "auto"),
]

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
                "name": "Combat Idle Stance",
                "type": "loop",
                "description": "Ready combat guard holding equipped weapons firmly, subtle breathing and rhythmic weight shift",
            },
            {
                "name": "Walk Forward",
                "type": "loop",
                "description": "Confident forward combat stride closing distance on opponents, treadmill root motion",
            },
            {
                "name": "Sprint Dash",
                "type": "loop",
                "description": "Low-center-of-gravity high-speed sprint rush forward with weapons trailing aerodynamic",
            },
            {
                "name": "3-Hit Light Attack Combo",
                "type": "action",
                "description": (
                    "Fast-paced arcade 3-hit light flurry (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery): "
                    "(Step 1, 0.0s-0.5s) explosive darting slip-step forward -> "
                    "(Step 2, 0.5s-1.8s) rapid diagonal slash -> "
                    "(Step 3, 1.8s-3.0s) cross-stepping hip pivot into reverse stab -> "
                    "(Step 4, 3.0s-4.0s) lunging spin-cut finisher with magic sparks -> "
                    "(Step 5, 4.0s-5.0s) nimble backstep recovery to guard"
                ),
            },
            {
                "name": "3-Hit Heavy Attack Combo",
                "type": "action",
                "description": (
                    "Fast-paced arcade 3-hit heavy cleave combo (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery): "
                    "(Step 1, 0.0s-0.5s) heavy power step-in advance -> "
                    "(Step 2, 0.5s-1.8s) heavy horizontal cleave -> "
                    "(Step 3, 1.8s-3.0s) deep cross-step torso torque into crushing overhead slam -> "
                    "(Step 4, 3.0s-4.0s) full-body advancing 360-degree whirlwind cleave with flaming shockwaves -> "
                    "(Step 5, 4.0s-5.0s) planted boot recovery backstep into guard"
                ),
            },
            {
                "name": "Heavy Double-Damage Charge Hit",
                "type": "action",
                "description": (
                    "Classic Beat 'em Up 1.5s Hold-to-Charge Super Attack: (0.0s-1.5s) holds crouched charge gathering pulsating kinetic energy and flame aura -> "
                    "(1.5s-4.0s) erupts forward into an explosive double-damage breaker hit with shockwaves -> "
                    "(4.0s-5.0s) slides to a halt and recovers to idle guard"
                ),
            },
            {
                "name": "Sprint Dash Light Strike",
                "type": "action",
                "description": (
                    "Shinobi: Art of Vengeance high-velocity blitz: bursts from sprint into a low-friction sliding lunge thrust "
                    "cutting across the combat plane, springing back to ready stance"
                ),
            },
            {
                "name": "Stationary Dodge Counter",
                "type": "action",
                "description": "SoR4 neutral blitz: zero horizontal camera drift, evasive duck/spin parry on the spot and immediate counter-strike",
            },
            {
                "name": "Hit Reaction (Hurt Recoil)",
                "type": "action",
                "description": "Takes incoming enemy blow: torso snaps back in pain, grimace, rapid recoil and recovery back into guard stance",
            },
            {
                "name": "Knockdown & Ground Recovery (Drop Down & Get Up)",
                "type": "action",
                "description": "Fatal blow: falls backward -> drops flat onto floor (i-frames) -> tactical ground roll -> athletic spring to feet into idle guard",
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

    @staticmethod
    def parse_resolution(resolution_input: Any) -> Tuple[int, int]:
        """
        Parses resolution string or tuple into standard (width, height) divisible by 16.
        E.g. '832x832', '1080x1080', '1024x1024', '512x512', (832, 832).
        """
        if isinstance(resolution_input, (tuple, list)) and len(resolution_input) >= 2:
            w, h = int(resolution_input[0]), int(resolution_input[1])
        elif isinstance(resolution_input, str):
            match = re.search(r'(\d+)\s*[xX*×]\s*(\d+)', resolution_input)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
            else:
                w, h = 832, 832
        else:
            w, h = 832, 832

        # Round to nearest multiple of 16 (minimum 256)
        w = max(256, (w // 16) * 16)
        h = max(256, (h // 16) * 16)
        return w, h

    @staticmethod
    def pad_sprite_to_square_canvas(
        pil_img: Image.Image,
        target_size: Tuple[int, int] = (832, 832),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """
        Intelligently places and scales a character sprite onto a square canvas (e.g. 832x832, 1024x1024, 512x512)
        on pure seamless white studio background #FFFFFF with ample horizontal and vertical padding.
        Leaves generous breathing room horizontally and vertically so sweeping sword arcs, advancing lunges,
        and tall rising flaming dragon uppercuts never clip against the canvas boundaries.
        """
        target_w, target_h = target_size
        canvas = Image.new("RGB", (target_w, target_h), bg_color)

        img_w, img_h = pil_img.size
        # Allow character to fill up to 72% of height and 60% of width
        max_h = int(target_h * 0.72)
        max_w = int(target_w * 0.60)
        scale = min(max_w / max(1, img_w), max_h / max(1, img_h))
        new_w = max(16, int(img_w * scale))
        new_h = max(16, int(img_h * scale))

        # Align dimensions to multiple of 16 for clean video diffusion
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        if new_w <= 0:
            new_w = 16
        if new_h <= 0:
            new_h = 16

        scaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        paste_x = (target_w - new_w) // 2
        # Ground feet at ~88% of canvas height
        paste_y = int(target_h * 0.88) - new_h
        if paste_y < 16:
            paste_y = max(0, (target_h - new_h) // 2)

        if scaled.mode == "RGBA":
            canvas.paste(scaled, (paste_x, paste_y), scaled)
        else:
            canvas.paste(scaled, (paste_x, paste_y))
        return canvas

    def _resolve_model_tag(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        task_role: str = "default",
        model_override: Optional[str] = None,
    ) -> str:
        """
        Resolves the actual Ollama model tag to use.
        If model is 'auto', routes task to the benchmarked optimal model:
          - 'character': gemma4:31b (Visual prompt isolation)
          - 'moveset', 'choreography': nemotron-3-ultra (Physics & weapon-locked action)
          - 'quick_edit', 'timeline': nemotron-3-nano:30b (Fast 2.8s turnaround)
        If messages contain images, automatically routes to vision-capable 'gemma4:31b'.
        """
        if model_override:
            return model_override

        if messages:
            for m in messages:
                c = m.get("content")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            return "gemma4:31b"

        m = (self.model or DEFAULT_OLLAMA_MODEL).strip()
        if "auto" in m.lower():
            if task_role in ("moveset", "choreography", "combat"):
                return "nemotron-3-ultra"
            elif task_role in ("quick_edit", "custom_move", "timeline"):
                return "nemotron-3-nano:30b"
            return "gemma4:31b"
        return m

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.5,
        task_role: str = "default",
        model_override: Optional[str] = None,
    ) -> str:
        """Invokes Ollama Cloud API with task-aware model routing."""
        actual_model = self._resolve_model_tag(messages=messages, task_role=task_role, model_override=model_override)
        payload = {
            "model": actual_model,
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
                # Support OpenAI /v1/chat/completions format and native Ollama /api/chat format
                if "choices" in data and len(data["choices"]) > 0:
                    return (data["choices"][0].get("message", {}).get("content") or "").strip()
                return (data.get("message", {}).get("content") or "").strip()
        except Exception as e:
            print(f"[SpritesPipeline] Ollama API call failed ({actual_model}): {e}")
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
            "2. Weapon & Gear Clarity: Explicitly identify the character's held weapon, tool, or signature gear (e.g., katana sword, broadsword, arcane staff, bow, twin daggers, plasma blaster, shield, or bare martial-arts gauntlets) and ensure it is clearly held in a natural combat-ready idle posture.\n"
            "3. Orientation: Full body head-to-toe vertical 9:16 overview, strictly standing on a neutral ground plane.\n"
            "4. Background: The character MUST be placed on a pure flat, solid, uniform white studio background (#FFFFFF) "
            "with high-key studio lighting, zero ground shadows, zero gradient, and zero scenery, ensuring flawless alpha cutout and VFX preservation.\n\n"
            "Respond ONLY with valid JSON following this exact schema:\n"
            "{\n"
            '  "name": "Character Name",\n'
            '  "title": "Short title or archetype",\n'
            '  "genre": "Selected game genre",\n'
            '  "art_style": "Art style",\n'
            '  "appearance": "Detailed head-to-toe visual description including costume, colors, and footwear",\n'
            '  "signature_gear": "Exact weapon, blade, staff, bow, blaster, prop, or equipment held by the character",\n'
            '  "combat_style": "Brief description of how this character moves and fights using their signature_gear",\n'
            '  "palette": ["Color 1", "Color 2", "Color 3"],\n'
            '  "z_image_prompt": "Complete prompt for Z-Image Turbo specifying isolated 2D game sprite, 9:16 vertical full-body portrait, standing head-to-toe holding their signature weapon/gear in ready idle stance, complete outfit and footwear visible, front/side game view, on pure flat solid white studio background #FFFFFF, high-key studio lighting, no floor shadows, clean crisp outline"\n'
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
                task_role="character",
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
        except Exception:
            # Smart deterministic fallback that extracts weapon/gear from user_input
            safe_name = "Game_Hero"
            if user_input and len(user_input.split()) <= 3:
                safe_name = user_input.title().replace(" ", "_")
            archetype_info = self.detect_character_combat_archetype(
                {"appearance": user_input, "signature_gear": user_input, "name": safe_name},
                genre=genre,
            )
            detected_gear = archetype_info["weapon_label"]
            data = {
                "name": safe_name,
                "title": archetype_info["role_title"],
                "genre": genre,
                "art_style": art_style,
                "appearance": f"2D game character based on '{user_input}', full body standing head-to-toe, vibrant game asset",
                "signature_gear": detected_gear,
                "combat_style": archetype_info["combat_style_summary"],
                "palette": ["Vibrant Blue", "Bright Yellow", "Pure White"],
                "z_image_prompt": (
                    f"2D game sprite character sheet of {user_input}, holding {detected_gear} in ready stance, "
                    f"9:16 vertical full-body portrait, standing head-to-toe, "
                    f"complete outfit and footwear visible, front-side game view, {art_style}, "
                    f"on pure flat solid white studio background #FFFFFF, high-key studio lighting, no floor shadows, clean crisp outline"
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
        if "#FFFFFF" not in prompt and "white" not in prompt.lower():
            prompt += ", on pure flat solid white studio background #FFFFFF, high-key studio lighting, no shadows, clean outline"

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
    # Phase 2: Smart Character-Aware Move Choreography & Custom Creator
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_weapons_and_combat_tiers(character_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the character prompt, signature gear, and appearance into discrete combat damage tiers:
        Primary Heavy Weapon (swords, broadswords, claymores, warhammers) vs Secondary Light Weapon (daggers, kunai, bucklers)
        with emotional combat posture and magic VFX detection.
        """
        gear_raw = str(character_data.get("signature_gear") or "").strip()
        appearance_raw = str(character_data.get("appearance") or "").strip()
        title_raw = str(character_data.get("title") or "").strip()
        concept_raw = str(character_data.get("user_concept") or "").strip()
        prompt_raw = str(character_data.get("z_image_prompt") or "").strip()

        combined = f"{gear_raw} {concept_raw} {appearance_raw} {prompt_raw} {title_raw}".lower()

        # Heavy / Primary weapon keywords
        heavy_patterns = [
            (r"\bdual[- ]edge\s+sword\b", "dual edge sword"),
            (r"\bbroadsword\b", "broadsword"),
            (r"\bgreatsword\b", "greatsword"),
            (r"\bclaymore\b", "claymore"),
            (r"\bkatana\b", "katana"),
            (r"\blongsword\b", "longsword"),
            (r"\bwarhammer\b", "warhammer"),
            (r"\bbattleaxe\b", "battleaxe"),
            (r"\bgreataxe\b", "greataxe"),
            (r"\bhalberd\b", "halberd"),
            (r"\bpolearm\b", "polearm"),
            (r"\bsword\b", "sword"),
            (r"\bblade\b", "blade"),
            (r"\baxe\b", "axe"),
            (r"\bhammer\b", "hammer"),
        ]
        # Light / Secondary weapon keywords
        light_patterns = [
            (r"\bsmall\s+magic\s+red\s+dagger\b", "small magic red dagger"),
            (r"\bmagic\s+red\s+dagger\b", "magic red dagger"),
            (r"\bred\s+dagger\b", "magic red dagger"),
            (r"\bmagic\s+dagger\b", "magic dagger"),
            (r"\bdagger\b", "dagger"),
            (r"\bkunai\b", "kunai"),
            (r"\bshuriken\b", "shurikens"),
            (r"\bbuckler\b", "buckler shield"),
            (r"\bknife\b", "knife"),
            (r"\bclaw\b", "claws"),
            (r"\bstiletto\b", "stiletto"),
        ]

        primary_heavy = None
        for pat, label in heavy_patterns:
            if re.search(pat, combined):
                primary_heavy = label
                break

        secondary_light = None
        for pat, label in light_patterns:
            if re.search(pat, combined):
                secondary_light = label
                break

        # Check dual-wield
        is_dual_wield = bool(
            (primary_heavy and secondary_light) or
            re.search(r"\b(dual|twin|two\s+weapons|sword\s+and\s+dagger|pair\s+of)\b", combined)
        )

        # Magic elements
        has_magic = bool(re.search(r"\b(magic|red\s+dagger|flame|plasma|energy|spark|glow|arcane|rune)\b", combined))

        # Emotional & armor tone
        emotional_tone = "disciplined"
        if re.search(r"\b(heartbroken|sad|grief|melanchol|weep)\b", combined):
            emotional_tone = "heartbroken and melancholic (desperation swings, heavy emotional recovery frames)"
        elif re.search(r"\b(berserk|rage|furious|wild)\b", combined):
            emotional_tone = "furious berserker (reckless forward momentum)"
        elif re.search(r"\b(broken\s+armor|shattered|damaged)\b", combined):
            emotional_tone = "battle-worn with broken armor (vulnerable stance, protective guard)"

        p_label = primary_heavy or "dual edge sword"
        s_label = secondary_light or ("small magic red dagger" if (has_magic and is_dual_wield) else None)

        return {
            "primary_heavy_weapon": p_label,
            "secondary_light_weapon": s_label,
            "is_dual_wield": is_dual_wield,
            "has_magic_vfx": has_magic,
            "emotional_tone": emotional_tone,
            "heavy_role": "High poise damage, wide cleaving arcs, armor break, knockdowns, charged double-damage hits",
            "light_role": "Rapid combo filler, low startup latency, surgical stabs, glowing spark trails" if s_label else "Unarmed strikes/kicks",
            "weapon_descriptor": (
                f"gleaming polished silver-steel {p_label} with reflective chrome highlights in dominant hand"
                + (f" and {s_label} with glowing crimson magic energy in off-hand" if s_label else "")
            ),
        }

    @staticmethod
    def detect_character_combat_archetype(character_data: Dict[str, Any], genre: str = "") -> Dict[str, Any]:
        """
        Analyzes the character's weapon, signature gear, appearance, and title to determine their
        exact combat archetype, idle weapon posture, and forbidden mismatched animations
        (e.g. preventing boxing animations when the character wields a sword).
        """
        gear_raw = str(character_data.get("signature_gear") or "").strip()
        appearance_raw = str(character_data.get("appearance") or "").strip()
        title_raw = str(character_data.get("title") or "").strip()
        name_raw = str(character_data.get("name") or "").strip()
        prompt_raw = str(character_data.get("z_image_prompt") or "").strip()
        concept_raw = str(character_data.get("user_concept") or "").strip()

        combined = f"{gear_raw} {title_raw} {appearance_raw} {name_raw} {prompt_raw} {concept_raw}".lower()

        # Multi-weapon parsing
        weapon_tiers = SpritesPipeline.parse_weapons_and_combat_tiers(character_data)
        p_weapon = weapon_tiers["primary_heavy_weapon"]
        s_weapon = weapon_tiers["secondary_light_weapon"]

        def _has_word(words: List[str]) -> Optional[str]:
            for w in words:
                if re.search(r"\b" + re.escape(w) + r"s?\b", combined):
                    return w
            return None

        # 0. Dual Wield (Heavy + Light / Asymmetric Beat 'em Up)
        if weapon_tiers["is_dual_wield"] and s_weapon:
            weapon_label = f"{p_weapon} & {s_weapon}"
            emotional_clause = f", reflecting {weapon_tiers['emotional_tone']}" if weapon_tiers['emotional_tone'] != "disciplined" else ""
            return {
                "archetype": "dual_wield",
                "role_title": "Dual-Weapon Specialist",
                "weapon_label": weapon_label,
                "primary_heavy_weapon": p_weapon,
                "secondary_light_weapon": s_weapon,
                "damage_tiers": weapon_tiers,
                "idle_posture": f"poised dual-wield combat guard gripping gleaming silver-steel {p_weapon} in dominant hand and {s_weapon} in reverse off-hand guard{emotional_clause}",
                "combat_style_summary": (
                    f"Streets of Rage 4 asymmetric dual-weapon combat: heavy {p_weapon} poise-breaking cleaves, "
                    f"360-degree spin knockdowns, and double-damage charged hits synthesized with rapid {s_weapon} slashes, "
                    "crimson spark trails, and sliding blitz strikes"
                ),
                "forbidden_actions": (
                    f"NEVER use unarmed boxing jabs, bare-fisted uppercuts, or arm-cannon blasts. The character MUST "
                    f"wield both their {p_weapon} (heavy attacks) and {s_weapon} (light attacks/VFX) actively."
                ),
            }

        # 1. Sword / Blade / Samurai / Knight
        blade_match = _has_word([
            "katana", "greatsword", "broadsword", "claymore", "rapier", "scimitar",
            "saber", "sabre", "cutlass", "scythe", "excalibur", "longsword", "shortsword",
            "sword", "blade", "samurai", "swordsman", "swordswoman", "fencer", "ronin", "knight", "paladin",
        ])
        if blade_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else f"{blade_match} blade"
            return {
                "archetype": "blade",
                "role_title": "Sword Master",
                "weapon_label": weapon_label,
                "idle_posture": f"poised sword-guard stance gripping {weapon_label} at the ready",
                "combat_style_summary": f"Precision blade combat using {weapon_label} slashes, sword-drawn dashes, parries, and sweeping blade arcs",
                "forbidden_actions": (
                    "NEVER use unarmed boxing jabs, bare-fisted uppercuts, boxer bounce stances, arm-cannon blasts, "
                    f"or soda-can drinking. The character MUST keep their {weapon_label} firmly in hand for all stances and attacks."
                ),
            }

        # 2. Heavy Melee (Axe, Hammer, Mace, Club)
        heavy_match = _has_word([
            "battleaxe", "warhammer", "greataxe", "sledgehammer", "maul", "mace", "flail",
            "club", "cleaver", "axe", "hammer", "berserker", "barbarian", "juggernaut",
        ])
        if heavy_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else f"heavy {heavy_match}"
            return {
                "archetype": "heavy_melee",
                "role_title": "Heavy Vanguard",
                "weapon_label": weapon_label,
                "idle_posture": f"heavy grounded combat stance hefting {weapon_label} with two hands",
                "combat_style_summary": f"High-impact heavy melee using {weapon_label} cleaves, crushing overhead slams, and wide shockwave swings",
                "forbidden_actions": (
                    "NEVER use light boxer jabs, unarmed martial arts kicks, or firearm/cannon blasts. "
                    f"All movements must convey the weight and momentum of their {weapon_label}."
                ),
            }

        # 3. Polearm / Spear / Lance
        polearm_match = _has_word([
            "halberd", "naginata", "glaive", "trident", "lance", "spear", "pike", "javelin", "bo staff", "quarterstaff", "dragoon", "lancer",
        ])
        if polearm_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else polearm_match
            return {
                "archetype": "polearm",
                "role_title": "Polearm Lancer",
                "weapon_label": weapon_label,
                "idle_posture": f"balanced polearm guard holding {weapon_label} diagonally across the body",
                "combat_style_summary": f"Long-reach piercing thrusts, sweeping {weapon_label} spins, and vaulting impale strikes",
                "forbidden_actions": (
                    f"NEVER use short-range boxing punches or unarmed brawling; always leverage the reach of {weapon_label}."
                ),
            }

        # 4. Agile Melee / Daggers / Ninja / Claws
        agile_match = _has_word([
            "dagger", "kunai", "shuriken", "katar", "claw", "nunchaku", "sai", "whip",
            "twin blades", "dual blades", "knife", "ninja", "shinobi", "assassin", "rogue", "thief",
        ])
        if agile_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else agile_match
            return {
                "archetype": "agile_melee",
                "role_title": "Shadow Shinobi",
                "weapon_label": weapon_label,
                "idle_posture": f"low agile shinobi crouch holding {weapon_label} in reverse grip",
                "combat_style_summary": f"Lightning-fast acrobatic strikes, shadow dashes, and rapid {weapon_label} flurries",
                "forbidden_actions": (
                    f"NEVER use slow heavy clumsy swings or upright boxing stances; keep movements sleek, acrobatic, and centered on {weapon_label}."
                ),
            }

        # 5. Bow / Crossbow / Archer
        bow_match = _has_word([
            "longbow", "crossbow", "shortbow", "bow", "arrow", "quiver", "archer", "ranger", "huntress", "marksman",
        ])
        if bow_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else bow_match
            return {
                "archetype": "ranged_bow",
                "role_title": "Master Archer",
                "weapon_label": weapon_label,
                "idle_posture": f"poised archer stance holding {weapon_label} with an arrow nocked at the ready",
                "combat_style_summary": f"Ranged precision archery drawing {weapon_label}, rapid arrow volleys, and agile backstep dodges",
                "forbidden_actions": (
                    f"NEVER use close-range boxing punches or sword slashes; all attacks must draw and fire {weapon_label}."
                ),
            }

        # 6. Magic / Staff / Wand / Mage
        magic_match = _has_word([
            "staff", "wand", "scepter", "orb", "grimoire", "spellbook", "tome", "crystal", "rune",
            "wizard", "mage", "sorcerer", "sorceress", "witch", "warlock", "necromancer", "archmage", "cleric", "druid", "shaman",
        ])
        if magic_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else f"arcane {magic_match}"
            return {
                "archetype": "magic",
                "role_title": "Arcane Spellcaster",
                "weapon_label": weapon_label,
                "idle_posture": f"mystical spellcaster stance holding {weapon_label} with subtle arcane glow",
                "combat_style_summary": f"Arcane spellcasting channeling energy through {weapon_label}, summoning sigils, and firing elemental blasts",
                "forbidden_actions": (
                    f"NEVER use physical boxing punches, brawling kicks, or mechanical gunfire; all actions channel magic through {weapon_label}."
                ),
            }

        # 7. Firearm / Blaster / Arm Cannon / Cyborg
        gun_match = _has_word([
            "buster", "arm cannon", "cannon", "blaster", "rifle", "pistol", "revolver", "shotgun",
            "laser", "plasma gun", "raygun", "launcher", "gun", "cyborg", "gunner", "commando", "trooper",
        ])
        if gun_match:
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else gun_match
            return {
                "archetype": "ranged_firearm",
                "role_title": "Tactical Gunner",
                "weapon_label": weapon_label,
                "idle_posture": f"tactical combat stance with {weapon_label} aimed forward",
                "combat_style_summary": f"Run-and-gun ranged combat firing bursts and charged blasts from {weapon_label} with crisp recoil recovery",
                "forbidden_actions": (
                    f"NEVER use medieval sword swings or bare-knuckle boxing jabs; all attacks fire or aim {weapon_label}."
                ),
            }

        # 8. Spaceship / Vehicle / Mech
        ship_match = _has_word([
            "spaceship", "starfighter", "spacecraft", "fighter jet", "interceptor", "ship", "hovercraft", "tank", "drone",
        ])
        if ship_match or "shmup" in genre.lower() or "space invaders" in genre.lower():
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else "twin plasma cannons & ion thrusters"
            return {
                "archetype": "vehicle_ship",
                "role_title": "Arcade Starfighter",
                "weapon_label": weapon_label,
                "idle_posture": f"steady aerodynamic hover with {weapon_label} and thruster glow",
                "combat_style_summary": f"Precision aerial banking, thruster boost, and weapon volleys from {weapon_label}",
                "forbidden_actions": "NEVER use humanoid limb animations; strictly mechanical banking, thrusting, and weapon bay firing.",
            }

        # 9. Action Runner / Beverage / Skateboard Prop
        runner_match = _has_word([
            "soda", "pepsi", "can", "bottle", "skateboard", "hoverboard", "rollerblades", "surfboard", "parkour",
        ])
        if runner_match or ("runner" in genre.lower() and not gear_raw):
            weapon_label = gear_raw if (gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none")) else (runner_match or "athletic runner gear")
            return {
                "archetype": "runner_acrobat",
                "role_title": "Agile Runner",
                "weapon_label": weapon_label,
                "idle_posture": f"high-energy athletic runner stance with {weapon_label}",
                "combat_style_summary": f"High-speed parkour sprinting, hurdle slides, aerial vaults, and celebratory {weapon_label} actions",
                "forbidden_actions": "Keep movements focused on high-speed agility, dodging, vaulting, and expressive prop interaction.",
            }

        # 10. Default / Martial Artist / Custom Gear
        if gear_raw and gear_raw.lower() not in ("standard arcade equipment", "standard gear", "none", "fists", "bare fists"):
            return {
                "archetype": "custom_gear",
                "role_title": title_raw or "Arcade Specialist",
                "weapon_label": gear_raw,
                "idle_posture": f"ready combat stance holding {gear_raw}",
                "combat_style_summary": f"Dynamic combat and movement actively utilizing {gear_raw}",
                "forbidden_actions": (
                    f"NEVER ignore the character's equipped {gear_raw} or substitute unrelated generic boxing/arm-cannon moves."
                ),
            }

        return {
            "archetype": "unarmed_martial",
            "role_title": title_raw or "Martial Arts Brawler",
            "weapon_label": gear_raw or "martial arts gauntlets",
            "idle_posture": "balanced martial arts guard stance with fists raised and rhythmic footwork",
            "combat_style_summary": "Fluid hand-to-hand martial arts strikes, agile footwork, and dynamic combo kicks",
            "forbidden_actions": "Do not conjure random weapons out of thin air; keep strikes grounded in martial arts kinematics.",
        }

    @classmethod
    def reason_combo_footwork_choreography(
        cls,
        character_data: Dict[str, Any],
        move_name: str,
        move_description: str = "",
        genre: str = "",
    ) -> Dict[str, Any]:
        """
        AI Combat Kinematics & Footwork Decision Engine modeled after 'Streets of Rage 4'
        and 'Shinobi: Art of Vengeance' (Lizardcube 2D Action Canon).
        Analyzes the character's weapon weight, combat archetype, and move intent to decide
        exact advancing step-ins, cross-pivots, lunges, and acrobatic recovery steps so the
        character dynamically moves and hits rather than standing still like a turret.
        """
        arch_info = cls.detect_character_combat_archetype(character_data, genre=genre)
        weapon_tiers = cls.parse_weapons_and_combat_tiers(character_data)
        p_weapon = weapon_tiers["primary_heavy_weapon"]
        s_weapon = weapon_tiers["secondary_light_weapon"] or arch_info["weapon_label"]
        vfx_tag = "crimson magic spark trails" if weapon_tiers["has_magic_vfx"] else "sharp kinetic blade streaks"

        m_blob = f"{move_name} {move_description}".lower()
        is_stationary_dodge = "stationary" in m_blob or "neutral blitz" in m_blob
        is_combo_or_strike = (
            not is_stationary_dodge
            and any(
                k in m_blob
                for k in (
                    "combo", "3-hit", "2-hit", "4-hit", "flurry", "slash", "cleave",
                    "strike", "smash", "impale", "whirlwind", "spin", "lotus", "dragon",
                )
            )
        )

        is_heavy = any(k in move_name.lower() for k in ("heavy", "cleave", "slam", "power", "crush")) or (
            "heavy" in m_blob and not any(k in move_name.lower() for k in ("light", "quick", "iaido", "dash"))
        )

        # 1. Light / Agile / Shinobi Combo Footwork (Shinobi: Art of Vengeance Canon)
        if is_combo_or_strike and not is_heavy and any(k in m_blob for k in ("light", "dagger", "kunai", "shinobi", "flurry", "twin", "iaido", "quick", "sprint dash")):
            active_w = s_weapon if ("light" in m_blob or "dagger" in m_blob) else arch_info["weapon_label"]
            steps = [
                f"Step 1 (0.0s-0.5s — Shinobi Slip-Step Advance): Character bursts from guard with an instantaneous darting front-foot slip-step forward, closing distance into point-blank range with zero delay",
                f"Step 2 (0.5s-1.8s — Rapid High-Speed Flurry): Instantly snaps {active_w} in a blistering diagonal slash ({vfx_tag})",
                f"Step 3 (1.8s-3.0s — Rotational Cross-Pivot Strike): Rear foot steps aggressively across in a tight 180-degree hip-pivot, driving torso forward into a reverse-grip cross-slice with {active_w}",
                f"Step 4 (3.0s-4.0s — High-Impact Acrobatic Finisher): Launches off lead foot into a 'Shinobi: Art of Vengeance' lunging spin-slice or rising somersault blitz cut with {active_w} at peak extension, emitting bright energy bursts",
                f"Step 5 (4.0s-5.0s — Nimble Shadow-Hop Recovery): Executes a crisp athletic backstep hop, returning feet and weapons to exact starting coordinates and {arch_info['idle_posture']} at t=5.0s",
            ]
            return {
                "is_advancing_combo": True,
                "combat_reference": "Shinobi: Art of Vengeance & Streets of Rage 4 (Fast-Paced Advancing Footwork)",
                "footwork_style": "Agile darting slip-step -> rapid flurry -> rotational cross-pivot -> acrobatic lunge finisher -> nimble backstep recovery",
                "active_weapon": active_w,
                "ai_decision_reasoning": (
                    f"Light/agile attacks with {active_w} require high-velocity forward momentum and evasive agility. "
                    "Following fast-paced arcade pacing (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery), the AI drives each strike with a "
                    "'Shinobi: Art of Vengeance' advancing slip-step, cross-pivot, and lunging finisher before hopping back to the origin anchor."
                ),
                "step_sequence": steps,
            }

        # 2. Heavy / Power / Sword / Brawler Combo Footwork (Streets of Rage 4 Canon)
        if is_combo_or_strike:
            active_w = p_weapon if ("heavy" in m_blob or "sword" in m_blob or "cleave" in m_blob) else arch_info["weapon_label"]
            w_lower = (active_w or "").lower()
            if any(k in w_lower for k in ("sword", "blade", "katana", "cleaver")):
                climax_step = f"Step 4 (3.0s-4.0s — Flaming Blade Whirlwind Cleave): Commits full body torque into an advancing 360-degree whirlwind cleave with {active_w}, unleashing a blazing flaming blade wave and knockdown shockwaves"
            elif any(k in w_lower for k in ("fist", "glove", "knuckle", "brawler", "bare")):
                climax_step = f"Step 4 (3.0s-4.0s — Super Power Flaming Uppercut): Launches off lead boot into a classic Streets of Rage 4 'Grand Upper' rising flaming dragon uppercut with fire erupting skyward"
            else:
                climax_step = f"Step 4 (3.0s-4.0s — Devastating Breaker Finisher): Commits full body weight into an advancing 360-degree whirlwind pivot slam with {active_w} that detonates ground-cracking shockwaves"

            steps = [
                f"Step 1 (0.0s-0.5s — Advancing Power Step-In): Stomps lead boot forward in a heavy 'Streets of Rage 4' power step-in, instantly closing striking distance with zero sluggishness",
                f"Step 2 (0.5s-1.8s — Crushing Opening Cleave): Drives shoulder and hip weight into a high-speed opening cleave with {active_w} ({vfx_tag})",
                f"Step 3 (1.8s-3.0s — Torque Cross-Step Hammer): Rear leg steps through across centerline in an aggressive advancing cross-step, torquing torso to whip {active_w} into a brutal second strike",
                climax_step,
                f"Step 5 (4.0s-5.0s — Planted Boot Recovery Step): Draws lead boot back in a disciplined athletic recovery step, re-centering weight at exact origin coordinates and {arch_info['idle_posture']} at t=5.0s",
            ]
            return {
                "is_advancing_combo": True,
                "combat_reference": "Streets of Rage 4 & Shinobi: Art of Vengeance (Fast-Paced Heavy Step-and-Hit)",
                "footwork_style": "Grounded power step-in -> opening cleave -> torque cross-step -> 360 advancing pivot lunge -> planted recovery backstep",
                "active_weapon": active_w,
                "ai_decision_reasoning": (
                    f"Heavy melee combos with {active_w} demand visible weight transfer and forward momentum. "
                    "Following Streets of Rage 4 & Shinobi fast-paced arcade pacing (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery), every swing is propelled "
                    "by a grounded power step-in, cross-step hip torque, and 360 pivot lunge so the character moves and hits dynamically with zero delay."
                ),
                "step_sequence": steps,
            }

        return {
            "is_advancing_combo": False,
            "combat_reference": "Streets of Rage 4 & Shinobi: Art of Vengeance",
            "footwork_style": "Anchored stance kinematics",
            "active_weapon": arch_info["weapon_label"],
            "ai_decision_reasoning": "Non-combo state or stationary neutral maneuver.",
            "step_sequence": [],
        }

    @classmethod
    def build_pre_animation_wan_prompt(
        cls,
        character_data: Dict[str, Any],
        genre: str,
        move_name: str,
        move_type: str,
        move_description: str,
    ) -> str:
        """
        Compiles a structured, temporally-anchored Wan 2.2 pre-animation instruction prompt
        that locks the character's identity, equipped weapon/gear, advancing combo footwork,
        start/end posture, and camera.
        """
        char_name = character_data.get("name") or "Character"
        appearance = character_data.get("appearance") or "2D game character"
        art_style = character_data.get("art_style") or "16-bit arcade pixel art style"
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        arch = cls.detect_character_combat_archetype(character_data, genre=genre)
        weapon_label = arch["weapon_label"]
        idle_posture = arch["idle_posture"]
        is_loop = str(move_type).lower() == "loop"

        weapon_tiers = cls.parse_weapons_and_combat_tiers(character_data)
        p_weapon = weapon_tiers.get("primary_heavy_weapon") or weapon_label
        s_weapon = weapon_tiers.get("secondary_light_weapon")

        if s_weapon and s_weapon.lower() not in p_weapon.lower():
            weapon_lock_text = (
                f"strictly gripping {p_weapon} firmly in primary hand and {s_weapon} in off-hand across all frames. "
                f"Both weapons are permanently locked in character's hands and MUST NEVER disappear, holster, or morph into bare fists"
            )
        else:
            weapon_lock_text = (
                f"strictly gripping {p_weapon} firmly in hand across all frames. "
                f"The weapon is permanently locked in character's grip and MUST NEVER disappear, holster, or morph into bare hands"
            )

        invariant_clause = (
            f"[Character & Weapon Invariants]: 2D game sprite of {char_name} ({appearance}, {art_style}), "
            f"strictly preserving exact costume proportions, silhouette, and {weapon_lock_text}."
        )
        m_lower = move_name.lower()
        footwork_info = cls.reason_combo_footwork_choreography(
            character_data=character_data,
            move_name=move_name,
            move_description=move_description,
            genre=genre,
        )

        if "knockdown" in m_lower or "drop down" in m_lower:
            timeline_clause = (
                f"[Pre-Animation Timeline]: (4-Phase Staged Knockdown & Recovery) "
                f"Phase 1 (0.0s-1.2s): Takes heavy fatal blow, torso snaps back in pain, airborne recoil falling backward; "
                f"Phase 2 (1.2s-2.5s): Drops flat onto ground plane on back with invulnerable down frames and subtle impact slide; "
                f"Phase 3 (2.5s-3.8s): Tactical ground roll to the right, arms push off floor in an athletic spring back to feet; "
                f"Phase 4 (3.8s-5.0s): Plants feet firmly, brings {weapon_label} back into guard, and settles into exact starting {idle_posture}."
            )
        elif is_loop and any(k in m_lower for k in ("walk", "sprint", "run", "dash")):
            timeline_clause = (
                f"[Pre-Animation Timeline]: (Locomotion Stride Loop) "
                f"Phase 1 (0.0s-1.5s): Character begins already in full dynamic stride contact pose (treadmill root, zero camera drift); "
                f"Phase 2 (1.5s-3.5s): Continuous locomotive leg cycle and weapon pumping ({move_description}); "
                f"Phase 3 (3.5s-5.0s): Seamlessly completes the step cycle so final frame at t=5.0s matches the initial contact stride pose perfectly for zero frame skipping."
            )
        elif "charge" in m_lower or "double damage" in m_lower:
            w_lower = weapon_label.lower()
            if any(k in w_lower for k in ("sword", "blade", "katana", "cleaver", "axe", "saber")):
                climax_action = f"unleashing a colossal flaming blade cleave with fiery arc shockwave trails"
            elif any(k in w_lower for k in ("fist", "glove", "knuckle", "brawler", "bare")):
                climax_action = f"unleashing a super power rising flaming dragon uppercut with blazing vertical shockwaves"
            elif s_weapon:
                climax_action = f"unleashing an explosive dual-weapon cross-breaker slash combining {p_weapon} and {s_weapon} with blazing shockwaves"
            else:
                climax_action = f"unleashing a devastating double-damage breaker impact with radial ground shockwaves"

            timeline_clause = (
                f"[Pre-Animation Timeline]: (Classic Beat 'em Up 1.5s Hold-to-Charge Super Attack) "
                f"Phase 1 (0.0s-1.5s — The 1.5s Button-Hold Charge): Character plants boots in a deep grounded crouch, holding {weapon_label} cocked back in ready stance; glowing kinetic energy, crackling sparks, and heat aura gather and pulsate along the weapon, surging in visual power for 1.5 full seconds; "
                f"Phase 2 (1.5s-4.0s — Explosive Super Hit Release): At t=1.5s, the charge erupts! Character explodes forward in a blinding dash, {climax_action}; "
                f"Phase 3 (4.0s-5.0s — Athletic Recovery): Decisively slides to a halt, absorbs recoil, and steps back smoothly into exact starting {idle_posture}."
            )
        elif "dodge" in m_lower:
            timeline_clause = (
                f"[Pre-Animation Timeline]: (Stationary Neutral Blitz) "
                f"Phase 1 (0.0s-1.2s): Zero horizontal camera movement, quick evasive pivot/duck on the spot to evade incoming attack; "
                f"Phase 2 (1.2s-3.2s): Instant counter-strike with {weapon_label} on the exact same coordinate; "
                f"Phase 3 (3.2s-5.0s): Swift recovery resetting into starting {idle_posture}."
            )
        elif not is_loop and footwork_info["is_advancing_combo"]:
            steps_joined = "; ".join(footwork_info["step_sequence"])
            timeline_clause = (
                f"[Pre-Animation Timeline]: (Fast-Paced Kinetic Advancing Combo — {footwork_info['combat_reference']}) "
                f"FAST-PACED ARCADE CADENCE (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery): "
                f"CRITICAL MOVE-AND-HIT RULE: NEVER stand still with glued feet while swinging; every strike in {move_name} MUST dynamically propel the body forward ({move_description}) with zero sluggishness or slow-motion. "
                f"{steps_joined}."
            )
        elif is_loop:
            timeline_clause = (
                f"[Pre-Animation Timeline]: "
                f"Phase 1 (0.0s-1.5s): starts in {idle_posture} and smoothly initiates {move_name}; "
                f"Phase 2 (1.5s-3.5s): {move_description} while keeping {weapon_label} consistent; "
                f"Phase 3 (3.5s-5.0s): completes the rhythmic cycle so the final frame seamlessly matches the t=0.0s starting pose for an endless loop."
            )
        else:
            timeline_clause = (
                f"[Pre-Animation Timeline]: "
                f"Phase 1 (0.0s-1.2s): starts anchored in {idle_posture} holding {weapon_label}; "
                f"Phase 2 (1.2s-3.6s): dynamically steps in and executes {move_name} ({move_description}); "
                f"Phase 3 (3.6s-5.0s): follows through and cleanly recovers back into the exact starting {idle_posture} by t=5.0s."
            )
        camera_clause = (
            f"[Camera & Studio Background]: {genre_info['camera']}, {genre_info['perspective']}, "
            "on pure flat seamless solid white studio background #FFFFFF, high-key studio lighting, zero floor shadows, crisp 2D sprite outline."
        )
        return f"{invariant_clause} {timeline_clause} {camera_clause}"

    @staticmethod
    def get_default_moves_for_genre(genre: str) -> List[Dict[str, Any]]:
        """Returns the base signature animation template for the specified genre."""
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

    @classmethod
    def get_smart_contextual_moves(cls, character_data: Dict[str, Any], genre: str) -> List[Dict[str, Any]]:
        """
        Generates a smart, weapon-and-character-aware move-set tailored to both the character's
        equipped weapon/archetype and the selected game genre. Incorporates Streets of Rage 4
        and Shinobi: Art of Vengeance advancing step-and-hit combo footwork.
        """
        preset = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        target_count = len(preset["default_moves"])
        arch_info = cls.detect_character_combat_archetype(character_data, genre=genre)
        arch = arch_info["archetype"]
        weapon = arch_info["weapon_label"]
        is_top_down = "top-down" in genre.lower() or "rpg" in genre.lower()
        is_runner = "runner" in genre.lower()
        is_beat_em_up = "beat 'em up" in genre.lower()

        # 0. Streets of Rage 4 & Shinobi: Art of Vengeance Dual-Wield Suite
        if arch == "dual_wield":
            weapon_tiers = cls.parse_weapons_and_combat_tiers(character_data)
            p_w = weapon_tiers["primary_heavy_weapon"]
            s_w = weapon_tiers["secondary_light_weapon"] or "secondary weapon"
            vfx_clause = " with glowing crimson magic sparks" if weapon_tiers["has_magic_vfx"] else " with gleaming blade streaks"

            raw_moves = [
                ("Combat Idle Stance", "loop", f"Alert combat guard gripping {p_w} in dominant hand and {s_w} in ready off-hand stance, subtle breathing and weight shift"),
                ("Walk Forward", "loop", f"Confident forward combat stride holding {p_w} and {s_w} poised, treadmill root motion centered in frame"),
                ("Sprint Dash", "loop", f"High-speed aerodynamic rush forward leaning low with {p_w} trailing and {s_w} leading"),
                (
                    "3-Hit Light Attack Combo",
                    "action",
                    f"Shinobi: Art of Vengeance & SoR4 fast-paced arcade 3-strike flurry with {s_w} (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery): (Step 1, 0.0s-0.5s) darting front-foot slip-step advance -> "
                    f"(Step 2, 0.5s-1.8s) rapid diagonal slash -> (Step 3, 1.8s-3.0s) cross-stepping hip pivot into reverse stab -> "
                    f"(Step 4, 3.0s-4.0s) lunging spin-slice upward flare{vfx_clause} -> (Step 5, 4.0s-5.0s) nimble backstep recovery to guard",
                ),
                (
                    "3-Hit Heavy Attack Combo",
                    "action",
                    f"Streets of Rage 4 & Shinobi fast-paced arcade 3-strike cleave with {p_w} (0.5s step-in -> 3.5s rapid combat -> 1.0s recovery): (Step 1, 0.0s-0.5s) heavy power step-in advance -> "
                    f"(Step 2, 0.5s-1.8s) horizontal cleave -> (Step 3, 1.8s-3.0s) deep cross-step torso torque into overhead slam -> "
                    f"(Step 4, 3.0s-4.0s) full-body advancing 360-degree whirlwind cleave causing knockdown with flaming shockwaves -> (Step 5, 4.0s-5.0s) stepping back to guard",
                ),
                (
                    "Heavy Double-Damage Charge Hit",
                    "action",
                    f"Classic Beat 'em Up 1.5s Hold-to-Charge Super Attack: (0.0s-1.5s) holds crouched charge gathering fiery kinetic energy along {p_w} and {s_w} -> "
                    f"(1.5s-4.0s) erupts forward into an explosive dual-blade cross-breaker slash with blazing shockwaves -> "
                    f"(4.0s-5.0s) slides to a halt and recovers smoothly to dual-wield guard",
                ),
                (
                    "Sprint Dash Light Strike",
                    "action",
                    f"Shinobi blitz attack: bursts from sprint into a low-friction sliding lunge thrust with {s_w} closing distance across the floor, springing back to ready stance",
                ),
                ("Stationary Dodge Counter", "action", f"Streets of Rage 4 neutral blitz: zero horizontal camera drift, evasive duck/spin parry on the spot and instant counter-strike with {p_w} and {s_w}"),
                ("Hit Reaction (Hurt Recoil)", "action", f"Takes an enemy hit: torso snaps back in pain, grimace, rapid recoil and recovery back into idle guard stance"),
                ("Knockdown & Ground Recovery (Drop Down & Get Up)", "action", f"Takes heavy fatal blow: falls backward -> drops flat onto floor (invulnerable down frames) -> tactical ground roll -> athletic spring to feet into idle guard"),
            ]
        elif arch == "blade":
            if is_top_down:
                raw_moves = [
                    ("Sword Guard Idle", "loop", f"Standing alert in overhead RPG view gripping {weapon} in ready guard stance with subtle breathing bounce"),
                    ("Walk South (Sword Ready)", "loop", f"Walking downward toward camera carrying {weapon} safely at the side in rhythmic steps"),
                    ("Walk North (Sword Ready)", "loop", f"Walking upward away from camera holding {weapon} in ready guard"),
                    ("Walk Lateral (East/West)", "loop", f"Striding sideways across the plane carrying {weapon} in combat readiness"),
                    ("Horizontal Blade Slash", "action", f"Steps forward aggressively and swings {weapon} in a wide 180-degree horizontal cutting arc with a bright blade trail, stepping back to sword guard"),
                    ("Spinning Whirlwind Cut", "action", f"Advances with a pivot step into a full 360-degree spinning slash with {weapon} creating a circular blade wave, settling cleanly back to guard"),
                    ("Skyward Sword Raise", "action", f"Lifts {weapon} high overhead as a heroic energy glint flashes along the blade, returning to resting sword guard"),
                ]
            elif is_runner:
                raw_moves = [
                    ("Blade Runner Idle", "loop", f"High-energy runner stance gripping {weapon} in a sleek low guard with rhythmic breathing"),
                    ("Katana Sprint Dash", "loop", f"Leaning forward in a high-speed sprint with {weapon} held trailing at the hip in aerodynamic posture"),
                    ("Lateral Blade Dodge", "loop", f"Quick side-to-side evasive footwork while sprinting and keeping {weapon} poised"),
                    ("Low Slash Slide", "action", f"Drops into a fast low slide while sweeping {weapon} horizontally to clear low hazards, springing back to stance"),
                    ("Aerial Vault Slash", "action", f"Leaps high over an obstacle while performing a mid-air somersault cut with {weapon}, landing cleanly in stance"),
                    ("Blade Parry Deflect", "action", f"Raises {weapon} in a crisp defensive cross-parry that deflects an incoming hazard with sparks, resetting to stance"),
                    ("Sheathe & Flourish", "action", f"Spins {weapon} with a masterful flourish and snaps into a victorious sword salute before returning to ready stance"),
                ]
            elif is_beat_em_up:
                raw_moves = [
                    ("Sword Guard Idle", "loop", f"Poised combat stance gripping {weapon} at the ready, subtle breathing and weight shift"),
                    ("Walk Forward (Sword Ready)", "loop", f"Confident forward combat stride keeping {weapon} raised in defensive guard, treadmill root motion"),
                    ("Blade Sprint Dash", "loop", f"Low-center-of-gravity high-speed shinobi dash forward with {weapon} poised for a rapid strike"),
                    (
                        "3-Hit Light Blade Slash Combo",
                        "action",
                        f"Shinobi: Art of Vengeance advancing 3-hit speed combo with {weapon}: (Step 1) rapid slip-step forward diagonal cut -> "
                        f"(Step 2) cross-pivot rising slash -> (Step 3) lunging thrust with gleaming blade streak, hopping back to sword guard",
                    ),
                    (
                        "3-Hit Heavy Sword Cleave Combo",
                        "action",
                        f"Streets of Rage 4 advancing 3-hit heavy combo with {weapon}: (Step 1) heavy power step-in wide cleave -> "
                        f"(Step 2) cross-step torque overhead slam -> (Step 3) advancing 360-degree whirlwind knockdown cut, stepping back to guard",
                    ),
                    (
                        "Heavy Double-Damage Sword Charge Hit",
                        "action",
                        f"Crouches into kinetic charge windup with {weapon} glowing, then lunges forward in an explosive double-damage breaker cleave",
                    ),
                    (
                        "Sprint Dash Blade Strike",
                        "action",
                        f"High-velocity sliding blitz lunge with {weapon} cutting forward across the stage, recovering cleanly to sword guard",
                    ),
                    ("Stationary Sword Dodge Counter", "action", f"SoR4 neutral blitz: zero horizontal camera drift, evasive spin-parry on the spot and instant counter-slash with {weapon}"),
                    ("Hit Reaction (Hurt Recoil)", "action", f"Takes incoming enemy blow: torso snaps back in pain, rapid recoil and recovery back into {weapon} guard stance"),
                    ("Knockdown & Ground Recovery (Drop Down & Get Up)", "action", f"Takes fatal blow: falls backward -> drops flat onto floor (i-frames) -> tactical ground roll -> athletic spring to feet holding {weapon}"),
                ]
            else:
                raw_moves = [
                    ("Sword Guard Idle", "loop", f"Poised two-handed combat stance gripping {weapon} at the ready, subtle breathing and cape/cloth sway"),
                    ("Combat Advance Walk", "loop", f"Confident forward stride keeping {weapon} raised in defensive guard"),
                    ("Blade Rush Dash", "loop", f"Low-center-of-gravity forward dash with {weapon} poised for a rapid strike"),
                    ("Quick Iaido Slash", "action", f"Darts forward with a slip-step to deliver a lightning-fast horizontal slash with {weapon}, hopping back to sword guard"),
                    (
                        "3-Hit Sword Combo",
                        "action",
                        f"Shinobi & SoR4 advancing 3-hit blade combo with {weapon}: (Step 1) forward step-in downward diagonal cut -> "
                        f"(Step 2) cross-pivot rising reverse slash -> (Step 3) deep lunging forward thrust, stepping back to idle sword guard",
                    ),
                    ("Rising Dragon Cleave", "action", f"Steps in and crouches briefly then leaps into an upward arcing slash with {weapon}, landing firmly back in sword guard stance"),
                    ("Crescent Blade Wave", "action", f"Steps forward channeling energy along {weapon} and unleashes a wide crescent slash wave, settling back into resting guard"),
                ]
        elif arch == "heavy_melee":
            raw_moves = [
                ("Heavy Weapon Idle", "loop", f"Grounded powerhouse stance hefting {weapon} across the shoulders with steady rhythmic breathing"),
                ("Lumbering Combat Walk", "loop", f"Heavy intimidating forward stride carrying {weapon} with powerful momentum"),
                ("Bull Charge Rush", "loop", f"Aggressive forward charge leaning in with {weapon} held low to plow through obstacles"),
                ("Wide Horizontal Cleave", "action", f"Stomps lead boot forward and swings {weapon} in a massive horizontal arc with heavy follow-through, pulling back into idle stance"),
                ("Crushing Overhead Slam", "action", f"Steps in, hoists {weapon} high above the head, and slams it down onto the ground plane with impact sparks, recovering to stance"),
                ("2-Hit Berserker Smash Combo", "action", f"Streets of Rage 4 advancing heavy combo: power step-in brutal side swing followed by a cross-stepping upward heave with {weapon}, settling back into guard"),
                ("Earthshaker Spin Finisher", "action", f"Advances with a torso pivot using the momentum of {weapon} for a full 360-degree heavy whirlwind swing, anchoring back into resting stance"),
            ]
        elif arch == "polearm":
            raw_moves = [
                ("Lancer Guard Idle", "loop", f"Poised sentinel stance holding {weapon} diagonally across the body with steady breathing"),
                ("Forward March", "loop", f"Disciplined forward combat walk holding {weapon} at the ready"),
                ("Lunge Dash", "loop", f"Fast forward rush with the tip of {weapon} leveled ahead"),
                ("Lightning Lunge Thrust", "action", f"Driving front-foot lunge extending {weapon} in a long-reach piercing forward thrust, retracting crisply back to guard stance"),
                ("Sweeping Halberd Arc", "action", f"Steps across and swings {weapon} in a wide low-to-high crescent sweep, recovering smoothly to lancer guard"),
                ("Flurry Impale Combo", "action", f"Shinobi advancing polearm combo: slip-step thrust, cross-step rapid impale, and spinning shaft lunge flourish with {weapon}, returning to idle"),
                ("Vaulting Sky Dive", "action", f"Pole-vaults upward using {weapon} and strikes downward at an angle before resetting to guard stance"),
            ]
        elif arch == "agile_melee":
            raw_moves = [
                ("Shinobi Crouch Idle", "loop", f"Low agile combat stance holding {weapon} in reverse grip with alert breathing"),
                ("Stealth Stride", "loop", f"Light-footed silent forward walk keeping {weapon} tucked ready to strike"),
                ("Shadow Sprint", "loop", f"High-speed ninja dash leaning low with {weapon} trailing behind"),
                ("Twin-Blade Cross Slash", "action", f"Shinobi: Art of Vengeance darting slip-step forward with a rapid X-shaped cross slash using {weapon}, flipping back into crouch idle"),
                ("Acrobatic Flurry Combo", "action", f"Shinobi: Art of Vengeance advancing 4-hit slash combo with {weapon}: slip-step cut, cross-pivot slash, low lunge stab, and spinning somersault slice, hopping back to stance"),
                ("Cartwheel Evasion Strike", "action", f"Performs a swift advancing cartwheel flip while slashing with {weapon}, landing softly back in idle stance"),
                ("Shadow Lotus Finisher", "action", f"Lunges forward in a blur of blade streaks with {weapon} and strikes a poised shinobi hand-seal before returning to idle"),
            ]
        elif arch == "ranged_bow":
            raw_moves = [
                ("Archer Ready Idle", "loop", f"Poised marksman stance holding {weapon} with an arrow at the ready and calm breathing"),
                ("Ranger Stride", "loop", f"Light agile forward walk carrying {weapon} in ready position"),
                ("Evasive Sidestep Dash", "loop", f"Quick lateral ranger dash keeping {weapon} stabilized"),
                ("Quick Snap Shot", "action", f"Steps into a braced stance, draws {weapon} in a split second, and looses a high-velocity arrow, recovering to ready stance"),
                ("Triple Arrow Volley", "action", f"Advances a step, nocks three arrows at once on {weapon}, and fires a spread volley forward, resetting to archer idle"),
                ("Acrobatic Backstep Shot", "action", f"Hops nimblly backward while firing an aimed shot from {weapon} mid-air, landing cleanly in stance"),
                ("Charged Piercing Arrow", "action", f"Draws {weapon} back as glowing energy condenses on the arrowhead, releasing a powerful piercing shot and recovering"),
            ]
        elif arch == "magic":
            raw_moves = [
                ("Arcane Channel Idle", "loop", f"Mystical stance holding {weapon} as soft magical energy pulses and robes flutter gently"),
                ("Mage Glide Walk", "loop", f"Graceful forward stride holding {weapon} upright with poise"),
                ("Astral Hover Dash", "loop", f"Levitate-gliding forward dash with {weapon} leading and arcane trail"),
                ("Arcane Bolt Cast", "action", f"Steps forward and thrusts {weapon} to fire a concentrated arcane energy bolt, returning smoothly to caster idle"),
                ("Elemental Sigil Summon", "action", f"Raises {weapon} high and traces a glowing magical rune circle in the air, settling back to resting stance"),
                ("Twin Spell Barrage Combo", "action", f"Advances with a flowing cross-step while swinging {weapon} in a figure-eight pattern unleashing dual elemental waves, recovering to idle"),
                ("Grand Nova Invocation", "action", f"Channels a brilliant aura around {weapon} and releases an expanding arcane pulse before returning to stance"),
            ]
        elif arch == "ranged_firearm":
            raw_moves = [
                ("Tactical Aim Idle", "loop", f"Armed combat stance holding {weapon} aimed steady with subtle breathing"),
                ("Run-and-Gun Stride", "loop", f"Aggressive forward sprint keeping {weapon} leveled and ready to fire"),
                ("Booster Slide Dash", "loop", f"Low tactical crouch dash with {weapon} stabilized forward"),
                ("Rapid Burst Fire", "action", f"Steps into braced firing stance and fires a 3-round energy/muzzle burst from {weapon} with crisp recoil kickback, returning to idle"),
                ("Charged Plasma Cannon", "action", f"Charges glowing energy inside {weapon} and fires a heavy beam with recoil recovery back to stance"),
                ("Jump Hover Shot", "loop", f"Airborne tactical leap keeping {weapon} aimed at the target plane"),
                ("Tactical Reload & Salute", "action", f"Vents heat from {weapon}, locks in a fresh energy cell with a crisp click, and resets to combat idle"),
            ]
        elif arch == "vehicle_ship":
            raw_moves = [
                ("Thruster Hover Idle", "loop", f"Steady aerodynamic hover with {weapon} armed and rhythmic ion exhaust plumes"),
                ("Bank Left Roll", "loop", "Smoothly banks and tilts hull to the left showing upper armor plating"),
                ("Bank Right Roll", "loop", "Smoothly banks and tilts hull to the right showing upper armor plating"),
                ("Afterburner Boost", "loop", "Full-throttle forward thrust with elongated blue ion engine flares"),
                ("Primary Cannon Volley", "action", f"Fires rapid synchronized energy pulses from {weapon} with bright muzzle flares, returning to cruise hover"),
                ("Heavy Torpedo Salvo", "action", "Opens ordnance bays to launch homing photon torpedoes, closing bays and returning to cruise"),
                ("Aegis Shield Pulse", "action", "Projects a hexagonal energy barrier around the hull to deflect impact, settling back to hover"),
            ]
        elif arch == "custom_gear":
            raw_moves = [
                ("Equipped Guard Idle", "loop", f"Ready combat stance holding {weapon} with balanced posture and rhythmic breathing"),
                ("Forward Stride", "loop", f"Confident forward locomotion carrying {weapon} at the ready"),
                ("Agile Dash", "loop", f"Fast rushing sprint while keeping {weapon} poised for action"),
                ("Primary Gear Strike", "action", f"Steps forward aggressively and executes a signature attack using {weapon}, cleanly recovering back to idle stance"),
                ("Advancing Combo Technique", "action", f"Performs a fluid multi-step advancing strike sequence (step-in, cross-pivot, lunge) with {weapon}, resetting smoothly to resting pose"),
                ("Evasive Maneuver", "action", f"Dodges swiftly while guarding with {weapon} and returns to starting stance"),
                ("Signature Special Finisher", "action", f"Lunges in and unleashes an energized special ability using {weapon}, settling back into idle stance"),
            ]
        else:
            # Unarmed martial artist or runner preset adapted with character details
            raw_moves = [
                (m["name"], m.get("type", "loop"), m["description"])
                for m in preset["default_moves"]
            ]

        # Match exact slot count for the genre
        raw_moves = raw_moves[:target_count]
        moves: List[Dict[str, Any]] = []
        for idx, (m_name, m_type, m_desc) in enumerate(raw_moves):
            footwork_meta = cls.reason_combo_footwork_choreography(
                character_data=character_data,
                move_name=m_name,
                move_description=m_desc,
                genre=genre,
            )
            compiled_prompt = cls.build_pre_animation_wan_prompt(
                character_data=character_data,
                genre=genre,
                move_name=m_name,
                move_type=m_type,
                move_description=m_desc,
            )
            moves.append({
                "id": f"move_{idx+1}",
                "name": m_name,
                "type": m_type,
                "description": m_desc,
                "footwork_reasoning": footwork_meta["ai_decision_reasoning"],
                "footwork_style": footwork_meta["footwork_style"],
                "refined_prompt": compiled_prompt,
                "wan_seed": 42 + idx * 1007,
                "video_path": None,
                "sheet_path": None,
                "json_path": None,
            })
        moves = cls.ensure_mandatory_locomotion_loops(moves, character_data, genre)
        return moves

    @classmethod
    def ensure_mandatory_locomotion_loops(
        cls,
        moves: List[Dict[str, Any]],
        character_data: Dict[str, Any],
        genre: str,
    ) -> List[Dict[str, Any]]:
        """
        Enforces the mandatory architectural foundation for 2D arcade sprites:
        1. Idle Loop Animation (breathing/guard stance loop)
        2. Walking Animation Loop (locomotive stride loop closing distance)
        3. Action Behaviors (combos, strikes, dodges, knockdowns)
        If either Idle or Walk loop is missing, automatically synthesizes and injects it.
        """
        has_idle = any(
            str(m.get("type", "")).lower() == "loop"
            and any(k in str(m.get("name", "")).lower() for k in ("idle", "stance", "rest", "guard"))
            for m in moves
        )
        has_any_locomotion = any(
            str(m.get("type", "")).lower() == "loop"
            and any(k in str(m.get("name", "")).lower() for k in ("walk", "stride", "march", "advance", "run", "sprint", "dash", "hover", "thruster"))
            for m in moves
        )
        has_walk = any(
            str(m.get("type", "")).lower() == "loop"
            and any(k in str(m.get("name", "")).lower() for k in ("walk", "stride", "march", "advance"))
            for m in moves
        )

        arch = cls.detect_character_combat_archetype(character_data, genre=genre)
        weapon = arch["weapon_label"]
        char_name = character_data.get("name", "Character")

        result_moves = list(moves)
        if not has_idle:
            idle_name = "Combat Idle Stance"
            idle_desc = f"Poised ready combat stance holding {weapon} firmly, steady rhythmic breathing and subtle weight shift"
            idle_prompt = cls.build_pre_animation_wan_prompt(
                character_data=character_data,
                genre=genre,
                move_name=idle_name,
                move_type="loop",
                move_description=idle_desc,
            )
            result_moves.insert(0, {
                "id": "move_mandatory_idle",
                "name": idle_name,
                "type": "loop",
                "description": idle_desc,
                "footwork_reasoning": "Foundational Idle Loop: anchored combat stance with natural cyclic breathing.",
                "footwork_style": "Poised ready guard stance",
                "refined_prompt": idle_prompt,
                "wan_seed": 42,
                "video_path": None,
                "sheet_path": None,
                "json_path": None,
            })

        if not has_walk and not has_any_locomotion:
            walk_name = "Walk Forward"
            walk_desc = f"Confident forward combat stride closing distance on opponents carrying {weapon} at the ready, continuous treadmill stride loop"
            walk_prompt = cls.build_pre_animation_wan_prompt(
                character_data=character_data,
                genre=genre,
                move_name=walk_name,
                move_type="loop",
                move_description=walk_desc,
            )
            insert_pos = 1 if (result_moves and any(k in result_moves[0].get("name", "").lower() for k in ("idle", "stance"))) else 0
            result_moves.insert(insert_pos, {
                "id": "move_mandatory_walk",
                "name": walk_name,
                "type": "loop",
                "description": walk_desc,
                "footwork_reasoning": "Foundational Walking Loop: continuous treadmill stride contact pose matching at start and end for zero frame skipping.",
                "footwork_style": "Treadmill forward locomotive stride",
                "refined_prompt": walk_prompt,
                "wan_seed": 1049,
                "video_path": None,
                "sheet_path": None,
                "json_path": None,
            })

        for idx, m in enumerate(result_moves):
            m["id"] = f"move_{idx+1}"

        return result_moves

    def generate_smart_moveset_with_gemma(
        self,
        character_data: Dict[str, Any],
        genre: str,
        combat_notes: str = "",
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Uses Gemma 4: 31B / Nemotron-3-Ultra to intelligently design a complete character-specific
        move-set and pre-animation Wan 2.2 choreography instructions tailored to the character's weapon,
        signature gear, anatomy, and game genre, with Streets of Rage 4 & Shinobi: Art of Vengeance
        advancing combo footwork reasoning and mandatory Idle + Walk loops.
        """
        preset = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        target_count = len(preset["default_moves"])
        arch_info = self.detect_character_combat_archetype(character_data, genre=genre)

        if not use_llm:
            return self.get_smart_contextual_moves(character_data, genre)

        char_name = character_data.get("name") or "Hero"
        appearance = character_data.get("appearance") or ""
        gear = arch_info["weapon_label"]
        notes_clause = f"\nUser Combat / Style Directives: {combat_notes}" if combat_notes else ""

        system_prompt = (
            "You are a Lead 2D Gameplay Combat Director and Technical Animator specializing in arcade beat 'em ups "
            "('Streets of Rage 4' and 'Shinobi: Art of Vengeance' Lizardcube combat canon) and 2D sprite engines.\n"
            f"Your task is to design a tailored {target_count}-move animation set and pre-animation instructions for this character:\n"
            f"- Character Name: {char_name}\n"
            f"- Title / Role: {character_data.get('title') or arch_info['role_title']}\n"
            f"- Visual Appearance: {appearance}\n"
            f"- Equipped Weapon / Signature Gear: {gear}\n"
            f"- Detected Combat Archetype: {arch_info['archetype']} ({arch_info['combat_style_summary']})\n"
            f"- Game Genre & Camera: {genre} ({preset['perspective']}, {preset['camera']})"
            f"{notes_clause}\n\n"
            "MANDATORY COMBAT & LOCOMOTION REASONING PRINCIPLES:\n"
            "0. MANDATORY CORE TRIAD (IDLE LOOP + WALKING LOOP + ACTION BEHAVIORS):\n"
            "   Every character moveset MUST contain BOTH fundamental locomotion loops AND combat actions:\n"
            "   - Combat Idle Stance (type: 'loop'): Poised ready guard posture, rhythmic breathing and subtle weight shift holding weapons.\n"
            "   - Walking Animation Loop (type: 'loop'): Essential forward distance-closing combat stride holding weapons in guard (treadmill loop, zero frame skipping).\n"
            "   - Sprint Dash Loop (type: 'loop'): High-speed aggressive rush.\n"
            "   - Action Behaviors (type: 'action'): Dynamic multi-weapon combos, charged strikes, dodges, and knockdowns.\n"
            "   NEVER omit either the Idle Loop or the Walking Animation Loop!\n"
            "1. MULTI-WEAPON ASYMMETRY & DAMAGE TIERS:\n"
            "   - If character has multiple weapons (e.g., Heavy Sword + Magic Dagger), NEVER treat them identically.\n"
            "   - Primary Heavy Weapon: Dedicated to heavy poise damage, wide cleaving arcs, armor break, knockdowns, and slow high-impact swings.\n"
            "   - Secondary Light Weapon: Dedicated to speed, rapid combo filler, quick stabs, lunges, and elemental/spark VFX trails.\n"
            "   - Dual-Wield Synthesis: Combining both blades (sword parries incoming blow while dagger delivers point-blank puncture).\n"
            "2. DYNAMIC ADVANCING COMBO FOOTWORK ('STREETS OF RAGE 4' & 'SHINOBI: ART OF VENGEANCE' CANON):\n"
            "   - CRITICAL RULE: During any combo or multi-hit attack, the character MUST MOVE AND HIT—NEVER stand still like a statue with glued feet!\n"
            "   - Light / Fast Combos (Shinobi: Art of Vengeance style): Pair strikes with darting front-foot slip-steps, rotational cross-pivots, low lunging thrusts, or acrobatic somersault slices, followed by a nimble backstep hop to neutral guard.\n"
            "   - Heavy / Power Combos (Streets of Rage 4 style): Pair cleaves with grounded power step-ins, deep cross-step hip torque, and full-body advancing 360-degree whirlwind pivot steps, followed by a planted recovery step back to guard.\n"
            "3. STREETS OF RAGE 4 BEAT 'EM UP BEHAVIORAL CANON (10 SIGNATURE MOVES):\n"
            "   - Combat Idle Stance (Cyclic Loop): Ready posture holding weapons, breathing and weight shift.\n"
            "   - Walk Forward (Cyclic Stride Loop): Treadmill-anchored locomotion with weapons at the ready.\n"
            "   - Sprint Dash (Cyclic Stride Loop): Low-center-of-gravity high-speed run with weapon trailing.\n"
            "   - 3-Hit Light Attack Combo (Action): Advancing 3-step flurry with secondary light weapon (slip-step slash -> cross-pivot stab -> lunging magic spark slice -> backstep recovery).\n"
            "   - 3-Hit Heavy Attack Combo (Action): Advancing 3-step cleave with primary heavy weapon (power step-in cleave -> cross-step overhead slam -> 360 pivot spin knockdown -> recovery step).\n"
            "   - Heavy Double-Damage Charge Hit (Action): Braced windup charge -> explosive forward lunging double-damage blow with shockwave.\n"
            "   - Sprint Dash Light Strike (Action): High-speed sliding blitz lunge attack closing distance instantly with light weapon.\n"
            "   - Stationary Dodge Counter (Action): SoR4 neutral blitz: zero horizontal camera drift, evasive duck/spin parry on the spot and immediate counter.\n"
            "   - Hit Reaction (Action): Torso snaps back in pain, rapid recoil and recovery back into guard.\n"
            "   - Knockdown & Ground Recovery (Action): 4-phase staged timeline: fatal impact fall -> drop flat onto floor (i-frames) -> tactical ground roll -> athletic rise to feet into guard.\n"
            "4. ZERO-FRAME-SKIP LOOP LAWS & RETURN-TO-IDLE:\n"
            "   - For 'loop' moves (Walk/Run): Character starts already in full dynamic stride contact pose, cycles steps, and returns to that exact stride pose at t=5.0s for infinite loop.\n"
            f"   - For 'action' moves: Starts in {arch_info['idle_posture']}, advances through the strike/combo footwork, and cleanly steps back into {arch_info['idle_posture']} at the starting coordinate by t=5.0s.\n"
            "   - Fixed static camera, zero camera pan/zoom, pure flat solid white studio background (#FFFFFF), high-key studio lighting, no shadows.\n\n"
            "Respond ONLY with a valid JSON object following this exact schema:\n"
            "{\n"
            '  "combat_summary": "1-sentence summary of the character\'s tailored animation style and footwork",\n'
            '  "moves": [\n'
            "    {\n"
            '      "name": "Move Name",\n'
            '      "type": "loop or action",\n'
            '      "footwork_reasoning": "AI reasoning explaining how the character\'s feet, hips, and body step/lunge/pivot during this move (inspired by Streets of Rage 4 & Shinobi: Art of Vengeance)",\n'
            '      "description": "Clear pre-animation choreography instructions describing how the character steps, moves, and strikes with their weapon",\n'
            '      "wan_prompt": "Complete temporally-staged Wan 2.2 prompt on pure flat solid white studio background #FFFFFF"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_msg = (
            f"Generate {target_count} character-specific moves and pre-animation instructions for {char_name} "
            f"(equipped with {gear}) in genre '{genre}'."
            + (f" Additional instructions: {combat_notes}" if combat_notes else "")
        )

        try:
            raw_reply = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                task_role="moveset",
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.MULTILINE)
            parsed = json.loads(clean_json)
            raw_list = parsed.get("moves", []) if isinstance(parsed, dict) else parsed
            if not isinstance(raw_list, list) or len(raw_list) == 0:
                raise ValueError("Empty moves list from Gemma")

            moves: List[Dict[str, Any]] = []
            for idx, item in enumerate(raw_list[:target_count]):
                m_name = str(item.get("name") or f"Move {idx+1}").strip()
                m_type = "loop" if str(item.get("type", "")).lower() == "loop" else "action"
                m_desc = str(item.get("description") or item.get("action_summary") or m_name).strip()
                footwork_meta = self.reason_combo_footwork_choreography(
                    character_data=character_data,
                    move_name=m_name,
                    move_description=m_desc,
                    genre=genre,
                )
                m_footwork = str(item.get("footwork_reasoning") or footwork_meta["ai_decision_reasoning"]).strip()
                m_prompt = str(item.get("wan_prompt") or item.get("refined_prompt") or "").strip()
                if not m_prompt or ("#ffffff" not in m_prompt.lower() and "white" not in m_prompt.lower()):
                    m_prompt = self.build_pre_animation_wan_prompt(
                        character_data=character_data,
                        genre=genre,
                        move_name=m_name,
                        move_type=m_type,
                        move_description=m_desc,
                    )
                moves.append({
                    "id": f"move_{idx+1}",
                    "name": m_name,
                    "type": m_type,
                    "description": m_desc,
                    "footwork_reasoning": m_footwork,
                    "footwork_style": footwork_meta["footwork_style"],
                    "refined_prompt": m_prompt,
                    "wan_seed": 42 + idx * 1007,
                    "video_path": None,
                    "sheet_path": None,
                    "json_path": None,
                })

            # Pad with contextual fallback moves if LLM returned fewer than target_count
            if len(moves) < target_count:
                fallback_moves = self.get_smart_contextual_moves(character_data, genre)
                for fb in fallback_moves[len(moves):target_count]:
                    fb["id"] = f"move_{len(moves)+1}"
            moves = self.ensure_mandatory_locomotion_loops(moves, character_data, genre)
            return moves
        except Exception as e:
            print(f"[SpritesPipeline] Using smart contextual moves fallback: {e}")
            return self.get_smart_contextual_moves(character_data, genre)

    def refine_custom_animation_with_gemma(
        self,
        character_data: Dict[str, Any],
        genre: str,
        anim_name: str,
        raw_description: str,
        anim_type: str = "action",
    ) -> Dict[str, Any]:
        """
        Uses Gemma 4: 31B to translate the user's custom animation idea into a weapon-aware
        micro-choreography prompt with Streets of Rage 4 & Shinobi: Art of Vengeance advancing
        footwork reasoning, strict camera lockdown, and anchor rules.
        """
        char_name = character_data.get("name", "Character")
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        arch_info = self.detect_character_combat_archetype(character_data, genre=genre)
        weapon_label = arch_info["weapon_label"]
        is_loop = anim_type.lower() == "loop"
        footwork_meta = self.reason_combo_footwork_choreography(
            character_data=character_data,
            move_name=anim_name,
            move_description=raw_description,
            genre=genre,
        )

        system_prompt = (
            "You are a Lead Gameplay Animator and Technical Director for 2D arcade video games "
            "specializing in 'Streets of Rage 4' and 'Shinobi: Art of Vengeance' dynamic combat animation.\n"
            f"Game Genre: {genre} ({genre_info['perspective']})\n"
            f"Character: {char_name} ({character_data.get('appearance', '')})\n"
            f"Equipped Weapon / Signature Gear: {weapon_label} ({arch_info['combat_style_summary']})\n"
            f"Animation Name: {anim_name}\n"
            f"Animation Type: {'Seamless Looping Cycle' if is_loop else 'Action / Combo Attack with Advancing Footwork & Return-to-Idle Recovery'}\n\n"
            "ANIMATION LAWS:\n"
            f"1. WEAPON & CHARACTER CONSISTENCY: {arch_info['forbidden_actions']} Ensure {char_name} properly holds/uses {weapon_label}.\n"
            "2. MOVE-AND-HIT FOOTWORK LAW (Streets of Rage 4 & Shinobi: Art of Vengeance): For any attack or combo, the character MUST NOT stand still with glued feet! "
            "Choreograph dynamic forward slip-steps, power step-ins, cross-pivots, lunges, or acrobatic spins with each strike, followed by a crisp recovery backstep to the origin.\n"
            f"3. Camera: {genre_info['camera']}. The camera must NEVER pan, tilt, shake, or zoom.\n"
            "4. Background: Must remain fixed pure solid flat white studio background #FFFFFF, high-key studio lighting, zero shadows.\n"
            + (
                f"5. LOOPING RULE: This animation is an endless loop. The character's motion and {weapon_label} position at t=5.0s must seamlessly match their starting pose at t=0.0s."
                if is_loop
                else f"5. RECOVERY RULE: Frame 0 starts in the character's Idle pose ({arch_info['idle_posture']}). The middle delivers the advancing step-and-hit action using {weapon_label}. By t=5.0s, the character MUST cleanly step back into the exact Idle resting pose."
            )
            + "\n\nRespond ONLY with valid JSON following this exact schema:\n"
            "{\n"
            '  "action_summary": "One sentence summary of the advancing movement and strike incorporating the character\'s weapon/gear",\n'
            '  "footwork_reasoning": "Why this specific step-in, cross-pivot, lunge, and recovery footwork fits this move",\n'
            '  "choreography_prompt": "Micro-level footwork, limb, body, and weapon action breakdown: Step 1 step-in, Step 2 cross-pivot, Step 3 peak lunge/finisher, Step 4 recovery backstep",\n'
            '  "wan_prompt": "Complete prompt for Wan 2.2 specifying 2D sprite animation, advancing footwork, weapon consistency, fixed static camera, zero camera movement, on pure flat solid white studio background #FFFFFF",\n'
            '  "motion_intensity": 3\n'
            "}"
        )

        user_msg = f"Animation Name: {anim_name}. User description of movement/behavior: {raw_description}"
        try:
            raw_reply = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                task_role="quick_edit",
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_reply.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
            if "footwork_reasoning" not in data:
                data["footwork_reasoning"] = footwork_meta["ai_decision_reasoning"]
        except Exception:
            # Weapon-aware and footwork-aware fallback prompt
            desc_text = raw_description.strip() if raw_description and raw_description.strip() else f"executes {anim_name} with {weapon_label}"
            if weapon_label.lower() not in desc_text.lower():
                desc_text = f"{desc_text} (wielding {weapon_label})"
            wan_prompt = self.build_pre_animation_wan_prompt(
                character_data=character_data,
                genre=genre,
                move_name=anim_name,
                move_type="loop" if is_loop else "action",
                move_description=desc_text,
            )
            data = {
                "action_summary": f"{char_name} performs {anim_name} with {weapon_label} ({footwork_meta['footwork_style']}): {raw_description}",
                "footwork_reasoning": footwork_meta["ai_decision_reasoning"],
                "choreography_prompt": f"{char_name} ({arch_info['idle_posture']}) executes {anim_name} using {footwork_meta['footwork_style']}: {desc_text}.",
                "wan_prompt": wan_prompt,
                "motion_intensity": 3,
            }

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Wan 2.2 Animation Synthesis
    # ─────────────────────────────────────────────────────────────────────────

    def synthesize_stride_keyframe_with_qwen(
        self,
        character_data: Dict[str, Any],
        move_name: str = "walk",
        output_dir: Optional[str] = None,
    ) -> Image.Image:
        """
        Uses Qwen Image Edit on Modal to synthesize a dynamic locomotion stride keyframe
        (left leg forward, right leg back, contact pose, weapons poised, centered on treadmill)
        derived directly from the character's Master Sprite.
        Anchors both Frame 0 and Frame 80 in Wan 2.2 FLF2V to eliminate frame skipping.
        """
        if "stride_keyframes" not in character_data:
            character_data["stride_keyframes"] = {}

        move_key = "sprint" if any(k in move_name.lower() for k in ("sprint", "run", "dash")) else "walk"
        if move_key in character_data["stride_keyframes"]:
            return character_data["stride_keyframes"][move_key]

        master_pil = character_data.get("master_sprite_pil")
        if master_pil is None:
            master_path = character_data.get("master_sprite_path")
            if master_path and os.path.exists(str(master_path)):
                master_pil = Image.open(master_path).convert("RGB")
            else:
                master_pil, master_path = self.synthesize_master_character_sprite(character_data)

        char_name = character_data.get("name", "Hero")
        is_sprint = move_key == "sprint"

        stride_instruction = (
            f"Transform this character's posture from standing idle into a high-speed athletic sprint stride contact pose: "
            f"deep forward lean, left foot lunging forward, right leg trailing back, arms pumping with weapons firmly in hand, "
            f"centered in frame on pure flat solid white studio background #FFFFFF, preserve exact face, armor, and gear colors"
            if is_sprint else
            f"Transform this character's posture from standing idle into a dynamic walking stride contact pose: "
            f"left heel planting forward, right foot stepping back, upright balanced torso, weapons held in ready guard, "
            f"centered in frame on pure flat solid white studio background #FFFFFF, preserve exact face, armor, and gear colors"
        )

        print(f"[SpritesPipeline] Synthesizing Qwen {move_key} stride keyframe for '{char_name}'...")
        try:
            from backend import qwen_image_edit_remote
            import io
            img_bytes_list = qwen_image_edit_remote(
                base_image=master_pil,
                prompt=stride_instruction,
                negative_prompt="blurry, distorted face, changing clothes, cropped limbs, shadows, colored background, green screen",
                steps=18,
                seed=42,
            )
            if img_bytes_list and len(img_bytes_list) > 0:
                stride_img = Image.open(io.BytesIO(img_bytes_list[0])).convert("RGB")
                save_folder = Path(output_dir or self.char_dir)
                stride_path = save_folder / f"{char_name}_{move_key}_stride_keyframe.png"
                stride_img.save(str(stride_path))
                character_data["stride_keyframes"][move_key] = stride_img
                print(f"[SpritesPipeline] Qwen {move_key} stride keyframe saved: {stride_path}")
                return stride_img
        except Exception as e:
            print(f"[SpritesPipeline] Notice: Qwen stride synthesis fallback ({e}). Using master sprite as base anchor.")

        character_data["stride_keyframes"][move_key] = master_pil
        return master_pil

    def synthesize_animation_clip(
        self,
        character_data: Dict[str, Any],
        move_data: Dict[str, Any],
        wan_seed: int = 42,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        resolution: Any = (832, 832),
    ) -> str:
        """
        Synthesizes a 5-second video clip using Wan 2.2 on Modal.
        For Locomotion loops: anchors Frame 0 & Frame 80 to Qwen Stride Keyframe (Zero Frame Skipping).
        For Idle loops: anchors Frame 0 & Frame 80 to Master Sprite (Sine-wave cyclic breathing).
        For Action moves: anchors Frame 0 & Frame 80 to Master Sprite (Clean return-to-idle).
        Uses intelligent square canvas padding (e.g. 832x832 or 1024x1024) to ensure wide VFX and weapon sweeps never clip.
        """
        anim_name = move_data.get("name", "move")
        is_loop = move_data.get("type", "loop") == "loop"
        char_name = character_data.get("name", "Character")
        is_locomotion_loop = is_loop and any(k in anim_name.lower() for k in ("walk", "sprint", "run", "dash"))
        target_w, target_h = self.parse_resolution(resolution)

        master_sprite_path = character_data.get("master_sprite_path")
        master_sprite_pil = character_data.get("master_sprite_pil")
        if master_sprite_pil is None:
            if master_sprite_path and os.path.exists(str(master_sprite_path)):
                master_sprite_pil = Image.open(master_sprite_path).convert("RGB")
                character_data["master_sprite_pil"] = master_sprite_pil
            else:
                seed_val = int(character_data.get("master_seed", 42))
                if progress_cb:
                    progress_cb(0.05, f"Synthesizing master sprite for '{char_name}' on Z-Image Turbo...")
                master_sprite_pil, master_sprite_path = self.synthesize_master_character_sprite(
                    character_data, seed=seed_val
                )
                character_data["master_sprite_pil"] = master_sprite_pil
                character_data["master_sprite_path"] = master_sprite_path
        elif not master_sprite_path or not os.path.exists(str(master_sprite_path)):
            temp_path = self.char_dir / f"{char_name}_master.png"
            master_sprite_pil.save(str(temp_path))
            master_sprite_path = str(temp_path)
            character_data["master_sprite_path"] = master_sprite_path

        # Dual keyframe preparation with intelligent square canvas padding (guaranteeing VFX and reach room)
        if is_locomotion_loop:
            if progress_cb:
                progress_cb(0.10, f"Synthesizing Qwen stride contact keyframe for '{anim_name}'...")
            stride_pil = self.synthesize_stride_keyframe_with_qwen(character_data, move_name=anim_name)
            start_sprite = self.pad_sprite_to_square_canvas(stride_pil, target_size=(target_w, target_h))
            last_sprite = start_sprite  # Matching stride contact pose at start and end eliminates frame skipping
        else:
            start_sprite = self.pad_sprite_to_square_canvas(master_sprite_pil, target_size=(target_w, target_h))
            last_sprite = start_sprite  # Anchors return-to-idle for actions, or closed cyclic loop for resting idle

        # Compile Wan prompt
        refined_wan = move_data.get("refined_prompt") or move_data.get("description", "")
        if is_locomotion_loop:
            loop_clause = "continuous treadmill stride loop, start stride contact pose matches end stride contact pose perfectly with zero frame skipping"
        elif is_loop:
            loop_clause = "seamless cyclic breathing loop, continuous motion, start resting stance matches end resting stance"
        else:
            loop_clause = "starts in resting idle pose, performs dynamic fast-paced arcade action, cleanly recovers back to resting idle pose at end"

        full_wan_prompt = (
            f"2D game sprite animation of {char_name} executing {anim_name}: {refined_wan}. "
            f"{loop_clause}. Fixed tripod shot, static arcade 2D camera plane, zero camera pan, zero zoom, "
            f"on pure flat solid seamless white studio background #FFFFFF, high-key studio lighting, no floor shadows, clean outline"
        )

        # Dynamic weapon retention and anti-disappearing negative prompt
        weapon_tiers = self.parse_weapons_and_combat_tiers(character_data)
        p_w = (weapon_tiers.get("primary_heavy_weapon") or "").strip()
        s_w = (weapon_tiers.get("secondary_light_weapon") or "").strip()
        weapon_loss_negatives = [
            "disappearing weapons", "dropping weapon", "missing weapon",
            "empty hands", "bare hands", "bare fist", "hands merging",
            "morphing weapon", "holster", "sheathing", "weapon fading out",
            "sluggish movement", "slow motion", "floaty animation", "hesitating",
            "low action", "static posture", "frozen limbs"
        ]
        if p_w:
            weapon_loss_negatives.extend([f"missing {p_w}", f"vanishing {p_w}", f"dropping {p_w}"])
        if s_w:
            weapon_loss_negatives.extend([f"missing {s_w}", f"vanishing {s_w}", f"dropping {s_w}"])

        dynamic_negative_prompt = (
            "camera motion, camera pan, camera tilt, zoom, dynamic background, complex background, "
            "scenery, floor shadows, ground clutter, perspective shift, rotation, 3d camera orbit, "
            "dark background, colored background, green screen, chroma green, "
            + ", ".join(weapon_loss_negatives)
        )

        if progress_cb:
            progress_cb(0.15, f"Synthesizing '{anim_name}' with Wan 2.2 FLF2V at {target_w}x{target_h} (Seed {wan_seed})...")

        def _safe_progress(pct, *args, **kwargs):
            if not progress_cb:
                return
            msg = kwargs.get("desc") or (args[0] if args else "")
            try:
                progress_cb(pct, desc=msg)
            except TypeError:
                try:
                    progress_cb(pct, msg)
                except Exception:
                    pass
            except Exception:
                pass

        char_name = character_data.get("name", "hero")
        safe_char = re.sub(r'[^a-zA-Z0-9_\-]', '_', char_name.lower())
        safe_anim = re.sub(r'[^a-zA-Z0-9_\-]', '_', anim_name.lower())
        clip_filename = f"{safe_char}_{safe_anim}_clip.mp4"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        target_clip_path = self.clips_dir / clip_filename

        local_video_path, _ = run_remote_inference(
            resized_image=start_sprite,
            processed_last_image=last_sprite,
            prompt=full_wan_prompt,
            steps=8,
            negative_prompt=dynamic_negative_prompt,
            num_frames=81,  # 81 frames @ 16 fps ≈ 5.06s
            guidance_scale=2.0,
            guidance_scale_2=1.5,
            current_seed=int(wan_seed),
            scheduler_name="UniPCMultistep",
            flow_shift=3.0,
            frame_multiplier=16,
            quality=6,
            duration_seconds=5.0,
            safe_mode=True,
            deflicker_frames=True,
            progress=_safe_progress,
            last_frame_image=last_sprite,
            width=target_w,
            height=target_h,
            save_dir=str(self.clips_dir),
            output_filename=clip_filename,
        )

        if local_video_path and Path(local_video_path).exists():
            if Path(local_video_path).resolve() != target_clip_path.resolve():
                shutil.copy2(local_video_path, target_clip_path)
            final_clip_path = str(target_clip_path)
        elif target_clip_path.exists():
            final_clip_path = str(target_clip_path)
        else:
            final_clip_path = local_video_path

        move_data["video_path"] = final_clip_path
        move_data["wan_seed"] = wan_seed
        move_data["resolution"] = f"{target_w}x{target_h}"

        if progress_cb:
            progress_cb(1.0, f"Animation '{anim_name}' synthesized successfully -> {final_clip_path}")

        return final_clip_path

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Sprite Cutter & Sheet Exporter
    # ─────────────────────────────────────────────────────────────────────────

    def cut_and_export_sprite_sheet(
        self,
        character_data: Dict[str, Any],
        move_data: Dict[str, Any],
        target_fps: int = 12,
        chroma_tolerance: float = 38.0,
        bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
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
            bg_color=bg_color,
            tolerance=chroma_tolerance,
            output_dir=save_dir,
        )

        move_data["sheet_path"] = result["sheet_path"]
        move_data["json_path"] = result["json_path"]
        move_data["fps"] = target_fps
        move_data["total_frames"] = result["frame_count"]

        return result
