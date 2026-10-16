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

        def _has_word(words: List[str]) -> Optional[str]:
            for w in words:
                if re.search(r"\b" + re.escape(w) + r"s?\b", combined):
                    return w
            return None

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
        that locks the character's identity, equipped weapon/gear, start/end posture, and camera.
        """
        char_name = character_data.get("name") or "Character"
        appearance = character_data.get("appearance") or "2D game character"
        art_style = character_data.get("art_style") or "16-bit arcade pixel art style"
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        arch = cls.detect_character_combat_archetype(character_data, genre=genre)
        weapon_label = arch["weapon_label"]
        idle_posture = arch["idle_posture"]
        is_loop = str(move_type).lower() == "loop"

        invariant_clause = (
            f"[Character & Weapon Invariants]: 2D game sprite of {char_name} ({appearance}, {art_style}), "
            f"strictly preserving exact costume proportions, silhouette, and equipped {weapon_label} without morphing or vanishing."
        )
        if is_loop:
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
                f"Phase 2 (1.2s-3.6s): dynamically executes {move_name} ({move_description}); "
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
        equipped weapon/archetype and the selected game genre. Prevents mismatched moves (such as
        boxing animations on a sword-wielding character).
        """
        preset = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        target_count = len(preset["default_moves"])
        arch_info = cls.detect_character_combat_archetype(character_data, genre=genre)
        arch = arch_info["archetype"]
        weapon = arch_info["weapon_label"]
        is_top_down = "top-down" in genre.lower() or "rpg" in genre.lower()
        is_runner = "runner" in genre.lower()

        if arch == "blade":
            if is_top_down:
                raw_moves = [
                    ("Sword Guard Idle", "loop", f"Standing alert in overhead RPG view gripping {weapon} in ready guard stance with subtle breathing bounce"),
                    ("Walk South (Sword Ready)", "loop", f"Walking downward toward camera carrying {weapon} safely at the side in rhythmic steps"),
                    ("Walk North (Sword Ready)", "loop", f"Walking upward away from camera holding {weapon} in ready guard"),
                    ("Walk Lateral (East/West)", "loop", f"Striding sideways across the plane carrying {weapon} in combat readiness"),
                    ("Horizontal Blade Slash", "action", f"Steps in and swings {weapon} in a wide 180-degree horizontal cutting arc with a bright blade trail, recovering to sword guard"),
                    ("Spinning Whirlwind Cut", "action", f"Executes a full 360-degree spinning slash with {weapon} creating a circular blade wave, settling cleanly back to guard"),
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
            else:
                raw_moves = [
                    ("Sword Guard Idle", "loop", f"Poised two-handed combat stance gripping {weapon} at the ready, subtle breathing and cape/cloth sway"),
                    ("Combat Advance Walk", "loop", f"Confident forward stride keeping {weapon} raised in defensive guard"),
                    ("Blade Rush Dash", "loop", f"Low-center-of-gravity forward dash with {weapon} poised for a rapid strike"),
                    ("Quick Iaido Slash", "action", f"Deliver a lightning-fast horizontal slash with {weapon} leaving a sharp blade streak, smoothly recovering to sword guard"),
                    ("3-Hit Sword Combo", "action", f"Executes a downward diagonal cut, rising reverse slash, and forward thrust with {weapon}, resetting to idle sword guard"),
                    ("Rising Dragon Cleave", "action", f"Crouches briefly then leaps into an upward arcing slash with {weapon}, landing firmly back in sword guard stance"),
                    ("Crescent Blade Wave", "action", f"Channels energy along {weapon} and unleashes a wide crescent slash wave forward, settling back into resting guard"),
                ]
        elif arch == "heavy_melee":
            raw_moves = [
                ("Heavy Weapon Idle", "loop", f"Grounded powerhouse stance hefting {weapon} across the shoulders with steady rhythmic breathing"),
                ("Lumbering Combat Walk", "loop", f"Heavy intimidating forward stride carrying {weapon} with powerful momentum"),
                ("Bull Charge Rush", "loop", f"Aggressive forward charge leaning in with {weapon} held low to plow through obstacles"),
                ("Wide Horizontal Cleave", "action", f"Swings {weapon} in a massive horizontal arc with heavy follow-through, pulling back into idle stance"),
                ("Crushing Overhead Slam", "action", f"Hoists {weapon} high above the head and slams it down onto the ground plane with impact sparks, recovering to stance"),
                ("2-Hit Berserker Smash", "action", f"Delivers a brutal side swing followed by an upward heave with {weapon}, settling back into heavy guard"),
                ("Earthshaker Spin", "action", f"Uses the momentum of {weapon} for a full 360-degree heavy whirlwind swing, anchoring back into resting stance"),
            ]
        elif arch == "polearm":
            raw_moves = [
                ("Lancer Guard Idle", "loop", f"Poised sentinel stance holding {weapon} diagonally across the body with steady breathing"),
                ("Forward March", "loop", f"Disciplined forward combat walk holding {weapon} at the ready"),
                ("Lunge Dash", "loop", f"Fast forward rush with the tip of {weapon} leveled ahead"),
                ("Lightning Thrust", "action", f"Extends {weapon} in a long-reach piercing forward thrust, retracting crisply back to guard stance"),
                ("Sweeping Halberd Arc", "action", f"Swings {weapon} in a wide low-to-high crescent sweep, recovering smoothly to lancer guard"),
                ("Flurry Impale Combo", "action", f"Executes three rapid pinpoint thrusts with {weapon} followed(by) a spinning shaft flourish, returning to idle"),
                ("Vaulting Sky Dive", "action", f"Poles-vaults upward using {weapon} and strikes downward at an angle before resetting to guard stance"),
            ]
        elif arch == "agile_melee":
            raw_moves = [
                ("Shinobi Crouch Idle", "loop", f"Low agile combat stance holding {weapon} in reverse grip with alert breathing"),
                ("Stealth Stride", "loop", f"Light-footed silent forward walk keeping {weapon} tucked ready to strike"),
                ("Shadow Sprint", "loop", f"High-speed ninja dash leaning low with {weapon} trailing behind"),
                ("Twin-Blade Cross Slash", "action", f"Snaps forward with a rapid X-shaped cross slash using {weapon}, flipping back into crouch idle"),
                ("Acrobatic Flurry Combo", "action", f"Unleashes a 4-hit rapid slash combination with {weapon} ending in a spinning slice, recovering to stance"),
                ("Cartwheel Evasion Strike", "action", f"Performs a swift acrobatic flip while throwing/slashing with {weapon}, landing softly in idle stance"),
                ("Shadow Lotus Finisher", "action", f"Spins in a blur of blade streaks with {weapon} and strikes a poised shinobi hand-seal before returning to idle"),
            ]
        elif arch == "ranged_bow":
            raw_moves = [
                ("Archer Ready Idle", "loop", f"Poised marksman stance holding {weapon} with an arrow at the ready and calm breathing"),
                ("Ranger Stride", "loop", f"Light agile forward walk carrying {weapon} in ready position"),
                ("Evasive Sidestep Dash", "loop", f"Quick lateral ranger dash keeping {weapon} stabilized"),
                ("Quick Snap Shot", "action", f"Draws {weapon} to full tension in a split second and looses a high-velocity arrow, recovering to ready stance"),
                ("Triple Arrow Volley", "action", f"Nocks three arrows at once on {weapon} and fires a spread volley forward, resetting to archer idle"),
                ("Acrobatic Backstep Shot", "action", f"Hops nimble backward while firing an aimed shot from {weapon} mid-air, landing cleanly in stance"),
                ("Charged Piercing Arrow", "action", f"Draws {weapon} back as glowing energy condenses on the arrowhead, releasing a powerful piercing shot and recovering"),
            ]
        elif arch == "magic":
            raw_moves = [
                ("Arcane Channel Idle", "loop", f"Mystical stance holding {weapon} as soft magical energy pulses and robes flutter gently"),
                ("Mage Glide Walk", "loop", f"Graceful forward stride holding {weapon} upright with poise"),
                ("Astral Hover Dash", "loop", f"Levitate-gliding forward dash with {weapon} leading and arcane trail"),
                ("Arcane Bolt Cast", "action", f"Thrusts {weapon} forward to fire a concentrated arcane energy bolt, returning smoothly to caster idle"),
                ("Elemental Sigil Summon", "action", f"Raises {weapon} high and traces a glowing magical rune circle in the air, settling back to resting stance"),
                ("Twin Spell Barrage", "action", f"Swings {weapon} in a figure-eight pattern unleashing dual elemental waves, recovering to idle posture"),
                ("Grand Nova Invocation", "action", f"Channels a brilliant aura around {weapon} and releases an expanding arcane pulse before returning to stance"),
            ]
        elif arch == "ranged_firearm":
            raw_moves = [
                ("Tactical Aim Idle", "loop", f"Armed combat stance holding {weapon} aimed steady with subtle breathing"),
                ("Run-and-Gun Stride", "loop", f"Aggressive forward sprint keeping {weapon} leveled and ready to fire"),
                ("Booster Slide Dash", "loop", f"Low tactical crouch dash with {weapon} stabilized forward"),
                ("Rapid Burst Fire", "action", f"Fires a 3-round energy/muzzle burst from {weapon} with crisp recoil kickback, returning to idle stance"),
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
                ("Primary Gear Strike", "action", f"Executes a signature forward attack using {weapon}, cleanly recovering back to idle stance"),
                ("Combo Technique", "action", f"Performs a fluid multi-part action sequence with {weapon}, resetting smoothly to resting pose"),
                ("Evasive Maneuver", "action", f"Dodges swiftly while guarding with {weapon} and returns to starting stance"),
                ("Signature Special Finisher", "action", f"Unleashes an energized special ability using {weapon} and settles back into idle stance"),
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
                "refined_prompt": compiled_prompt,
                "wan_seed": 42 + idx * 1007,
                "video_path": None,
                "sheet_path": None,
                "json_path": None,
            })
        return moves

    def generate_smart_moveset_with_gemma(
        self,
        character_data: Dict[str, Any],
        genre: str,
        combat_notes: str = "",
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Uses Gemma 4: 31B to intelligently design a complete character-specific move-set and
        pre-animation Wan 2.2 choreography instructions tailored to the character's weapon,
        signature gear, anatomy, and game genre.
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
            "You are a Lead 2D Gameplay Combat Designer and Technical Animator.\n"
            f"Your task is to design a tailored {target_count}-move animation set and pre-animation instructions for this specific character:\n"
            f"- Character Name: {char_name}\n"
            f"- Title / Role: {character_data.get('title') or arch_info['role_title']}\n"
            f"- Visual Appearance: {appearance}\n"
            f"- Equipped Weapon / Signature Gear: {gear}\n"
            f"- Detected Combat Archetype: {arch_info['archetype']} ({arch_info['combat_style_summary']})\n"
            f"- Game Genre & Camera: {genre} ({preset['perspective']}, {preset['camera']})"
            f"{notes_clause}\n\n"
            "CRITICAL CHOREOGRAPHY RULES:\n"
            f"1. WEAPON & CHARACTER CONSISTENCY: {arch_info['forbidden_actions']}\n"
            f"   Every single move MUST make physical and visual sense for {char_name} holding/using '{gear}'. "
            "For example, if the character holds a sword, create sword idle stances, blade dashes, and sword slash combos — NEVER boxing jabs or arm-cannon blasts!\n"
            f"2. MOVE COUNT & MIX: Generate exactly {target_count} moves: include essential locomotion/idle moves (type: 'loop') and signature attacks/skills (type: 'action').\n"
            "3. MOTION LAWS:\n"
            "   - For 'loop' moves: Seamless cycle where t=5.0s matches t=0.0s.\n"
            f"   - For 'action' moves: Starts in the character's idle stance ({arch_info['idle_posture']}), executes the move with '{gear}', and cleanly recovers back to the exact idle stance by t=5.0s.\n"
            "   - Fixed static camera, zero camera pan/zoom, pure flat solid white studio background (#FFFFFF), high-key studio lighting, no shadows.\n\n"
            "Respond ONLY with a valid JSON object following this exact schema:\n"
            "{\n"
            '  "combat_summary": "1-sentence summary of the character\'s tailored animation style",\n'
            '  "moves": [\n'
            "    {\n"
            '      "name": "Move Name (e.g. Katana Guard Idle, 3-Hit Blade Combo)",\n'
            '      "type": "loop or action",\n'
            '      "description": "Clear pre-animation choreography instructions describing how the character and their weapon move",\n'
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
                    moves.append(fb)

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
        micro-choreography prompt with strict camera lockdown and anchor rules.
        """
        char_name = character_data.get("name", "Character")
        genre_info = GENRE_PRESETS.get(genre, GENRE_PRESETS["Custom / Hybrid Genre"])
        arch_info = self.detect_character_combat_archetype(character_data, genre=genre)
        weapon_label = arch_info["weapon_label"]
        is_loop = anim_type.lower() == "loop"

        system_prompt = (
            "You are a Lead Gameplay Animator and Technical Director for 2D arcade video games.\n"
            f"Game Genre: {genre} ({genre_info['perspective']})\n"
            f"Character: {char_name} ({character_data.get('appearance', '')})\n"
            f"Equipped Weapon / Signature Gear: {weapon_label} ({arch_info['combat_style_summary']})\n"
            f"Animation Name: {anim_name}\n"
            f"Animation Type: {'Seamless Looping Cycle' if is_loop else 'Action / Attack with Return-to-Idle Recovery'}\n\n"
            "ANIMATION LAWS:\n"
            f"1. WEAPON & CHARACTER CONSISTENCY: {arch_info['forbidden_actions']} Ensure {char_name} properly holds/uses {weapon_label}.\n"
            f"2. Camera: {genre_info['camera']}. The camera must NEVER pan, tilt, shake, or zoom.\n"
            "3. Background: Must remain fixed pure solid flat white studio background #FFFFFF, high-key studio lighting, zero shadows.\n"
            + (
                f"4. LOOPING RULE: This animation is an endless loop. The character's motion and {weapon_label} position at t=5.0s must seamlessly match their starting pose at t=0.0s."
                if is_loop
                else f"4. RECOVERY RULE: Frame 0 starts in the character's Idle pose ({arch_info['idle_posture']}). The middle delivers the action using {weapon_label}. By t=5.0s, the character MUST cleanly settle back into the exact Idle resting pose."
            )
            + "\n\nRespond ONLY with valid JSON following this exact schema:\n"
            "{\n"
            '  "action_summary": "One sentence summary of the movement incorporating the character\'s weapon/gear",\n'
            '  "choreography_prompt": "Micro-level limb, body, and weapon action breakdown: start posture, acceleration, peak action, recovery",\n'
            '  "wan_prompt": "Complete prompt for Wan 2.2 specifying 2D sprite animation, weapon consistency, fixed static camera, zero camera movement, on pure flat solid white studio background #FFFFFF",\n'
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
        except Exception:
            # Weapon-aware fallback prompt
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
                "action_summary": f"{char_name} performs {anim_name} with {weapon_label}: {raw_description}",
                "choreography_prompt": f"{char_name} ({arch_info['idle_posture']}) executes {anim_name}: {desc_text}.",
                "wan_prompt": wan_prompt,
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
            if master_sprite_path and os.path.exists(str(master_sprite_path)):
                master_sprite_pil = Image.open(master_sprite_path).convert("RGB")
                character_data["master_sprite_pil"] = master_sprite_pil
            else:
                # Auto-synthesize master character sprite on the fly if not yet generated
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
            f"on pure flat solid seamless white studio background #FFFFFF, high-key studio lighting, no floor shadows, clean outline"
        )

        if progress_cb:
            progress_cb(0.15, f"Synthesizing '{anim_name}' with Wan 2.2 (Seed {wan_seed})...")

        # Resize master sprite to Wan 2.2 resolution (480x832)
        resized_sprite = master_sprite_pil.resize((480, 832), Image.Resampling.LANCZOS)
        last_sprite = resized_sprite if not is_loop else None

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

        local_video_path, _ = run_remote_inference(
            resized_image=resized_sprite,
            processed_last_image=last_sprite,
            prompt=full_wan_prompt,
            steps=8,
            negative_prompt=(
                "camera motion, camera pan, camera tilt, zoom, dynamic background, complex background, "
                "scenery, floor shadows, ground clutter, perspective shift, rotation, 3d camera orbit, "
                "dark background, colored background, green screen, chroma green"
            ),
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
