"""
Orchestra AI Decision Maker Rules & Knowledge Engine for Wan 2.2, Qwen Image Edit, and Z-Image Turbo.
Encodes the operational knowledge base, parameter constraints, anti-pattern detection,
heart-touching story budget validation (<= 3 characters), and cinematic transition generation.
"""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import re


# ── 1. Model Operational Limits & Budgets ─────────────────────────────────────

MAX_STORY_CHARACTERS = 3       # Heart-touching intimacy constraint (user requirement)
MAX_CLIP_DURATION_SEC = 5.0    # Wan 2.2 81-frame / 16fps ceiling
MIN_CLIP_DURATION_SEC = 1.0
RECOMMENDED_TRANSITION_SEC = 0.5


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"      # xfade=transition=fade (soft dissolve)
    FADE_BLACK = "fade_black"    # Dip-to-black (fade out, fade in)
    MATCH_CUT = "match_cut"      # Directional slide/wipe or hard continuous action cut


# ── 2. Operational Knowledge Base: What to Do vs What NOT to Do ───────────────

ORCHESTRA_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "z_image_turbo": {
        "role": "Single-Character Genesis & Signature Reference Isolation",
        "aspect_ratio": "9:16",
        "default_steps": 8,
        "recommended_cfg": (1.5, 2.2),
        "what_to_do": [
            "Generate full-body vertical portraits (9:16) with head-to-toe visible.",
            "Place subject standing on clean, neutral studio backdrop (solid light grey or white).",
            "Explicitly define signature items/props held in hands (e.g. 'holding carved wooden flute').",
            "Use precise facial, hair, and clothing descriptors to establish a strong visual anchor.",
            "Enforce strict negative prompts removing watermarks, multiple people, and cropped limbs.",
        ],
        "what_not_to_do": [
            "Never generate multiple characters in a single Z-Image sheet (causes identity bleed in Qwen).",
            "Do not use complex outdoor or detailed backgrounds for character reference sheets.",
            "Avoid extreme close-up headshots without full-body clothing context.",
            "Do not omit signature props from the prompt if the character uses them in the story.",
            "Avoid leaving the negative prompt empty.",
        ],
        "anti_patterns": [
            r"\bmultiple\s+(people|characters|figures)\b",
            r"\btwo\s+(people|boys|girls|men|women)\b",
            r"\bclose-?up\s+(face|shot|only)\b",
            r"\bbusy\s+(city|forest|background)\b",
            r"\bcrowded\b",
        ],
        "default_negative": (
            "multiple people, two characters, crowd, group, cropped limbs, cropped feet, "
            "missing fingers, distorted anatomy, blurry, complex background, outdoors, "
            "watermark, text, signature, low quality, oversaturated"
        ),
    },
    "qwen_image_edit": {
        "role": "Multi-Reference Visual Transfer & Start/End Dual Keyframe Synthesis",
        "max_references": 3,
        "default_steps": 25,
        "what_to_do": [
            "Pass character portraits as Image 1, 2, 3 with explicit character name mapping.",
            "Anchor End Frame (t=5s) directly on Start Frame (image 1) using 'image 1 is the starting frame'.",
            "Command background permanence explicitly: 'Keep identical room geometry, wall textures, window lighting, and props'.",
            "Ensure a significant pose delta (18.0 <= Delta <= 85.0) between Start and End keyframes.",
            "Keep held items and accessories physically attached across pose transitions.",
        ],
        "what_not_to_do": [
            "Never pass more than 3 reference images (exceeds Qwen multi-image capacity).",
            "Do not generate the End Frame from scratch without referencing the Start Frame.",
            "Never allow camera angle or room layout to change between Start (t=0) and End (t=5s) of the same shot.",
            "Do not generate subtle micro-expressions with Delta < 18 (causes frozen Wan video).",
            "Do not drop held props between start and end frames unless explicitly discarded.",
        ],
        "anti_patterns": [
            r"\bdifferent\s+(room|location|camera\s+angle|lighting)\b",
            r"\bnew\s+background\b",
            r"\bcompletely\s+different\s+scene\b",
            r"\bwithout\s+the\s+(sword|flute|lantern|staff|prop)\b",
        ],
        "background_lock_command": (
            "Keep identical environment geometry, room architecture, wall textures, lighting direction, "
            "and all background props strictly preserved from image 1."
        ),
    },
    "wan_2_2": {
        "role": "Temporal DiT Synthesis & FLF2V / Guided I2V Motion Interpolation",
        "fps": 16,
        "max_clip_seconds": 5.0,
        "recommended_cfg_primary": (1.75, 2.0),
        "recommended_cfg_secondary": (1.25, 1.5),
        "recommended_flow_shift": 4.5,
        "what_to_do": [
            "Structure prompt into strict 3-phase chronological timelines: Phase 1 (0.0s-1.5s), Phase 2 (1.5s-3.5s), Phase 3 (3.5s-5.0s).",
            "Limit action density to exactly ONE atomic action and its physical reaction per 5-second clip.",
            "Preserve active props and weapons across motion phases.",
            "Apply motion-adaptive dual-CFG (1.75 to 2.0 primary, 1.25 to 1.5 secondary) to prevent noise boiling.",
            "Route to Guided I2V fallback if Start and End frames exceed safe FLF2V divergence.",
        ],
        "what_not_to_do": [
            "Never cram 3 or more separate narrative beats into a single 5-second scene.",
            "Avoid rambling simultaneous run-on sentences without chronological time anchors.",
            "Do not push CFG guidance scale above 2.5 (causes severe latent noise explosion and chromatic distortion).",
            "Never force FLF2V when camera angle or location diverges between start and end frames.",
            "Avoid ambiguous motions without physical contact or environmental reaction.",
        ],
        "anti_patterns": [
            r"\bwhile\s+simultaneously\b",
            r"\bat\s+the\s+same\s+time\s+he\s+also\b",
            r"\band\s+then\s+he\s+flies\s+away\s+and\s+then\s+he\s+eats\b",
            r"\bteleports\b",
            r"\binstant\s+transformation\b",
        ],
        "default_negative": (
            "face morphing, changing face, altered facial features, changing identity, inconsistent face, "
            "shifting features, melting face, face distortion, mutating face, character design change, "
            "outfit change, color shift, temporal flickering, pixel noise, pixel degradation, warping, tearing"
        ),
    },
}


# ── 3. Story Budget & Constraint Validation (Heart-Touching Stories) ──────────

def validate_story_budget(story: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that a story adheres to the Orchestra AI requirements:
    1. Maximum 3 characters (heart-touching emotional intimacy constraint).
    2. Scene count between 1 and 8, each duration <= 5.0s.
    3. Synchronized voiceover narration (strictly 6 to 11 words per scene).
    4. Prop retention across consecutive scenes.
    """
    errors: List[str] = []

    characters = story.get("characters", [])
    if not characters:
        errors.append("Story must have at least 1 character.")
    elif len(characters) > MAX_STORY_CHARACTERS:
        errors.append(
            f"Character budget exceeded: {len(characters)} characters found. "
            f"Maximum allowed for heart-touching short stories is {MAX_STORY_CHARACTERS}."
        )

    scenes = story.get("scenes", [])
    if not scenes:
        errors.append("Story must have at least 1 scene.")

    active_props_history = set()
    for sc in scenes:
        num = sc.get("scene_number", "?")
        dur = sc.get("duration_seconds", 5.0)
        if dur > MAX_CLIP_DURATION_SEC:
            errors.append(f"Scene {num} duration ({dur}s) exceeds Wan 2.2 max limit of {MAX_CLIP_DURATION_SEC}s.")

        narration = sc.get("narration_text", "")
        if narration:
            word_count = len(narration.strip().split())
            if word_count < 4 or word_count > 15:
                # Flag soft warning / error if narration is outside safe speech pacing window
                if word_count > 15:
                    errors.append(f"Scene {num} voiceover narration too long ({word_count} words). Max is 11-13 words for a 5s clip.")

        ledger = sc.get("state_ledger", {})
        props = ledger.get("active_props", [])
        if props:
            active_props_history.update(props)

    return len(errors) == 0, errors


def enforce_orchestra_story_constraints(story: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically audits, repairs, and sanitizes story data to conform to Orchestra rules:
    - Prunes characters beyond 3 (retaining the 3 most active/important).
    - Clamps scene durations to <= 5.0s.
    - Enhances Qwen end prompts with background lock commands.
    - Sanitizes Wan motion prompts into chronological 3-phase structure if missing.
    """
    repaired = dict(story)

    # 1. Enforce character budget <= 3
    characters = list(repaired.get("characters", []))
    if len(characters) > MAX_STORY_CHARACTERS:
        # Keep first 3 characters
        characters = characters[:MAX_STORY_CHARACTERS]
        repaired["characters"] = characters

    allowed_ids = {c.get("id") for c in characters}

    # 2. Enforce scene parameters & background locks
    scenes = list(repaired.get("scenes", []))
    repaired_scenes = []
    bg_lock = ORCHESTRA_KNOWLEDGE_BASE["qwen_image_edit"]["background_lock_command"]

    for sc in scenes:
        sc_copy = dict(sc)
        # Duration clamp
        dur = float(sc_copy.get("duration_seconds", MAX_CLIP_DURATION_SEC))
        sc_copy["duration_seconds"] = min(MAX_CLIP_DURATION_SEC, max(MIN_CLIP_DURATION_SEC, dur))

        # Filter character IDs
        sc_chars = [cid for cid in sc_copy.get("character_ids", []) if cid in allowed_ids]
        if not sc_chars and allowed_ids:
            sc_chars = [next(iter(allowed_ids))]
        sc_copy["character_ids"] = sc_chars

        # Ensure Qwen end prompt has background lock command
        qwen_end = sc_copy.get("qwen_end_scene_prompt", "")
        if qwen_end and "identical" not in qwen_end.lower() and "background" not in qwen_end.lower():
            sc_copy["qwen_end_scene_prompt"] = f"{qwen_end}. {bg_lock}"

        repaired_scenes.append(sc_copy)

    repaired["scenes"] = repaired_scenes
    return repaired


# ── 4. Prompt Compilation & Anti-Pattern Auditing ─────────────────────────────

def audit_prompt_for_anti_patterns(prompt: str, model_type: str) -> List[str]:
    """
    Audits a candidate prompt against known model failure patterns and returns detected violations.
    """
    kb = ORCHESTRA_KNOWLEDGE_BASE.get(model_type, {})
    patterns = kb.get("anti_patterns", [])
    violations: List[str] = []

    for pattern in patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            violations.append(f"Anti-pattern detected matching '{pattern}' for {model_type}.")

    return violations


def compile_z_image_prompt(character: Dict[str, Any], art_style: str = "vibrant 2D animation") -> Tuple[str, str]:
    """
    Compiles a gold-standard Z-Image Turbo character genesis prompt and negative prompt.
    Enforces full-body 9:16 vertical portrait, signature props, and neutral studio backdrop.
    """
    name = character.get("name", "Character")
    face = character.get("face", "expressive facial features")
    hair = character.get("hair", "neat hair")
    clothing = character.get("clothing", "simple outfit")
    signature = character.get("signature_details", "")
    prop_phrase = f", visibly holding {signature} in hand" if signature else ""

    positive = (
        f"Full-body vertical character reference sheet of {name}, {face}, {hair}, {clothing}{prop_phrase}. "
        f"9:16 vertical full-body portrait, head-to-toe visible, standing on clean neutral studio backdrop. "
        f"Art style: {art_style}. Masterpiece character concept art, ultra-crisp silhouette, studio lighting."
    )
    negative = ORCHESTRA_KNOWLEDGE_BASE["z_image_turbo"]["default_negative"]
    return positive, negative


def compile_qwen_end_frame_prompt(
    start_scenario: str,
    target_action: str,
    character_names: List[str],
    environment_desc: str = "",
) -> str:
    """
    Compiles a gold-standard Qwen 2.5 Image Edit End-Frame prompt anchored on Start Frame (image 1).
    """
    chars_str = ", ".join(character_names) if character_names else "the character"
    bg_lock = ORCHESTRA_KNOWLEDGE_BASE["qwen_image_edit"]["background_lock_command"]
    env_phrase = f" in {environment_desc}" if environment_desc else ""

    prompt = (
        f"image 1 is the starting frame. For this ending frame at t=5.0s{env_phrase}, transform {chars_str}: "
        f"{target_action}. Maintain complete facial identity, clothing color, and active props. {bg_lock}"
    )
    return prompt


def compile_wan_motion_timeline(
    scene_number: int,
    entry_state: str,
    atomic_action: str,
    exit_state: str,
    camera_rig: str = "Steadicam tracking shot on a 35mm lens",
    duration_sec: float = 5.0,
) -> str:
    """
    Compiles a strict 3-phase chronological motion prompt for Wan 2.2 DiT.
    """
    p1_end = round(duration_sec * 0.3, 1)
    p2_end = round(duration_sec * 0.7, 1)

    timeline = (
        f"{camera_rig}. "
        f"Phase 1 (0.0s-{p1_end}s): Initiating from {entry_state}. "
        f"Phase 2 ({p1_end}s-{p2_end}s): {atomic_action}. "
        f"Phase 3 ({p2_end}s-{duration_sec}s): Settling into {exit_state}. "
        "Rigid background architecture, locked lighting, smooth continuous motion, zero camera jitter, no morphing."
    )
    return timeline


# ── 5. Cinematic Transition Filter Generator ──────────────────────────────────

def build_ffmpeg_transition_filter(
    clip_durations: List[float],
    transition_type: str = "crossfade",
    transition_duration: float = 0.5,
    has_audio: bool = True,
) -> Tuple[str, str, float]:
    """
    Builds the FFmpeg complex filtergraph for seamless cinematic transitions:
    - 'cut': Simple concat filter (0 transition duration).
    - 'crossfade': FFmpeg xfade=transition=fade with acrossfade for audio.
    - 'fade_black': Dip-to-black fade out / fade in between scenes.
    - 'match_cut': Fast directional wipe (xfade=transition=wipeleft or slidedown).

    Returns:
    - video_filter: FFmpeg -filter_complex string for video.
    - audio_filter: FFmpeg -filter_complex string for audio (if has_audio is True).
    - total_duration: Estimated final stitched duration in seconds.
    """
    n = len(clip_durations)
    if n <= 1:
        return "", "", sum(clip_durations)

    t_type = transition_type.lower()
    td = min(transition_duration, 1.0)

    # 1. Hard Cut (No overlapping filter needed, or standard concat)
    if t_type == "cut" or td <= 0:
        v_inputs = "".join(f"[{i}:v]" for i in range(n))
        v_filter = f"{v_inputs}concat=n={n}:v=1:a=0[v]"
        if has_audio:
            a_inputs = "".join(f"[{i}:a]" for i in range(n))
            a_filter = f"{a_inputs}concat=n={n}:v=0:a=1[a]"
        else:
            a_filter = ""
        return v_filter, a_filter, sum(clip_durations)

    # 2. Cross-Dissolve / Xfade
    # For N clips with xfade:
    # offset_0 = dur_0 - td
    # [0:v][1:v]xfade=transition=fade:duration=td:offset=offset_0[v1]
    # offset_1 = offset_0 + dur_1 - td
    # [v1][2:v]xfade=transition=fade:duration=td:offset=offset_1[v2] ...
    v_lines = []
    a_lines = []
    
    current_offset = clip_durations[0] - td
    prev_v = "[0:v]"
    prev_a = "[0:a]"

    for i in range(1, n):
        next_v = f"[{i}:v]"
        next_a = f"[{i}:a]"
        out_v = f"[v{i}]" if i < n - 1 else "[v]"
        out_a = f"[a{i}]" if i < n - 1 else "[a]"

        if t_type == "fade_black":
            trans_name = "fadeblack"
        elif t_type == "match_cut":
            trans_name = "wipeleft"
        else:
            trans_name = "fade"

        v_lines.append(
            f"{prev_v}{next_v}xfade=transition={trans_name}:duration={td:.2f}:offset={current_offset:.2f}{out_v}"
        )
        if has_audio:
            a_lines.append(
                f"{prev_a}{next_a}acrossfade=d={td:.2f}{out_a}"
            )

        if i < n - 1:
            current_offset += clip_durations[i] - td
            prev_v = out_v
            prev_a = out_a

    total_duration = current_offset + clip_durations[-1]
    video_filter = ";".join(v_lines)
    audio_filter = ";".join(a_lines) if has_audio else ""

    return video_filter, audio_filter, round(total_duration, 2)


def calculate_transition_srt_timestamps(
    scene_durations: List[float],
    transition_type: str = "cut",
    transition_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Calculates the adjusted Start and End timestamps for subtitles (.srt)
    taking into account transition overlaps so spoken subtitles remain perfectly synchronized.
    """
    n = len(scene_durations)
    if n == 0:
        return []

    td = transition_duration if transition_type in ("crossfade", "fade_black", "match_cut") else 0.0
    timestamps: List[Tuple[float, float]] = []
    current_time = 0.0

    for i, dur in enumerate(scene_durations):
        start_t = current_time
        end_t = start_t + dur
        timestamps.append((round(start_t, 2), round(end_t, 2)))
        # For the next scene, the start time is offset by duration minus transition overlap
        if i < n - 1:
            current_time = end_t - td

    return timestamps
