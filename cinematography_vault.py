"""
Cinematography Knowledge Vault & Few-Shot Director Knowledge Base for Wan 2.2 DiT.
Provides battle-tested lens optics, camera rigs, reactive environmental physics,
and period-sequenced prompt blueprints based on state-of-the-art AI video engineering.
"""

from typing import Dict, Any, List, Optional
import re

# ── 1. Gold-Standard Camera Rigs & Lens Optical Presets ─────────────────────
CAMERA_RIG_PRESETS: Dict[str, Dict[str, str]] = {
    "slow push-in": {
        "lens": "50mm prime cinema lens",
        "movement": "Slow steady push-in on a 50mm lens. Shallow depth of field with creamy background bokeh.",
        "shutter": "Natural 180-degree cinematic shutter, zero camera jitter.",
    },
    "slow dolly-in": {
        "lens": "50mm prime cinema lens",
        "movement": "Smooth dolly-in moving gently toward the subject. Foreground parallax with soft bokeh.",
        "shutter": "Fluid camera motion, locked optical axis.",
    },
    "steadicam tracking": {
        "lens": "35mm wide-angle cinema lens",
        "movement": "Steadicam tracking shot moving alongside the subject at waist level. Smooth lateral glide with natural depth parallax.",
        "shutter": "Stable horizon line, organic walking cadence.",
    },
    "orbital tracking shot": {
        "lens": "85mm telephoto portrait lens",
        "movement": "360-degree orbital arc shot on an 85mm lens. Subject remains centered in frame while background glides smoothly past.",
        "shutter": "Continuous circular trajectory, locked subject framing.",
    },
    "crane boom-down": {
        "lens": "28mm wide lens",
        "movement": "High-angle jib crane descending smoothly toward the subject. Expansive environmental perspective settling into a medium shot.",
        "shutter": "Gradual vertical descent, smooth deceleration.",
    },
    "static tripod shot": {
        "lens": "50mm normal lens",
        "movement": "Locked-off camera on a sturdy fluid-head tripod. Zero camera movement or drift. Rigid background architecture and locked spatial geometry.",
        "shutter": "Rock-solid horizon, fixed perspective.",
    },
    "whip-pan tracking": {
        "lens": "35mm action lens",
        "movement": "Fast dynamic tracking shot with slight directional motion blur, locking smoothly onto the subject at destination.",
        "shutter": "Crisp focal lock, controlled directional blur.",
    },
}

# ── 2. Reactive Environmental Physics Rules ──────────────────────────────────
# Video DiTs (Wan, Kling, Hailuo) achieve photorealism when the environment
# reacts with physical weight to character movement instead of remaining static.
REACTIVE_PHYSICS_RULES: List[Dict[str, Any]] = [
    {
        "keywords": ["forest", "wood", "tree", "leaf", "leaves", "grass", "meadow", "grove", "jungle"],
        "physics": "Leaves rustle and dry pine needles scatter underfoot. Ambient ferns and canopy foliage sway gently in the breeze.",
    },
    {
        "keywords": ["dust", "path", "road", "dirt", "ground", "step", "walk", "run", "jump", "land"],
        "physics": "Footsteps strike the ground with physical weight, kicking up faint dust particles caught in the rim lighting.",
    },
    {
        "keywords": ["water", "river", "stream", "lake", "ocean", "rain", "splash", "pond", "puddle"],
        "physics": "Water surface ripples smoothly with realistic surface tension and light refraction. Ambient droplets scatter with natural gravity.",
    },
    {
        "keywords": ["magic", "glow", "spell", "light", "energy", "star", "spark", "flame", "crystal"],
        "physics": "Luminous energy casts dynamic sweeping light across nearby surfaces with soft specular highlights and shifting cast shadows.",
    },
    {
        "keywords": ["snow", "ice", "winter", "frost", "cold"],
        "physics": "Crisp snow crunches underfoot leaving soft impressions. Fine powder drifts in the cold air currents.",
    },
    {
        "keywords": ["room", "indoor", "interior", "kitchen", "bakery", "library", "house", "cabin"],
        "physics": "Dust motes drift slowly through soft volumetric window light. Ambient curtains and hanging props flutter subtly.",
    },
]

# ── 3. Structural Invariant Shields (Prevents Latent Melt & Warp) ─────────────
DEFAULT_INVARIANT_SHIELD = (
    "Rigid background geometry, continuous spatial perspective, locked lighting, zero camera jitter, no morphing"
)

# ── 4. Few-Shot Gold Standard Director Blueprints for Gemma ─────────────────
FEW_SHOT_DIRECTOR_EXAMPLES = [
    {
        "scenario": "Barnaby walks through the whispering pine forest and spots a glowing berry",
        "director_output": {
            "shot_type": "medium tracking shot",
            "camera_movement": "Steadicam tracking shot on a 35mm lens, smooth lateral glide",
            "motion_intensity": 2,
            "subject_action": "Barnaby walks forward along the forest trail and halts with surprised curiosity",
            "timeline": [
                "Barnaby steps forward along the pine-needle trail with steady momentum.",
                "Barnaby notices the glowing light and halts his stride, leaning forward to inspect it.",
                "Barnaby settles on both feet, smiling with bright inquisitive eyes."
            ],
            "environment_motion": "Dry pine needles scatter underfoot as ferns sway in the morning breeze.",
            "identity_anchors": "Plump brown bear, red bowtie, warm fur texture",
            "wan_prompt": (
                "Steadicam tracking shot on a 35mm lens. "
                "Barnaby steps forward along the pine-needle trail with steady momentum. "
                "Barnaby halts his stride and leans forward to inspect the glowing light. "
                "Barnaby settles on both feet with an inquisitive smile. "
                "Dry pine needles scatter underfoot as ambient ferns sway gently. "
                "Rigid forest geometry, locked lighting, no face morphing."
            ),
        },
    },
    {
        "scenario": "Knight and Mage confront the shadow guardian at the castle gate",
        "director_output": {
            "shot_type": "low-angle two-shot",
            "camera_movement": "Slow steady push-in on a 50mm lens with shallow depth of field",
            "motion_intensity": 3,
            "subject_action": "Knight draws longsword while Mage raises glowing staff in battle readiness",
            "timeline": [
                "Knight steps his right boot forward into a deep stance; Mage grips crystal staff with two hands.",
                "Knight draws steel sword with a fluid sweep; Mage's staff flares with pulsing blue light.",
                "Both heroes lock into resolved combat readiness, facing forward."
            ],
            "environment_motion": "Faint stone dust puffs outward from boot impact; azure spell light casts dynamic shadows on stone arches.",
            "identity_anchors": "Knight polished silver armor, Mage cobalt blue hooded robe",
            "wan_prompt": (
                "Slow steady push-in on a 50mm lens. "
                "Knight steps into a deep stance as Mage grips crystal staff with two hands. "
                "Knight draws steel sword with a fluid sweep while Mage's staff flares with azure light. "
                "Both heroes lock into resolved combat readiness. "
                "Faint dust puffs from boot impact and dynamic light sweeps across stone arches. "
                "Rigid architectural geometry, zero camera jitter, no morphing."
            ),
        },
    },
]


def retrieve_cinematography_blueprint(
    scenario: str = "",
    camera_movement: str = "",
    art_style: str = "",
) -> Dict[str, Any]:
    """
    Semantic & rule-based retriever that selects the best camera rig,
    reactive environmental physics clause, and invariant shield for any scene.
    """
    text_corpus = f"{scenario} {camera_movement} {art_style}".lower()

    # 1. Match Camera Rig
    matched_rig = CAMERA_RIG_PRESETS["slow push-in"]  # default cinema move
    cam_lower = (camera_movement or "").lower()
    sorted_rigs = sorted(CAMERA_RIG_PRESETS.items(), key=lambda x: len(x[0]), reverse=True)
    found = False
    for rig_key, rig_val in sorted_rigs:
        if rig_key in cam_lower:
            matched_rig = rig_val
            found = True
            break
    if not found:
        for rig_key, rig_val in sorted_rigs:
            if rig_key in text_corpus:
                matched_rig = rig_val
                found = True
                break
    if not found:
        for rig_key, rig_val in sorted_rigs:
            if any(w in cam_lower for w in rig_key.split() if len(w) > 4):
                matched_rig = rig_val
                break

    # 2. Match Environmental Physics
    matched_physics = []
    for rule in REACTIVE_PHYSICS_RULES:
        if any(re.search(rf"\b{kw}\b", text_corpus) for kw in rule["keywords"]):
            matched_physics.append(rule["physics"])
            if len(matched_physics) >= 2:
                break
    physics_clause = " ".join(matched_physics) if matched_physics else (
        "Ambient light shifts subtly across surfaces with natural atmospheric weight."
    )

    return {
        "lens": matched_rig["lens"],
        "camera_movement_directive": matched_rig["movement"],
        "shutter_directive": matched_rig["shutter"],
        "reactive_physics": physics_clause,
        "invariant_shield": DEFAULT_INVARIANT_SHIELD,
    }


def format_period_sequenced_wan_prompt(
    camera_directive: str,
    timeline_steps: List[str],
    environment_physics: str,
    identity_anchors: str,
    art_style: str = "",
) -> str:
    """
    Synthesizes a strict period-sequenced Wan 2.2 FLF2V prompt where each phase
    is an independent sentence to prevent token pooling and multi-limb warping.
    """
    sentences: List[str] = []

    # Sentence 1: Camera & Optic Baseline
    cam_clean = (camera_directive or "").strip().rstrip(".")
    if cam_clean:
        sentences.append(f"{cam_clean}.")

    # Sentences 2-4: Chronological Action Timeline
    for step in (timeline_steps or [])[:3]:
        s_clean = (step or "").strip().rstrip(".")
        if s_clean:
            sentences.append(f"{s_clean}.")

    # Sentence 5: Reactive Environmental Physics
    env_clean = (environment_physics or "").strip().rstrip(".")
    if env_clean:
        sentences.append(f"{env_clean}.")

    # Sentence 6: Identity Anchors & Structural Shield
    id_clean = (identity_anchors or "").strip().rstrip(".")
    if id_clean:
        sentences.append(f"{id_clean}, {DEFAULT_INVARIANT_SHIELD}.")
    else:
        sentences.append(f"{DEFAULT_INVARIANT_SHIELD}.")

    if art_style:
        sentences.append(f"{art_style.strip().rstrip('.')}, smooth fluid animation, 4k.")

    return " ".join(sentences)
