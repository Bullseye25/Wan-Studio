"""
Cinematography Knowledge Vault & Few-Shot Director Knowledge Base for Wan 2.2 DiT.
Provides battle-tested lens optics, camera rigs, reactive environmental physics,
and period-sequenced prompt blueprints based on state-of-the-art AI video engineering.
"""

from typing import Dict, Any, List, Optional
import re
import json
import os
import pathlib

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


# ── 5. Causal Narrative Beats & Transition Connectors ─────────────────────────
# Enforces the Pixar / South Park storytelling rule: Scenes must connect via
# "THEREFORE" (cause-and-effect reaction) or "BUT" (conflict/obstacle), NEVER "AND THEN".
NARRATIVE_BEATS_BY_COUNT: Dict[int, List[Dict[str, str]]] = {
    2: [
        {"beat": "Setup & Goal", "connector": "SETUP", "role": "Establish protagonist, desire, and immediate intent"},
        {"beat": "Climax & Resolution", "connector": "THEREFORE", "role": "Direct causal reaction achieving or altering the goal"},
    ],
    3: [
        {"beat": "Hook & Status Quo", "connector": "SETUP", "role": "Establish protagonist, environment, and initial physical action"},
        {"beat": "Inciting Obstacle", "connector": "BUT", "role": "Sudden complication interrupts the action and forces a tactical response"},
        {"beat": "Climax & Payoff", "connector": "THEREFORE", "role": "Decisive physical solution resolving the conflict into a new state"},
    ],
    4: [
        {"beat": "Hook & Status Quo", "connector": "SETUP", "role": "Establish protagonist, world, and baseline mission"},
        {"beat": "Inciting Incident", "connector": "BUT", "role": "A sharp disruption challenges the protagonist"},
        {"beat": "Escalation & Attempt", "connector": "THEREFORE", "role": "Protagonist commits to physical action to counter the disruption"},
        {"beat": "Climax & Resolution", "connector": "THEREFORE", "role": "Direct payoff resulting in emotional or physical transformation"},
    ],
    5: [
        {"beat": "Status Quo & Desire", "connector": "SETUP", "role": "Protagonist in familiar routine with a specific yearning"},
        {"beat": "Inciting Incident", "connector": "BUT", "role": "Sudden obstacle breaks the routine"},
        {"beat": "Failed Attempt / Escalation", "connector": "THEREFORE", "role": "Action taken to solve the problem backfires or raises stakes"},
        {"beat": "Peak Climax", "connector": "BUT", "role": "Unexpected turning point demanding maximum effort or courage"},
        {"beat": "New Equilibrium & Payoff", "connector": "THEREFORE", "role": "Lasting consequence; protagonist settles in altered world"},
    ],
    6: [
        {"beat": "Ordinary World", "connector": "SETUP", "role": "Introduce hero and normal environment"},
        {"beat": "Catalyst / Call", "connector": "BUT", "role": "Unforeseen event triggers urgency"},
        {"beat": "Commitment / Action", "connector": "THEREFORE", "role": "Hero steps into danger with specific gear"},
        {"beat": "Midpoint Crisis", "connector": "BUT", "role": "Plan falls apart or surprise threat attacks"},
        {"beat": "Decisive Climax", "connector": "THEREFORE", "role": "Hero unleashes peak skill or sacrifice"},
        {"beat": "Resolution", "connector": "THEREFORE", "role": "Final lingering consequence and changed reality"},
    ],
}


# ── 6. Gold-Standard Multi-Scene Causal Story Blueprints ──────────────────────
# Curated multi-scene training exemplars showing strict State Ledgers, 5s feasibility,
# prop permanence, and 180-degree camera/screen direction consistency.
CAUSAL_NARRATIVE_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "title": "The Emberblade Trial",
        "genre": "Fantasy / Action",
        "art_style": "High-end cinematic anime, Studio Ufotable aesthetic, crisp ink lines, volumetric lighting",
        "theme": "Courage under pressure transforms weakness into mastery",
        "premise": "A young warrior with a chipped fiery katana faces an ancient stone sentinel inside a cavern.",
        "characters": [
            {
                "id": "char_1",
                "name": "Ren",
                "face": "Determined amber eyes, scarred cheek, gritted teeth",
                "hair": "Spiky jet-black hair tied in a short low ponytail",
                "clothing": "Weathered crimson haori over dark hakama pants, straw sandals, wrapped forearms",
                "signature_details": "Chipped fiery steel katana in right hand, brass tsuba",
                "z_image_prompt": "Full-body character model sheet of Ren, young anime samurai with spiky jet-black ponytail, crimson haori, holding chipped fiery katana in right hand, 9:16 vertical full-body portrait, front view, neutral background",
            }
        ],
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat": "Hook & Status Quo",
                "causal_connector": "SETUP",
                "narration_text": "Deep inside the crystal cavern, Ren readies his fiery blade.",
                "narration_voice_style": "tense_whisper",
                "sync_trigger_word": "readies",
                "state_ledger": {
                    "entry_state": "Ren stands low in defensive stance at cavern entrance, katana gripped two-handed",
                    "active_props": ["chipped fiery katana (two-handed grip)"],
                    "spatial_zone": "Cavern entrance, screen-left facing screen-right",
                    "atomic_action_5s": "Ren spots the towering stone sentinel awaken -> drops into a sprint toward it, sparks flying from blade",
                    "exit_state": "Ren mid-sprint, blade trailing embers behind him, leaping forward",
                },
                "qwen_scene_prompt": "Ren the young anime samurai in crimson haori stands in a low defensive stance holding his chipped katana with both hands at the entrance of a crystalline cavern, facing screen-right. In the background a colossal stone sentinel awakens with glowing runic eyes. Volumetric cave light, cinematic framing.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, completely transform Ren's body: Ren is now mid-sprint leaping forward toward screen-right, his chipped katana swept back trailing bright fiery embers, teeth gritted in fierce determination. Keep identical crystal cavern walls and glowing runes.",
                "wan_motion_prompt": "Steadicam tracking shot on a 35mm lens. Ren launches forward from his defensive stance into an aggressive low sprint. Ren sweeps his glowing katana back as orange sparks scatter behind him. Ren leaps upward toward the sentinel. Stone dust puffs under his sandals and glowing embers cast dynamic light across crystal walls. Rigid cavern architecture, locked lighting, no morphing.",
                "camera_movement": "Steadicam tracking shot on a 35mm lens, lateral glide left to right",
            },
            {
                "scene_number": 2,
                "narrative_beat": "Inciting Obstacle",
                "causal_connector": "BUT — Sentinel slams its fist into the ground, rupturing the rock under Ren's leap",
                "narration_text": "But the ancient giant slams down, shattering the stone beneath him.",
                "narration_voice_style": "urgent_rising",
                "sync_trigger_word": "slams",
                "state_ledger": {
                    "entry_state": "Ren airborne mid-leap with katana raised, sentinel fist descending",
                    "active_props": ["chipped fiery katana (right hand)"],
                    "spatial_zone": "Center cavern floor, screen-left to screen-right alignment preserved",
                    "atomic_action_5s": "Shockwave knocks Ren off balance -> Ren lands hard on one knee, plunging blade into cracked rock to brake",
                    "exit_state": "Ren crouching on one knee, blade buried in fissure, shielding face with left arm from flying rubble",
                },
                "qwen_scene_prompt": "Ren the young samurai airborne with his glowing katana raised above his head as the stone sentinel's massive fist crashes into the cavern floor directly beneath him. Shattered rock fragments explode into the air. Screen-left to screen-right orientation.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Ren: Ren has absorbed the shockwave and crouches low on one knee, his chipped katana plunged deep into a rock fissure acting as a brake, his left forearm raised shielding his face from flying debris. Ambient rock fragments suspended in air.",
                "wan_motion_prompt": "Dynamic push-in on a 50mm action lens. Ren is jolted by the ground shockwave mid-air. Ren lands heavily on one knee and drives his blade into the stone floor to arrest momentum. Ren raises his left forearm to shield against rock fragments. Shattered stone chunks bounce across the ground as subterranean light flickers. Rigid cavern geometry, zero camera jitter, no morphing.",
                "camera_movement": "Dynamic push-in on a 50mm action lens, rapid deceleration",
            },
            {
                "scene_number": 3,
                "narrative_beat": "Climax & Payoff",
                "causal_connector": "THEREFORE — Ren channels the fissure's molten heat into his blade for an explosive rising strike",
                "narration_text": "Drawing molten fury from the earth, Ren cleaves the colossus apart!",
                "narration_voice_style": "epic_climax",
                "sync_trigger_word": "cleaves",
                "state_ledger": {
                    "entry_state": "Ren crouching on one knee with blade in rock fissure, glowing molten fissures underneath",
                    "active_props": ["flaming katana fully energized (two-handed grip)"],
                    "spatial_zone": "Directly before sentinel torso, facing screen-right",
                    "atomic_action_5s": "Ren pulls blade from fissure overflowing with magma -> executes a rising vertical spiral slash cleaving sentinel core",
                    "exit_state": "Ren lands behind sentinel in settled sheath follow-through pose as sentinel splits in two with blinding light",
                },
                "qwen_scene_prompt": "Ren crouching on one knee with his katana blade embedded in a glowing orange fissure, magma light reflecting off his crimson haori, facing screen-right. The towering sentinel looms above him preparing a second blow.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Ren: Ren has completed a soaring rising slash and lands softly in a low settled stance behind the sentinel with katana swept horizontally, blade glowing incandescent white, while the stone sentinel behind him is cleanly cleaved in two with brilliant light beaming from its core.",
                "wan_motion_prompt": "Low-angle crane boom-up on a 28mm wide lens. Ren wrenches the fiery katana from the stone fissure and surges upward. Ren executes a vertical spiral slash that cleaves cleanly through the sentinel's runic core. Ren lands smoothly in a grounded follow-through pose as the sentinel cracks in blinding azure light. Magma sparks shower the cavern floor and smoke billows. Rigid background architecture, locked lighting, no morphing.",
                "camera_movement": "Low-angle crane boom-up on a 28mm wide lens",
            },
        ],
    },
    {
        "title": "The Clockwork Aviator",
        "genre": "Steampunk / Heartwarming",
        "art_style": "Pixar 3D animated style, rich subsurface scattering, warm brass and copper textures",
        "theme": "Patience and trust give flight to fragile dreams",
        "premise": "A young apprentice clockmaker in an attic workshop tries to get a brass mechanical songbird to fly for the first time.",
        "characters": [
            {
                "id": "char_1",
                "name": "Milo",
                "face": "Round freckled cheeks, wide hopeful hazel eyes, leather goggles on forehead",
                "hair": "Messy curly auburn hair with a bronze gear hairpin",
                "clothing": "Leather craftsman apron over rolled-up cream linen shirt, brown corduroy trousers",
                "signature_details": "Brass mechanical songbird with ruby eyes held in palms",
                "z_image_prompt": "Full-body character model sheet of Milo, young steampunk apprentice with leather apron and goggles, holding a small brass mechanical songbird, 9:16 vertical portrait, warm workshop background, Pixar style",
            }
        ],
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat": "Hook & Status Quo",
                "causal_connector": "SETUP",
                "narration_text": "With one gentle turn, Milo awakens the tiny brass songbird.",
                "narration_voice_style": "warm_wonder",
                "sync_trigger_word": "awakens",
                "state_ledger": {
                    "entry_state": "Milo leans over cluttered wooden workbench holding mechanical bird in two cupped hands",
                    "active_props": ["brass mechanical songbird (two hands)", "brass winding key"],
                    "spatial_zone": "Workbench left side, screen-left facing screen-right toward open attic window",
                    "atomic_action_5s": "Milo turns the winding key with trembling fingers -> bird's clockwork wings flutter tentatively",
                    "exit_state": "Milo holds hands open, bird standing on palms with wings spread open, gears clicking softly",
                },
                "qwen_scene_prompt": "Milo the young steampunk apprentice in leather apron stands at a cluttered workbench in an attic workshop, cupping a delicate brass mechanical bird in both hands. Golden afternoon sun streams through an open circular window on the right. Gears and tools litter the desk.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Milo: Milo has opened both hands flat, breathless with excitement, as the brass bird stands proud on his open palms with its gossamer filigree wings spread wide and ruby eyes gleaming. Keep identical attic room, sunlight rays, and workbench clutter.",
                "wan_motion_prompt": "Slow steady push-in on a 50mm lens. Milo gently rotates the small winding key on the bird's back with his right thumb. Milo opens his palms flat with a hopeful smile. The brass songbird flaps its filigree wings with delicate mechanical precision. Dust motes drift through warm volumetric window sunlight. Rigid room architecture, locked lighting, no face morphing.",
                "camera_movement": "Slow steady push-in on a 50mm prime cinema lens",
            },
            {
                "scene_number": 2,
                "narrative_beat": "Inciting Obstacle",
                "causal_connector": "BUT — A sudden gust of wind blows through the open window, catching the bird and making it spiral helplessly toward the floor",
                "narration_text": "Suddenly, a fierce gust sweeps the fragile bird toward the floor.",
                "narration_voice_style": "urgent_rising",
                "sync_trigger_word": "sweeps",
                "state_ledger": {
                    "entry_state": "Milo standing proudly with palms open, bird just lifting off toward window",
                    "active_props": ["brass mechanical bird (in air)"],
                    "spatial_zone": "Center attic floor between desk and window, screen direction preserved",
                    "atomic_action_5s": "Wind gust buffets the bird -> Milo gasps and lunges forward across the room with outstretched arms",
                    "exit_state": "Milo sliding across wooden floorboards on knees, hands reaching desperately forward",
                },
                "qwen_scene_prompt": "Milo in his leather apron with hands extended as a gust of wind from the circular attic window catches the small brass bird, knocking it into an erratic downward spiral toward the floorboards. Scattered blueprints flutter off the desk.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Milo: Milo has lunged forward into a low knee-slide across the oak floorboards, both hands desperately cupped outward just inches from the falling bird, eyes wide with terror and concentration. Floating blueprints and gears caught in wind currents.",
                "wan_motion_prompt": "Steadicam tracking shot moving alongside Milo at waist level. A sudden crosswind knocks the brass bird into a steep downward spiral. Milo gasps in alarm and dives forward into a desperate knee-slide across the floor. Milo reaches both cupped hands outward toward the falling bird. Blueprints and cedar wood shavings scatter across the floorboards in the wind. Rigid workshop geometry, locked lighting, no morphing.",
                "camera_movement": "Steadicam tracking shot moving alongside Milo at waist level",
            },
            {
                "scene_number": 3,
                "narrative_beat": "Climax & Payoff",
                "causal_connector": "THEREFORE — Milo's gentle fingertips catch the air beneath the bird, giving it the updraft it needs to soar gracefully out into the sky",
                "narration_text": "Milo catches the draft, and the songbird soars into golden skies.",
                "narration_voice_style": "joyful_resolution",
                "sync_trigger_word": "soars",
                "state_ledger": {
                    "entry_state": "Milo sliding on knees near attic window, bird tumbling just above floor",
                    "active_props": ["brass mechanical bird (soaring)"],
                    "spatial_zone": "Attic window sill on screen-right, facing outdoor sky",
                    "atomic_action_5s": "Milo's palms cup the updraft -> bird catches the wind and loops up, soaring out the open circular window into golden clouds",
                    "exit_state": "Milo sitting back on heels by window sill, laughing with tearful joy, pointing at the soaring bird outside",
                },
                "qwen_scene_prompt": "Milo sliding near the circular attic window sill with hands cupped beneath the mechanical brass bird just as it catches the warm thermal current, glowing in late afternoon sunbeams.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Milo: Milo sits back on his heels resting one arm on the windowsill, laughing with boundless radiant joy and tears of relief, pointing with one finger outside the circular window where the brass bird soars gracefully into the golden sunset sky.",
                "wan_motion_prompt": "Slow dolly-in moving gently toward Milo and the circular window. Milo's cupped palms channel the air beneath the brass bird. The bird stabilizes its wings and soars upward in a triumphant spiral through the circular window. Milo sits back on his heels with a beaming joyful laugh, gazing out at the golden sunset horizon. Soft golden dusk light fills the workshop with warm specular highlights. Rigid room geometry, locked lighting, no face morphing.",
                "camera_movement": "Slow dolly-in moving gently toward Milo on a 50mm lens",
            },
        ],
    },
    {
        "title": "The Winter Robin",
        "genre": "Heartwarming / Emotional Drama",
        "art_style": "Warm whimsical storybook animation, Studio Ghibli watercolor aesthetic, soft hearth lighting",
        "theme": "Compassion heals both the giver and the receiver",
        "premise": "An elderly solitary clockmaker nurses a frozen robin back to health through a harsh winter.",
        "characters": [
            {
                "id": "char_1",
                "name": "Silas",
                "face": "Gentle crinkled blue eyes, silver mutton chops beard, wire-rimmed spectacles",
                "hair": "Thin wispy white hair neatly combed back",
                "clothing": "Soft brown woolen vest over linen shirt, rolled sleeves, canvas watchmaker apron",
                "signature_details": "Brass tweezers and magnifying jeweler loupe hung on chest cord",
                "z_image_prompt": "Full-body character model sheet of Silas, kindly elderly clockmaker with silver beard and wire spectacles, wearing brown wool vest, canvas apron, holding brass jeweler loupe, 9:16 vertical portrait, front view, neutral studio backdrop",
            },
            {
                "id": "char_2",
                "name": "Robin",
                "face": "Tiny curved dark beak, bright inquisitive black bead eyes",
                "hair": "Puffed russet-orange breast feathers, slate-grey wing plumage, tiny delicate claws",
                "clothing": "Small clean cotton bandage gently wrapped around right wing",
                "signature_details": "Vibrant orange breast plumage contrasting with winter snow",
                "z_image_prompt": "Full-body reference sheet of Robin, tiny European robin with puffed orange breast feathers and gentle gauze bandage on wing, perched alertly, 9:16 vertical view, clean neutral studio background",
            },
        ],
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat": "Inciting Compassion",
                "causal_connector": "SETUP",
                "narration_text": "On a frozen morning, old Silas finds a shivering little guest.",
                "narration_voice_style": "warm_wonder",
                "sync_trigger_word": "finds",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Silas standing by frost-patterned workshop window, cupping hands toward window sill",
                    "active_props": ["warm woolen handkerchief"],
                    "spatial_zone": "Workshop window, screen-left facing screen-right",
                    "atomic_action_5s": "Silas gently scoops the frozen robin into his wool handkerchief -> cradles it against his chest toward the warm hearth",
                    "exit_state": "Silas kneeling by stone fireplace hearth, cradling the wrapped robin near warm embers",
                },
                "qwen_scene_prompt": "Silas the elderly clockmaker with wire spectacles gently cupping his hands around a shivering tiny robin on the frosted wooden windowsill of his cozy clock workshop. Soft morning snowfall outside, warm wooden clocks ticking on walls.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Silas: Silas is now kneeling beside his stone fireplace hearth, tenderly cradling the robin wrapped in a soft wool handkerchief against his chest, smiling with deep gentle relief as amber hearth embers glow. Keep identical workshop clocks and rustic room geometry.",
                "wan_motion_prompt": "Slow dolly-in moving gently toward Silas on a 50mm lens. Silas cradles the fragile robin into his soft woolen handkerchief. Silas turns smoothly and steps toward the stone fireplace hearth. Silas sinks gently onto one knee by the hearth fire with a tender crinkled smile. Amber hearth flames flicker warmly across brass pendulums. Rigid workshop architecture, locked lighting, no face morphing.",
                "camera_movement": "Slow dolly-in moving gently toward Silas on a 50mm lens",
            },
            {
                "scene_number": 2,
                "narrative_beat": "Bonding & Recovery",
                "causal_connector": "THEREFORE — Warmth and care spark new life in the frail bird",
                "narration_text": "Tended with patient love, the robin pecks sweet honey crumbs.",
                "narration_voice_style": "joyful_resolution",
                "sync_trigger_word": "pecks",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Robin perched on wooden workbench edge, Silas sitting on stool holding breadcrumb saucer",
                    "active_props": ["ceramic saucer of breadcrumbs and seeds"],
                    "spatial_zone": "Center workbench, side-by-side close framing",
                    "atomic_action_5s": "Robin hops bravely forward -> pecks breadcrumb from Silas's open palm -> flutters onto his brass magnifying loupe",
                    "exit_state": "Robin perched triumphantly atop Silas's brass loupe, Silas chuckling with crinkled eyes",
                },
                "qwen_scene_prompt": "Silas sitting at his wooden clockmaker workbench holding a ceramic saucer of seeds, his palm resting open. Robin with healed wing perches on a gear cog facing Silas's open hand. Warm morning sunlight through clear glass.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform the subjects: the little robin has hopped up and perches proudly atop the brass magnifying loupe on Silas's spectacles, while Silas tilts his head laughing softly with crinkled eyes of pure warmth. Keep identical workbench tools, gears, and grandfather clock in background.",
                "wan_motion_prompt": "Orbital tracking shot on an 85mm portrait lens. Robin hops spritely across polished wooden gears. Robin pecks a crumb from Silas's outstretched palm and flutters up onto his brass magnifying loupe. Silas breaks into a hearty, silent chuckle, eyes brimming with fond delight. Dust motes dance in golden sunbeams. Fixed workbench geometry, locked lighting, no morphing.",
                "camera_movement": "Orbital tracking shot on an 85mm portrait lens, gentle 15-degree arc",
            },
            {
                "scene_number": 3,
                "narrative_beat": "Bittersweet Release & Joy",
                "causal_connector": "THEREFORE — Silas gives the gift of freedom as spring blooms",
                "narration_text": "With wings unfurled, the songbird takes flight into spring.",
                "narration_voice_style": "epic_climax",
                "sync_trigger_word": "unfurled",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Silas standing at wide open workshop window, robin perched on his extended forefinger",
                    "active_props": ["flowering cherry branch outside sill"],
                    "spatial_zone": "Window framing the blooming green valley, screen-left to right",
                    "atomic_action_5s": "Robin chirps a melodic note -> launches into the crisp spring air -> soars toward sunlit cherry blossoms",
                    "exit_state": "Silas leaning against windowsill with peaceful hand on chest, watching bird glide into golden sky",
                },
                "qwen_scene_prompt": "Silas standing before his wide open workshop window with the healthy robin resting on his index finger. Outside, white snow has melted into blooming pink cherry blossoms and golden morning skies.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Silas: the robin has launched into flight and glides gracefully into the golden cherry blossoms outside, while Silas rests both hands on the wooden sill with a serene, grateful smile of lasting fulfillment. Keep identical window frame and workshop wall.",
                "wan_motion_prompt": "Crane boom-down from high angle on a 35mm wide lens settling into medium shot. Robin ruffles its orange feathers and leaps into the warm spring breeze. Robin sweeps upward past blossoming cherry boughs. Silas leans on the windowsill with a deep breath of contentment, basking in morning golden light. Petals drift softly through the open window. Rigid architectural frame, locked exposure, no morphing.",
                "camera_movement": "Crane boom-down from high angle on a 35mm wide lens settling into medium shot",
            },
        ],
    },
    {
        "title": "The Stargazer's Lantern",
        "genre": "Heartwarming / Family",
        "art_style": "Luminous storybook illustration, soft pastel textures, deep indigo night sky with golden glow",
        "theme": "Love bridges the greatest distances through simple acts of devotion",
        "premise": "A little girl and her faithful hound carry a handmade glowing paper lantern up a hill to signal her bedridden grandmother.",
        "characters": [
            {
                "id": "char_1",
                "name": "Maeve",
                "face": "Rosy cheeks, button nose, wide hopeful green eyes, bright eager expression",
                "hair": "Wavy hazelnut brown pigtails tied with red yarn ribbons",
                "clothing": "Chunky ochre yellow knit sweater, indigo wool skirt, sturdy brown leather boots",
                "signature_details": "Hand-painted dodecahedron paper lantern with cut-out star silhouettes, glowing amber",
                "z_image_prompt": "Full-body character model sheet of Maeve, 8-year-old girl with hazelnut pigtails in ochre knit sweater, holding glowing handmade paper star lantern with both hands, 9:16 vertical full-body portrait, front view, clean neutral studio background",
            },
            {
                "id": "char_2",
                "name": "Barnaby",
                "face": "Gentle dark eyes, grey muzzle, warm panting tongue, expressive bushy eyebrows",
                "hair": "Thick golden-honey shaggy fur, feathery plumy tail, floppy ears",
                "clothing": "Woven red bandanna tied neatly around neck",
                "signature_details": "Woven red bandanna and gentle protective stance beside Maeve",
                "z_image_prompt": "Full-body reference model sheet of Barnaby, shaggy golden retriever with floppy ears and red bandanna around neck, standing loyally, 9:16 vertical view, clean neutral studio backdrop",
            },
        ],
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat": "Preparation of Love",
                "causal_connector": "SETUP",
                "narration_text": "Beside the fire, Maeve lights her lantern for grandma.",
                "narration_voice_style": "warm_wonder",
                "sync_trigger_word": "lights",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Maeve sitting cross-legged on rug before wooden table, Barnaby resting chin on her knee",
                    "active_props": ["paper star lantern with LED candle inside"],
                    "spatial_zone": "Cozy living room rug, center framing",
                    "atomic_action_5s": "Maeve touches switch inside lantern -> lantern illuminates with warm amber star constellations -> Barnaby wags tail and nudges her elbow",
                    "exit_state": "Maeve lifting the glowing lantern with both hands, Barnaby standing up alertly beside her",
                },
                "qwen_scene_prompt": "Maeve in her ochre sweater sitting cross-legged on a braided rug beside her shaggy golden dog Barnaby, placing a small candle inside a beautiful handmade paper star lantern. Warm cabin fire in background.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform the subjects: Maeve is now standing upright holding the brilliantly glowing paper star lantern high with both hands, her face lit with wonder, while Barnaby stands tall beside her wagging his tail. Keep identical braided rug, bookshelf, and fireplace architecture.",
                "wan_motion_prompt": "Slow push-in on a 50mm cinema lens. Maeve clicks the lantern candle alight. Amber light beams through star-shaped perforations across Maeve's face. Barnaby lifts his head and wags his bushy tail in enthusiastic approval. Maeve rises smoothly to her feet with lantern aloft. Warm amber shadows dance gently on the wooden walls. Fixed room geometry, locked lighting, no face morphing.",
                "camera_movement": "Slow push-in on a 50mm cinema lens with shallow depth of field",
            },
            {
                "scene_number": 2,
                "narrative_beat": "The Journey Upward",
                "causal_connector": "THEREFORE — Maeve and Barnaby brave the chilly night to reach the summit",
                "narration_text": "Through cold twilight winds, they climb the quiet hill.",
                "narration_voice_style": "tense_whisper",
                "sync_trigger_word": "climb",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Maeve walking up grassy hill slope, holding lantern, Barnaby trotting alongside her",
                    "active_props": ["glowing paper star lantern"],
                    "spatial_zone": "Hillside path, screen-left moving toward screen-right",
                    "atomic_action_5s": "Gust of wind rustles grass -> Barnaby leans against Maeve's legs to steady her -> Maeve smiles and takes final steps to summit",
                    "exit_state": "Maeve and Barnaby standing side-by-side at hill crest overlooking deep blue valley below",
                },
                "qwen_scene_prompt": "Maeve holding her glowing star lantern walking up a gentle grassy hill at twilight alongside Barnaby the golden dog. Cool twilight blue sky with early evening stars, breeze swaying prairie grass.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform the scene: Maeve and Barnaby have reached the rocky hilltop crest and stand side-by-side looking out toward screen-right, the lantern casting a warm golden radius around them against the deep indigo night sky. Keep identical hill terrain and starry sky.",
                "wan_motion_prompt": "Steadicam tracking shot moving alongside Maeve on a 35mm lens. Maeve strides determinedly up the grassy incline. Barnaby trots alongside her with red bandanna fluttering in the breeze. Maeve reaches the hill crest and plants her boots firmly, lifting the lantern. Tall prairie grass sways rhythmically in the wind. Solid horizon line, zero camera jitter, no morphing.",
                "camera_movement": "Steadicam tracking shot moving alongside Maeve on a 35mm lens",
            },
            {
                "scene_number": 3,
                "narrative_beat": "The Signal of Connection",
                "causal_connector": "THEREFORE — Across the valley, love answers without words",
                "narration_text": "Across the sleeping valley, a golden light blinks reply.",
                "narration_voice_style": "joyful_resolution",
                "sync_trigger_word": "blinks",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Maeve holding lantern aloft at hill crest, Barnaby sitting loyally beside her",
                    "active_props": ["glowing paper star lantern held high"],
                    "spatial_zone": "Hilltop foreground overlooking distant valley cottages",
                    "atomic_action_5s": "Maeve swings lantern in wide gentle arcs -> across the valley, cottage window flashes two bright golden pulses -> Maeve hugs Barnaby with joyful laughter",
                    "exit_state": "Maeve kneeling in embrace with Barnaby, lantern resting on rock beside them, both smiling at the blinking cottage light",
                },
                "qwen_scene_prompt": "Maeve holding her glowing star lantern high above her head on the dark hill crest with Barnaby looking into the valley. In the distant valley below, a tiny cozy cottage window glows warm.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Maeve: Maeve is now wrapped in a joyful hug around Barnaby's neck laughing happily with tears of joy, while the distant cottage window in the valley pulses brightly twice with warm amber light. The paper lantern rests safe and glowing on a rock beside them. Keep identical starry sky and mountain silhouettes.",
                "wan_motion_prompt": "Crane boom-up on a 28mm wide lens expanding to panoramic night view. Maeve sways her glowing lantern overhead in gentle arcs. Across the dark valley, the cottage window pulses two radiant flashes. Maeve squeals with delighted laughter and drops to hug Barnaby around the neck. Barnaby licks her cheek while night fireflies flicker around the lantern. Locked landscape geometry, stable starlight, no face morphing.",
                "camera_movement": "Crane boom-up on a 28mm wide lens expanding to panoramic night view",
            },
        ],
    },
    {
        "title": "The Baker's Secret",
        "genre": "Heartwarming / Whimsical",
        "art_style": "Cozy European picture-book art, warm amber lighting, rich textured brushstrokes",
        "theme": "A warm loaf of bread shared is worth a kingdom",
        "premise": "A rainy evening, an orphan boy with a soaked stray kitten, and a master baker who opens his warm door.",
        "characters": [
            {
                "id": "char_1",
                "name": "Leo",
                "face": "Hopeful amber eyes, smudged nose, shy tentative smile",
                "hair": "Tousled chestnut curls peeking out under a brown newsboy cap",
                "clothing": "Patched green wool coat with wooden toggles, brown trousers, wool socks",
                "signature_details": "Brown newsboy cap and bundled coat sheltering a kitten",
                "z_image_prompt": "Full-body character model sheet of Leo, 10-year-old boy in patched green coat and brown newsboy cap, gently cradling small calico kitten in coat fold, 9:16 vertical full-body portrait, front view, clean neutral studio backdrop",
            },
            {
                "id": "char_2",
                "name": "Ezra",
                "face": "Round jovial cheeks, kind laughing brown eyes, snowy handlebar mustache",
                "hair": "Bald crown with silver fringe, tall white baker toque hat",
                "clothing": "Flour-dusted white chef jacket, striped apron with large front pocket, wooden clogs",
                "signature_details": "Flour dusted apron, baker toque, wooden rolling pin in apron pocket",
                "z_image_prompt": "Full-body reference sheet of Ezra, kindly master baker with snowy mustache and baker toque hat, wearing flour-dusted apron, warm welcoming posture, 9:16 vertical view, clean neutral studio backdrop",
            },
            {
                "id": "char_3",
                "name": "Mochi",
                "face": "Mismatched calico patches over pink nose, large bright emerald eyes",
                "hair": "Soft tricolor calico fur (white, ginger, slate), tiny pink paw pads",
                "clothing": "None",
                "signature_details": "Tiny calico kitten with pink nose and alert twitching ears",
                "z_image_prompt": "Full-body reference model sheet of Mochi, tiny calico kitten with emerald eyes and pink nose, curled up sweetly, 9:16 vertical portrait, clean neutral studio background",
            },
        ],
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat": "Cold Need",
                "causal_connector": "SETUP",
                "narration_text": "In the cold rain, little Leo shelters shivering Mochi.",
                "narration_voice_style": "tense_whisper",
                "sync_trigger_word": "shelters",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Leo standing under cobblestone bakery awning in evening rain, coat pulled open to shield kitten",
                    "active_props": ["kitten Mochi tucked inside coat"],
                    "spatial_zone": "Bakery window exterior, screen-left looking at glowing window pane",
                    "atomic_action_5s": "Leo pulls coat tighter around shivering kitten -> peers through misted bakery glass at golden loaves inside",
                    "exit_state": "Leo pressing small hand against warm glass pane, kitten poking nose out with wide eyes",
                },
                "qwen_scene_prompt": "Leo the young boy in patched green coat and newsboy cap standing on wet cobblestones outside a glowing bakery shop in evening rain, carefully shielding tiny calico kitten Mochi inside his coat. Raindrops running down the warm bakery window.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform Leo: Leo is now leaning closer to the misted glass with his palm pressed gently against it, his eyes wide with yearning as the little kitten rests its paws on his coat collar looking at the warm bread inside. Keep identical brick bakery wall, cobblestone reflections, and rain streaks.",
                "wan_motion_prompt": "Slow dolly-in moving gently toward Leo on a 50mm cinema lens. Leo pulls his woolen coat snug around tiny shivering Mochi. Leo steps softly to the misted bakery glass and rests his palm against the warm surface. Mochi pokes its pink nose out and twitches its ears. Rain trickles down cobblestone eaves with glistening reflections. Solid brick architecture, locked warm window light, no morphing.",
                "camera_movement": "Slow dolly-in moving gently toward Leo on a 50mm cinema lens",
            },
            {
                "scene_number": 2,
                "narrative_beat": "The Warm Welcome",
                "causal_connector": "BUT — The heavy oak bakery door swings open with a golden bell chime",
                "narration_text": "Master Ezra swings the door wide with sweet brioche.",
                "narration_voice_style": "warm_wonder",
                "sync_trigger_word": "swings",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Bakery door cracking open, Ezra looking down from doorway, Leo startled stepping back",
                    "active_props": ["warm braided brioche loaf held on baker cloth"],
                    "spatial_zone": "Bakery doorway threshold",
                    "atomic_action_5s": "Ezra opens door wide with booming gentle smile -> extends steaming golden brioche toward Leo and kitten",
                    "exit_state": "Ezra gesturing warmly for them to enter, Leo smiling with disbelief as kitten sniffs the warm bread",
                },
                "qwen_scene_prompt": "The wooden bakery door opened wide spilling golden light into the rainy street. Ezra the baker with flour-dusted mustache smiles down offering a steaming golden brioche loaf to Leo and kitten Mochi.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform the subjects: Ezra has knelt down on one knee offering the warm loaf directly to Leo with an affectionate beckoning gesture, while Leo's face breaks into a radiant, shy smile of gratitude and Mochi reaches one tiny paw out toward the pastry. Keep identical doorway, warm interior oven glow, and rain outside.",
                "wan_motion_prompt": "Steadicam tracking shot moving alongside doorway on a 35mm lens. The wooden door opens smoothly as the brass bell rings. Ezra sinks onto one knee with an expansive, kindly grin, holding out the braided brioche. Steam wafts gently from the golden crust. Leo's guarded posture relaxes into a wide, tearful smile. Raindrops sparkle in the door's threshold light. Stable doorframe geometry, locked lighting, no face morphing.",
                "camera_movement": "Steadicam tracking shot moving alongside doorway on a 35mm lens",
            },
            {
                "scene_number": 3,
                "narrative_beat": "Belonging & Flour Dust",
                "causal_connector": "THEREFORE — Leo finds a home and the secret recipe of warmth",
                "narration_text": "Together they knead dough as Mochi sleeps safe.",
                "narration_voice_style": "joyful_resolution",
                "sync_trigger_word": "knead",
                "transition_type": "crossfade",
                "state_ledger": {
                    "entry_state": "Leo and Ezra standing at large wooden baking table, Mochi curled in bread basket by brick oven",
                    "active_props": ["mound of golden dough, wooden rolling pin, wicker bread basket"],
                    "spatial_zone": "Bakery kitchen, cozy oven hearth",
                    "atomic_action_5s": "Ezra places his flour-dusted hands over Leo's small hands to guide kneading -> Leo giggles -> puff of flour scatters in the air",
                    "exit_state": "Leo expertly folding dough with flour on tip of nose, Ezra laughing proudly, Mochi purring deeply in wool basket",
                },
                "qwen_scene_prompt": "Inside the warm cozy bakery kitchen. Ezra and Leo standing side-by-side at a wooden table covered in flour, kneading bread dough together. Little calico kitten Mochi sleeps curled up in a wicker basket beside the warm brick oven.",
                "qwen_end_scene_prompt": "image 1 is the starting frame. For this ending frame at t=5.0s, transform the scene: Leo is laughing with a dab of white flour on the tip of his nose as he proudly shapes a small round loaf, while Ezra claps his hands in proud celebration with flour puffing into the air. Mochi stretches contentedly in the basket with tail curled. Keep identical brick oven, copper pots, and warm golden illumination.",
                "wan_motion_prompt": "Slow dolly-in moving gently toward the kneading table on a 50mm lens. Ezra places his large warm hands over Leo's hands, guiding a rhythmic press-and-fold into the elastic dough. A soft cloud of flour dust rises and catches the golden lamplight. Leo beams with triumphant pride, glancing at kitten Mochi sleeping peacefully beside the brick hearth. Rigid kitchen cabinetry, locked lighting, no morphing.",
                "camera_movement": "Slow dolly-in moving gently toward the kneading table on a 50mm lens",
            },
        ],
    },
]


# ── 7. Self-Learning Episodic Story Memory Bank ───────────────────────────────
class StoryMemoryBank:
    """
    Manages long-term episodic memory for logical animated stories.
    Stores user-approved stories, human prompt edits, DPO training pairs, and
    clip quality telemetry in local JSONL files. Dynamically retrieves relevant
    few-shot exemplars to train LLMs in-context on user taste, causal narrative
    logic, and synchronized narration.
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        memory_filename: str = "story_training_memory.jsonl",
        dpo_filename: str = "story_dpo_pairs.jsonl",
        telemetry_filename: str = "story_clip_telemetry.jsonl",
        storage_dir: Optional[str] = None,
    ):
        base_dir = pathlib.Path(memory_dir or storage_dir or os.path.join("outputs", "studio"))
        base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = base_dir / memory_filename
        self.dpo_file = base_dir / dpo_filename
        self.telemetry_file = base_dir / telemetry_filename

    def save_exemplar(
        self,
        premise: str,
        story_data: Dict[str, Any],
        rating: float = 10.0,
        user_edited: bool = False,
        notes: str = "",
    ) -> bool:
        """Saves a high-quality or user-edited story to the long-term memory bank."""
        try:
            record = {
                "timestamp": str(pathlib.Path(self.memory_file).stat().st_mtime if self.memory_file.exists() else 0),
                "premise": premise,
                "rating": rating,
                "user_edited": user_edited,
                "notes": notes,
                "title": story_data.get("title", ""),
                "art_style": story_data.get("art_style", ""),
                "characters": story_data.get("characters", []),
                "scenes": [
                    {
                        "scene_number": s.get("scene_number"),
                        "title": s.get("title"),
                        "narrative_beat": s.get("narrative_beat"),
                        "causal_connector": s.get("causal_connector"),
                        "narration_text": s.get("narration_text", ""),
                        "narration_voice_style": s.get("narration_voice_style", "cinematic_warm"),
                        "sync_trigger_word": s.get("sync_trigger_word", ""),
                        "state_ledger": s.get("state_ledger", {}),
                        "scenario": s.get("scenario"),
                        "qwen_scene_prompt": s.get("qwen_scene_prompt"),
                        "qwen_end_scene_prompt": s.get("qwen_end_scene_prompt"),
                        "wan_motion_prompt": s.get("wan_motion_prompt"),
                        "camera_movement": s.get("camera_movement"),
                        "telemetry": s.get("telemetry", {}),
                    }
                    for s in story_data.get("scenes", [])
                ],
            }
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[StoryMemoryBank] Failed to save exemplar: {e}")
            return False

    def record_dpo_pair(
        self,
        premise: str,
        rejected_story: Dict[str, Any],
        chosen_story: Dict[str, Any],
        critique_feedback: str = "",
    ) -> bool:
        """
        Saves a (rejected, chosen) preference pair for Direct Preference Optimization (DPO)
        whenever the Story Doctor Critic fixes a flawed storyboard or the user edits scenes.
        """
        try:
            record = {
                "premise": premise,
                "critique_feedback": critique_feedback,
                "rejected": {
                    "title": rejected_story.get("title", ""),
                    "scenes": [
                        {
                            "scene_number": s.get("scene_number"),
                            "scenario": s.get("scenario"),
                            "narration_text": s.get("narration_text", ""),
                            "sync_trigger_word": s.get("sync_trigger_word", ""),
                            "qwen_scene_prompt": s.get("qwen_scene_prompt"),
                            "qwen_end_scene_prompt": s.get("qwen_end_scene_prompt"),
                            "state_ledger": s.get("state_ledger", {}),
                        }
                        for s in rejected_story.get("scenes", [])
                    ],
                },
                "chosen": {
                    "title": chosen_story.get("title", ""),
                    "scenes": [
                        {
                            "scene_number": s.get("scene_number"),
                            "scenario": s.get("scenario"),
                            "narration_text": s.get("narration_text", ""),
                            "sync_trigger_word": s.get("sync_trigger_word", ""),
                            "qwen_scene_prompt": s.get("qwen_scene_prompt"),
                            "qwen_end_scene_prompt": s.get("qwen_end_scene_prompt"),
                            "state_ledger": s.get("state_ledger", {}),
                        }
                        for s in chosen_story.get("scenes", [])
                    ],
                },
            }
            with open(self.dpo_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[StoryMemoryBank] Failed to save DPO pair: {e}")
            return False

    def record_clip_telemetry(
        self,
        scene_number: int,
        telemetry: Dict[str, Any],
        director_plan: Optional[Dict[str, Any]] = None,
        title: str = "",
    ) -> bool:
        """Records automated post-render video quality and audio-visual sync telemetry."""
        try:
            record = {
                "scene_number": scene_number,
                "title": title,
                "telemetry": telemetry,
                "director_plan": director_plan or {},
            }
            with open(self.telemetry_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[StoryMemoryBank] Failed to record telemetry: {e}")
            return False

    def retrieve_exemplars(self, premise: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves the most semantically relevant exemplars from memory + curated blueprints.
        Uses keyword overlap and narrative style matching.
        """
        candidates: List[Dict[str, Any]] = list(CAUSAL_NARRATIVE_BLUEPRINTS)

        # Also load from memory file if available
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            candidates.append(json.loads(line))
            except Exception as e:
                print(f"[StoryMemoryBank] Error reading memory bank: {e}")

        if not candidates:
            return []

        # Simple semantic keyword scoring against premise
        words = set(re.findall(r"\w+", premise.lower()))
        scored = []
        for cand in candidates:
            text = f"{cand.get('title', '')} {cand.get('theme', '')} {cand.get('genre', '')} {cand.get('premise', '')}".lower()
            cand_words = set(re.findall(r"\w+", text))
            score = len(words.intersection(cand_words))
            # Prefer user-edited or high-rated exemplars
            if cand.get("user_edited"):
                score += 2
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def format_few_shot_prompt(self, exemplars: List[Dict[str, Any]]) -> str:
        """Formats retrieved exemplars into high-impact in-context training examples."""
        if not exemplars:
            return ""

        output = ["### GOLD-STANDARD CAUSAL STORYBOARD EXAMPLES (Follow this exact structure):\n"]
        for idx, ex in enumerate(exemplars, 1):
            output.append(f"Example {idx}: '{ex.get('title', 'Storyboard')}'")
            output.append(f"Premise: {ex.get('premise', ex.get('theme', ''))}")
            output.append("Scenes with Causal Connectors, Voiceover Narration & State Ledgers:")
            for sc in ex.get("scenes", [])[:3]:
                connector = sc.get("causal_connector", "THEREFORE")
                beat = sc.get("narrative_beat", f"Beat {sc.get('scene_number', 1)}")
                ledger = sc.get("state_ledger", {})
                narration = sc.get("narration_text", "")
                trigger = sc.get("sync_trigger_word", "")
                output.append(f"- Scene {sc.get('scene_number', 1)} [{beat}] ({connector}):")
                if narration:
                    output.append(f"  Narration (6-11 words): \"{narration}\" [Sync Trigger Word: '{trigger}']")
                output.append(f"  Entry State: {ledger.get('entry_state', 'Character at starting stance')}")
                output.append(f"  Active Props: {', '.join(ledger.get('active_props', [])) or 'None'}")
                output.append(f"  Atomic 5s Action: {ledger.get('atomic_action_5s', sc.get('scenario', ''))}")
                output.append(f"  Exit State: {ledger.get('exit_state', 'Settled post-action posture')}")
                output.append(f"  Start Prompt: {sc.get('qwen_scene_prompt', '')}")
                output.append(f"  End Prompt: {sc.get('qwen_end_scene_prompt', '')}")
            output.append("\n" + "=" * 50 + "\n")

        return "\n".join(output)


