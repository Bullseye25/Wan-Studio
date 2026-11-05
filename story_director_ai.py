"""
Story Director AI — Multi-Pass Writers' Room, Tone-Aware Cinematography & Pose Library.
========================================================================================
Empowers the AI Director, Screenwriter, and Narrator with:
  1. Multi-Pass Screenplay Architecture (Beat Sheet -> Scene Drafting -> Editor Audit)
  2. Tone & Genre Awareness (Kids/Storybook, Adventure, Mystery/Noir, Comedy)
  3. Tone-Aware Dynamic Pose Library (replaces hardcoded combat poses with genre-authentic actions)
  4. Banned Vocabulary & Invariant Guard per Genre (zero shockwaves/cover in kids stories)
  5. Show-Don't-Tell Physical Emotional Cues (facial expressions, body language)
  6. Narrator Performance Scripting (prosody, emotional cadence, micro-pauses)
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple

# ── 1. Genre Taxonomy & Invariants ───────────────────────────────────────────

GENRE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "kids": {
        "name": "Kids Storybook",
        "description": "Gentle, heartwarming, cute 3D cartoon style for young children",
        "banned_words": [
            "shockwave", "weapon", "weapons", "debris", "recoil", "recoiling",
            "defensively", "behind cover", "blood", "gun", "bullet", "murder",
            "sinister", "corpse", "slay", "brutal", "combat", "fight", "blast",
            "shielding face", "horror", "nightmare", "deadly", "assassin"
        ],
        "tone_prompt": (
            "Tone: Warm, gentle, joyful, innocent, and heartwarming. "
            "Focus on kindness, friendship, sharing, curiosity, and colorful storybook wonder. "
            "Characters express emotions through big eyes, smiles, playful hops, and caring gestures. "
            "STRICTLY FORBIDDEN: Combat, weapons, ducking behind cover, shockwaves, or dark threats."
        ),
        "default_voice": "Warm Storyteller (Female - af_heart)",
        "reading_level": "B1",
        "default_art_style": "Cute vibrant 3D cartoon animation for kids, Pixar storybook aesthetic, bright sunny lighting, colorful and charming, 4K",
    },
    "adventure": {
        "name": "Heroic Adventure",
        "description": "Dynamic, courageous journey with heroic stakes",
        "banned_words": ["office", "boredom", "mundane", "corporate", "cubicle"],
        "tone_prompt": (
            "Tone: Courageous, inspiring, epic, and full of discovery. "
            "Characters face grand challenges with determination, bravery, and teamwork. "
            "Focus on majestic landscapes, dynamic movement, and heroic triumphs."
        ),
        "default_voice": "Energetic Action Narrator (Male - am_michael)",
        "reading_level": "B1",
        "default_art_style": "Dynamic cinematic 3D animation, Makoto Shinkai atmospheric lighting, epic sense of scale, rich textures, 4K",
    },
    "mystery": {
        "name": "Detective & Mystery",
        "description": "Atmospheric, suspenseful noir deduction",
        "banned_words": ["slapstick", "silly", "rainbows", "cuddly", "tummy rumbles"],
        "tone_prompt": (
            "Tone: Moody, atmospheric, analytical, and suspenseful. "
            "Focus on shadows, clues, deduction, sharp observations, and resolving puzzles."
        ),
        "default_voice": "Cinematic Trailer (Male - am_fenrir)",
        "reading_level": "B1",
        "default_art_style": "Atmospheric cinematic animation, dramatic chiaroscuro lighting, deep noir shadows with neon rim light, 4K",
    },
    "comedy": {
        "name": "Playful Comedy",
        "description": "Lively visual humor and expressive character timing",
        "banned_words": ["tragic", "gruesome", "depressing", "corpse"],
        "tone_prompt": (
            "Tone: Playful, humorous, bouncy, and lighthearted. "
            "Focus on expressive comedic timing, fun misunderstandings, and cheerful visual payoffs."
        ),
        "default_voice": "Sweet Expressive (Female - af_sarah)",
        "reading_level": "B1",
        "default_art_style": "Vibrant stylized cartoon animation, exaggerated expressions, bright colorful palette, studio lighting, 4K",
    },
    "unfiltered_action": {
        "name": "High-Octane Action & Combat",
        "description": "Visceral, high-stakes choreography, intense martial arts melee, and kinetic stunt work without cartoon censorship",
        "banned_words": [],
        "tone_prompt": (
            "Tone: Visceral, gritty, high-stakes, adrenaline-fueled, and kinetic. "
            "Focus on intense martial arts choreography, fast-paced melee exchanges, tactical positioning, "
            "clashing weapons with shower of sparks, shockwaves, dynamic evasive maneuvers, and relentless momentum. "
            "Cinematic bullet-time, impactful physical hits, and high-energy camera motion."
        ),
        "default_voice": "Energetic Action Narrator (Male - am_michael)",
        "reading_level": "B1",
        "default_art_style": "High-octane cinematic 3D anime action, gritty cyberpunk or dark fantasy, volumetric smoke, lens flare, intense sparks, 4K",
    },
    "mature_thriller": {
        "name": "Gritty Noir & Mature Thriller",
        "description": "Dark suspense, criminal underworld conflicts, visceral stakes, and psychological tension",
        "banned_words": ["slapstick", "silly", "rainbows", "cuddly", "tummy rumbles"],
        "tone_prompt": (
            "Tone: Dark, gritty, tense, uncompromising, and suspenseful. "
            "Focus on high-stakes underworld confrontations, intense interrogations, visceral danger, "
            "shadowy alleyways, neon reflections on wet asphalt, and razor-sharp tactical tension."
        ),
        "default_voice": "Cinematic Trailer (Male - am_fenrir)",
        "reading_level": "B1",
        "default_art_style": "Dark gritty cinematic animation, dramatic chiaroscuro lighting, deep noir shadows, neon reflections on wet pavement, 4K",
    },
}


def detect_story_genre(premise: str, art_style: str = "") -> str:
    """
    Infers the most appropriate genre taxonomy based on premise and visual cues.
    """
    text = (f"{premise} {art_style}").lower()

    kids_keywords = [
        "kids", "child", "children", "baby", "fluffy", "cute", "gentle", "bear",
        "rabbit", "bunny", "sharing", "caring", "friendship", "friend", "meadow",
        "storybook", "pixar", "robin", "clockmaker", "puppy", "kitten", "sweet"
    ]
    action_keywords = [
        "ninja", "assassin", "combat", "fight", "brawl", "melee", "duel", "sword",
        "gun", "bullet", "heist", "strike", "clash", "explos", "cybernetic", "operative",
        "unfiltered", "action", "showdown", "chase", "sniper", "ambush"
    ]
    mystery_keywords = [
        "batman", "joker", "gotham", "arkham", "detective", "mystery", "murder",
        "investigat", "clue", "noir", "shadow", "crime", "asylum", "inspector", "thriller"
    ]
    adventure_keywords = [
        "warrior", "dragon", "quest", "kingdom", "galaxy", "cyberpunk",
        "samurai", "journey", "explore", "relic", "ancient", "temple", "battle"
    ]

    kids_score = sum(1 for k in kids_keywords if k in text)
    action_score = sum(1 for k in action_keywords if k in text)
    mystery_score = sum(1 for k in mystery_keywords if k in text)
    adventure_score = sum(1 for k in adventure_keywords if k in text)

    # Kids genre check takes priority if explicitly targeted at children
    if kids_score >= max(action_score, mystery_score, adventure_score, 1):
        return "kids"
    if action_score >= max(kids_score, mystery_score, adventure_score, 1):
        return "unfiltered_action"
    if mystery_score >= max(kids_score, adventure_score, 1):
        return "mature_thriller" if any(k in text for k in ("thriller", "underworld", "mature", "gritty")) else "mystery"
    if adventure_score > 0:
        return "adventure"
    return "kids" if ("cute" in text or "rabbit" in text or "bear" in text) else "adventure"


def sanitize_genre_prompt(text: str, genre: str = "kids") -> str:
    """
    Strips out genre-banned words and replaces them with tone-appropriate synonyms.
    Guarantees that a kids' story never receives combat / shockwave directives.
    """
    cfg = GENRE_CONFIGS.get(genre, GENRE_CONFIGS["kids"])
    banned = cfg.get("banned_words", [])

    clean = text
    for word in banned:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        if genre == "kids":
            # Context-sensitive peaceful replacements
            if word in ("shockwave", "blast"):
                clean = pattern.sub("burst of colorful sparkles", clean)
            elif word in ("weapon", "weapons", "gun", "bullet"):
                clean = pattern.sub("friendly gift", clean)
            elif word in ("defensively", "behind cover"):
                clean = pattern.sub("warmly together", clean)
            elif word in ("recoil", "recoiling"):
                clean = pattern.sub("leaning back in joyful wonder", clean)
            elif word in ("combat", "fight"):
                clean = pattern.sub("playful game", clean)
            else:
                clean = pattern.sub("gentle moment", clean)
        else:
            clean = pattern.sub("", clean)

    # Clean up double spaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ── 2. Tone-Aware Dynamic Pose Library ───────────────────────────────────────

TONE_AWARE_POSE_LIBRARY: Dict[str, Dict[str, str]] = {
    "kids": {
        "share": "gently offering the basket with both paws outstretched, leaning forward with a joyful caring smile",
        "give": "holding out a sweet treat in both open paws, looking into their friend's eyes with warm affection",
        "eat": "sitting happily on the soft grass, nibbling the delicious fruit with eyes closed in pure delight",
        "chew": "happily chewing the berry with rosy cheeks, smiling warmly as sweet juice drips onto paws",
        "find": "proudly lifting the basket high with both paws, standing on tiptoes with wide sparkling eyes and a big grin",
        "discover": "kneeling down gently on the grass, peering into the colorful flowers with wonder and happy curiosity",
        "wave": "raising one friendly paw high overhead in an energetic wave, torso tilted with cheerful excitement",
        "meet": "stepping forward with open arms in a warm welcoming greeting, ears perked and tail wagging happily",
        "hop": "playfully hopping forward with one foot raised in a skip, arms swinging merrily",
        "run": "trotting forward with cheerful bouncing steps, face full of eager anticipation and joy",
        "sad": "sitting quietly with paws tucked close to chest, head tilted slightly with hopeful, gentle eyes",
        "hug": "leaning in for a warm, affectionate hug, arms wrapped gently around their friend with eyes closed happily",
        "celebrate": "both friends raising their arms in joyful celebration, jumping lightly in the meadow together",
        "default": "standing cheerfully with a radiant smile, arms open and body relaxed in friendly storybook joy",
    },
    "adventure": {
        "reach": "standing at the edge of the overlook, one hand raised shielding eyes toward the golden horizon",
        "leap": "landing firmly on one knee, one hand braced against the earth, eyes locked forward with fierce determination",
        "strike": "lunging forward in a strong heroic stance, blade held firm and steady, cape billowing in the wind",
        "run": "sprinting forward with powerful momentum, leaning low into the wind with focused heroic gaze",
        "discover": "raising an ancient artifact toward the light, glowing reflections illuminating their determined expression",
        "default": "standing tall in a resolute heroic posture, chest high, ready for the grand journey ahead",
    },
    "mystery": {
        "inspect": "crouching low beside the clue, shining the focused beam of light with an intense analytical gaze",
        "discover": "holding a small evidence vial up to the dim neon light, eyes narrowed in sharp realization",
        "enter": "stepping silently past the heavy doorway, one hand bracing the frame, observing the darkened room",
        "confront": "standing tall in the shadows, trench coat wrapped tight, gaze locked squarely on the adversary",
        "default": "standing motionless in the doorway, silhouetted against the dim ambient rim light",
    },
    "comedy": {
        "surprise": "stumbling backward with wide comedic eyes, arms flailing wildly in cheerful disbelief",
        "slip": "sliding comically across the floor, spinning around once and ending up with a sheepish grin",
        "laugh": "leaning back holding their stomach in uncontrollable laughter, mouth wide open in joyful giggles",
        "default": "standing with a cheeky sideways grin, one eyebrow raised in playful comedic amusement",
    },
    "unfiltered_action": {
        "strike": "recovering dynamically from a high-speed strike, weapon locked in a forward guard, sparks dissipating across armor",
        "clash": "locked blade-to-blade in an intense close-quarters standoff, muscles strained, sparks showering between their faces",
        "evade": "sliding low across the wet ground beneath an enemy slash, recovering instantly into a tactical counter-stance",
        "land": "landing heavily from an aerial leap with one fist cratering the pavement, kinetic shockwave rippling outwards",
        "aim": "raising weapon in a rock-steady two-handed grip, intense focused gaze locked on target through swirling smoke",
        "default": "standing in a primed, low-center-of-gravity combat stance, blades ready, eyes locked on the opponent",
    },
    "mature_thriller": {
        "confront": "stepping into the dim neon streetlight, face partially cast in shadow, eyes cold and unwavering",
        "draw": "hand resting deliberately on concealed weapon, posture relaxed but primed for lethal reaction",
        "escape": "ducking through the rain-slicked alley corridor, glancing sharply over one shoulder before vanishing into darkness",
        "default": "standing poised in the shadows, trench coat fluttering in the chill wind, watching the target silently",
    },
}


def get_tone_aware_end_pose(
    scenario: str,
    genre: str = "kids",
    char_names: str = "characters",
) -> str:
    """
    Selects an expressive, physically grounded, tone-authentic pose for the scene ending frame (t=5.0s).
    Guarantees no combat/shockwave hallucinations in kids/gentle stories.
    """
    g = genre.lower().strip()
    if g not in TONE_AWARE_POSE_LIBRARY:
        g = "kids" if any(k in g for k in ("kid", "storybook", "cute", "gentle")) else "adventure"

    poses = TONE_AWARE_POSE_LIBRARY[g]
    text = (scenario or "").lower()

    # Rich inflection & synonym patterns for core animation actions
    VERB_FORMS = {
        "eat": r"\b(eat|eats|eating|ate|nibble|nibbles|nibbling|chew|chews|chewing)\b",
        "give": r"\b(give|gives|giving|gave|offer|offers|offering|hand|hands|handing)\b",
        "share": r"\b(share|shares|sharing|shared)\b",
        "find": r"\b(find|finds|finding|found|pick|picks|picked|gather|gathers)\b",
        "discover": r"\b(discover|discovers|discovering|discovered|peep|peeps|peer|peers|peering)\b",
        "wave": r"\b(wave|waves|waving|waved)\b",
        "meet": r"\b(meet|meets|meeting|met|greet|greets|greeting)\b",
        "hop": r"\b(hop|hops|hopping|hopped|skip|skips|skipping)\b",
        "run": r"\b(run|runs|running|ran|trot|trots|trotting)\b",
        "hug": r"\b(hug|hugs|hugging|hugged|embrace|embraces)\b",
        "celebrate": r"\b(celebrate|celebrates|celebrating|cheer|cheers|cheering)\b",
        "sad": r"\b(sad|hungry|sigh|sighs|droop|droops|alone)\b",
        "reach": r"\b(reach|reaches|reaching|reached)\b",
        "leap": r"\b(leap|leaps|leaping|leaped|jump|jumps|jumping)\b",
        "strike": r"\b(strike|strikes|striking|struck|slash|slashes)\b",
        "inspect": r"\b(inspect|inspects|inspecting|examine|examines)\b",
    }

    # Search for matching action verb in scenario
    selected_pose = None
    for action_key, pose_desc in poses.items():
        if action_key == "default":
            continue
        pattern = VERB_FORMS.get(action_key, rf"\b{action_key}(?:s|es|ing|ed)?\b")
        if re.search(pattern, text):
            selected_pose = pose_desc
            break

    if not selected_pose:
        selected_pose = poses.get("default", "standing in a natural, expressive posture")

    return selected_pose


# ── 2b. Character Scale & Relative Height Registry ───────────────────────────

SPECIES_SCALE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "bear": {
        "species": "bear",
        "category": "giant",
        "scale_factor": 3.5,
        "height_m": 2.6,
        "build": "massive, broad-shouldered, towering presence",
        "description": "massive brown bear, towering and broad",
    },
    "rabbit": {
        "species": "rabbit",
        "category": "tiny",
        "scale_factor": 1.0,
        "height_m": 0.45,
        "build": "small, compact, fluffy build with long ears",
        "description": "tiny fluffy rabbit reaching only to knee height",
    },
    "bunny": {
        "species": "bunny",
        "category": "tiny",
        "scale_factor": 1.0,
        "height_m": 0.45,
        "build": "small, compact, fluffy build with long ears",
        "description": "tiny fluffy bunny reaching only to knee height",
    },
    "mouse": {
        "species": "mouse",
        "category": "micro",
        "scale_factor": 0.35,
        "height_m": 0.15,
        "build": "pocket-sized, diminutive stature",
        "description": "tiny pocket-sized mouse",
    },
    "fox": {
        "species": "fox",
        "category": "medium",
        "scale_factor": 1.8,
        "height_m": 0.9,
        "build": "slender medium canine stature",
        "description": "slender medium-sized fox",
    },
    "wolf": {
        "species": "wolf",
        "category": "medium_large",
        "scale_factor": 2.2,
        "height_m": 1.2,
        "build": "powerful medium-large canine stature",
        "description": "large powerful wolf",
    },
    "cat": {
        "species": "cat",
        "category": "small",
        "scale_factor": 0.9,
        "height_m": 0.4,
        "build": "small agile feline build",
        "description": "small agile cat",
    },
    "dog": {
        "species": "dog",
        "category": "medium",
        "scale_factor": 1.6,
        "height_m": 0.8,
        "build": "medium friendly canine build",
        "description": "medium friendly dog",
    },
    "human_adult": {
        "species": "human_adult",
        "category": "standard_adult",
        "scale_factor": 2.5,
        "height_m": 1.8,
        "build": "standard adult humanoid stature",
        "description": "tall adult humanoid",
    },
    "human_child": {
        "species": "human_child",
        "category": "child",
        "scale_factor": 1.4,
        "height_m": 1.1,
        "build": "small child humanoid build",
        "description": "small child reaching adult waist",
    },
}


def infer_character_scale(char_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infers the canonical species, relative scale factor, and physical stature of a character.
    Prevents visual continuity defects where large characters (e.g. Bear) shrink or small characters
    (e.g. Rabbit) balloon across shots.
    """
    if not isinstance(char_dict, dict):
        return {
            "species": "unknown",
            "scale_factor": 1.8,
            "category": "medium",
            "build": "medium build",
            "description": "medium character",
        }

    # If already explicitly declared in character sheet:
    if "scale_factor" in char_dict and "species" in char_dict:
        return {
            "species": char_dict["species"],
            "scale_factor": float(char_dict["scale_factor"]),
            "category": char_dict.get("scale_category", "custom"),
            "build": char_dict.get("build", "custom build"),
            "description": char_dict.get("scale_description", ""),
        }

    blob = " ".join([
        str(char_dict.get("name", "")),
        str(char_dict.get("description", "")),
        str(char_dict.get("face", "")),
        str(char_dict.get("hair", "")),
        str(char_dict.get("clothing", "")),
        str(char_dict.get("z_image_prompt", "")),
    ]).lower()

    # Match against species benchmarks
    for sp_key, sp_meta in SPECIES_SCALE_BENCHMARKS.items():
        if re.search(rf"\b{re.escape(sp_key)}\b", blob):
            return {
                "species": sp_key,
                "scale_factor": sp_meta["scale_factor"],
                "category": sp_meta["category"],
                "build": sp_meta["build"],
                "description": sp_meta["description"],
            }

    # Additional keyword matching
    if any(k in blob for k in ("bunny", "hare", "rabbit")):
        return SPECIES_SCALE_BENCHMARKS["rabbit"]
    if any(k in blob for k in ("bear", "grizzly", "polar")):
        return SPECIES_SCALE_BENCHMARKS["bear"]
    if any(k in blob for k in ("kid", "boy", "girl", "toddler", "child")):
        return SPECIES_SCALE_BENCHMARKS["human_child"]
    if any(k in blob for k in ("man", "woman", "detective", "knight", "warrior", "hero")):
        return SPECIES_SCALE_BENCHMARKS["human_adult"]

    return {
        "species": "medium_creature",
        "scale_factor": 1.8,
        "category": "medium",
        "build": "standard medium build",
        "description": "medium-stature character",
    }


def build_character_scale_anchor(characters: List[Dict[str, Any]]) -> str:
    """
    Builds an explicit, mathematically locked Scale Anchor clause for Qwen and Wan prompts.
    Anchors relative height ratios and physical stature across all scenes to prevent
    the 'bear going big and small' scale-shifting defect.
    """
    if not characters:
        return ""

    parsed_chars = []
    for c in characters:
        meta = infer_character_scale(c)
        name = c.get("name", meta["species"].capitalize())
        parsed_chars.append({"name": name, "meta": meta})

    if len(parsed_chars) == 1:
        c0 = parsed_chars[0]
        return (
            f"SCALE INVARIANT: {c0['name']} is a {c0['meta']['build']}. "
            f"Maintain this exact body size and framing proportion consistently across all shots."
        )

    # Sort characters by scale factor descending (largest to smallest)
    sorted_chars = sorted(parsed_chars, key=lambda x: x["meta"]["scale_factor"], reverse=True)
    big = sorted_chars[0]
    small = sorted_chars[-1]

    ratio = round(big["meta"]["scale_factor"] / max(0.1, small["meta"]["scale_factor"]), 1)

    if ratio >= 1.5:
        return (
            f"SCALE INVARIANT: {big['name']} is a massive {big['meta']['species']} ({ratio}x larger and taller than {small['name']}); "
            f"{small['name']} is a tiny {small['meta']['species']} reaching only up to {big['name']}'s knee or lower elbow. "
            f"{big['name']}'s wide broad body towers over {small['name']} in every shot. "
            f"NEVER shrink {big['name']} to small size and NEVER enlarge {small['name']} to bear size; "
            f"keep this strict {ratio}:1 relative height and volumetric scale locked across all scenes."
        )
    else:
        return (
            f"SCALE INVARIANT: {big['name']} and {small['name']} have consistent body proportions. "
            f"Maintain identical relative character scale across all shots."
        )


# ── 2c. Canonical Prop Taxonomy & Invariant Harmonizer ───────────────────────

PROP_CANONICAL_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "strawberries": {
        "canonical_name": "strawberries",
        "singular": "strawberry",
        "adjective": "sweet red",
        "visual_description": "ripe bright red strawberries with green leafy tops in a small wicker basket",
        "conflicting_variants": ["blueberries", "blueberry", "blackberries", "blackberry", "grapes", "dark berries", "blue berries"],
        "negative_guard": "blueberries, blackberries, dark berries, purple fruit, blue berries",
    },
    "blueberries": {
        "canonical_name": "blueberries",
        "singular": "blueberry",
        "adjective": "sweet blue",
        "visual_description": "round deep-blue blueberries with a light frosted dusting in a small wicker basket",
        "conflicting_variants": ["strawberries", "strawberry", "blackberries", "blackberry", "red berries", "red fruit"],
        "negative_guard": "strawberries, red berries, blackberries, red fruit",
    },
    "apples": {
        "canonical_name": "red apples",
        "singular": "apple",
        "adjective": "crisp red",
        "visual_description": "shiny red apples with green leaves",
        "conflicting_variants": ["oranges", "pears", "bananas"],
        "negative_guard": "oranges, pears, bananas",
    },
}


def audit_and_harmonize_props(story_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits and harmonizes prop naming and visual descriptions across the entire screenplay.
    Completely eliminates prop contradictions where narration mentions one fruit (e.g. blueberries)
    while characters are visibly eating another (e.g. strawberries).
    """
    data = dict(story_data or {})
    scenes = list(data.get("scenes", []))
    if not scenes:
        return data

    # 1. Detect which canonical prop family is dominant in the story
    full_corpus = " ".join([
        str(data.get("title", "")),
        str(data.get("theme", "")),
        str(data.get("logline", "")),
        " ".join(str(s.get("scenario", "")) for s in scenes),
        " ".join(str(s.get("narration_text", "")) for s in scenes),
        " ".join(str(s.get("qwen_scene_prompt", "")) for s in scenes),
        " ".join(str(s.get("qwen_end_scene_prompt", "")) for s in scenes),
    ]).lower()

    straw_score = len(re.findall(r"\b(strawberr\w+|red ber\w+)\b", full_corpus))
    blue_score = len(re.findall(r"\b(blueberr\w+|blue ber\w+)\b", full_corpus))

    canonical_key = None
    if straw_score > 0 or blue_score > 0:
        canonical_key = "strawberries" if straw_score >= blue_score else "blueberries"

    if not canonical_key:
        return data

    tax = PROP_CANONICAL_TAXONOMY[canonical_key]
    conflicts = tax["conflicting_variants"]
    conflict_pattern = re.compile(rf"\b({'|'.join(conflicts)})\b", re.IGNORECASE)

    # 2. Lock prop into world_bible
    world_bible = dict(data.get("world_bible", {}))
    world_bible["locked_prop"] = {
        "canonical_name": tax["canonical_name"],
        "visual_description": tax["visual_description"],
        "banned_variants": conflicts,
    }
    data["world_bible"] = world_bible

    # 3. Synchronize all scenes
    harmonized_scenes = []
    for s in scenes:
        sc = dict(s)
        # Harmonize narration text
        narr = sc.get("narration_text", "")
        if narr and conflict_pattern.search(narr):
            sc["narration_text"] = conflict_pattern.sub(tax["canonical_name"], narr)

        # Harmonize scenario
        scenario = sc.get("scenario", "")
        if scenario and conflict_pattern.search(scenario):
            sc["scenario"] = conflict_pattern.sub(tax["canonical_name"], scenario)

        # Harmonize start prompt
        q_start = sc.get("qwen_scene_prompt", "")
        if q_start and conflict_pattern.search(q_start):
            sc["qwen_scene_prompt"] = conflict_pattern.sub(f"{tax['adjective']} {tax['canonical_name']}", q_start)

        # Harmonize end prompt
        q_end = sc.get("qwen_end_scene_prompt", "")
        if q_end and conflict_pattern.search(q_end):
            sc["qwen_end_scene_prompt"] = conflict_pattern.sub(f"{tax['adjective']} {tax['canonical_name']}", q_end)

        # Harmonize wan motion prompt
        wan_p = sc.get("wan_motion_prompt", "")
        if wan_p and conflict_pattern.search(wan_p):
            sc["wan_motion_prompt"] = conflict_pattern.sub(tax["canonical_name"], wan_p)

        # Append negative guard to prevent alternate berry varieties from appearing in visuals
        dir_plan = dict(sc.get("director", {}))
        if dir_plan:
            neg = dir_plan.get("negative_prompt", "")
            if tax["negative_guard"] not in neg:
                dir_plan["negative_prompt"] = f"{neg}, {tax['negative_guard']}".strip(", ")
            sc["director"] = dir_plan

        harmonized_scenes.append(sc)

    data["scenes"] = harmonized_scenes
    return data


# ── 3. Multi-Pass Writers' Room Prompts ───────────────────────────────────────

def build_beat_sheet_prompt(
    premise: str,
    genre: str,
    num_scenes: int,
    target_duration: float,
) -> Tuple[str, str]:
    """
    Pass 1: Develops character motivations, core dramatic question, and Pixar-style causal beat sheet.
    """
    cfg = GENRE_CONFIGS.get(genre, GENRE_CONFIGS["kids"])

    system_prompt = (
        "You are an acclaimed animation Head of Story and master screenwriter.\n"
        "Your role is PASS 1: DEVELOP THE CAUSAL BEAT SHEET for a short animated film.\n\n"
        f"GENRE SPECIFICATION: {cfg['name']}\n{cfg['tone_prompt']}\n\n"
        "CORE DRAMATIC REQUIREMENTS:\n"
        f"1. Target Film Duration: {target_duration:.1f}s, exactly {num_scenes} continuous scenes (5.0s each).\n"
        "2. Character Arc: Introduce 1 or 2 distinct characters with clear, relatable wants and immediate physical obstacles.\n"
        "3. Pixar Causal Spine: Scene 1 establishes status quo and an inciting want; "
        "Scene 2 presents an obstacle connected by 'BUT'; Scene 3 pays off with a meaningful choice connected by 'THEREFORE'. "
        "NEVER connect scenes with 'AND THEN'.\n"
        "4. Show-Don't-Tell Emotion: Emotional beats must be expressible through observable physical actions "
        "(e.g. drooping ears, tummy rumbles, wide beaming smiles, offering a paw).\n"
        "5. Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "logline": "1-sentence captivating hook",\n'
        '  "theme": "The core moral or emotional theme (e.g. Sharing is caring)",\n'
        '  "protagonist_want": "What the main character visibly desires",\n'
        '  "emotional_climax": "The pivotal moment of empathy or discovery",\n'
        '  "payoff": "The warm, satisfying conclusion",\n'
        '  "beats": [\n'
        "    {\n"
        '      "scene_number": 1,\n'
        '      "dramatic_role": "Inciting Status Quo",\n'
        '      "causal_connector": "SETUP",\n'
        '      "core_action": "Observable 5s physical action",\n'
        '      "emotional_subtext": "What the characters feel (shown via facial/body cues)"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Story Premise: {premise}\n"
        f"Number of Scenes: {num_scenes}\n"
        f"Genre Target: {cfg['name']}\n"
        "Draft the causal beat sheet now."
    )

    return system_prompt, user_prompt


def build_scene_drafting_prompt(
    beat_sheet: Dict[str, Any],
    premise: str,
    genre: str,
    num_scenes: int,
    target_duration: float,
    art_style: str,
) -> Tuple[str, str]:
    """
    Pass 2: Converts the approved Beat Sheet into production-ready screenplay,
    World Bible, character sheets, and CEFR B1 storytelling voiceover.
    """
    cfg = GENRE_CONFIGS.get(genre, GENRE_CONFIGS["kids"])

    system_prompt = (
        "You are a master animation director and AI screenwriter translating a Story Beat Sheet into a production screenplay.\n"
        f"GENRE: {cfg['name']}\n{cfg['tone_prompt']}\n\n"
        "CRITICAL SCREENPLAY RULES:\n"
        "1. STRICT 5-SECOND ATOMICITY: Each scene is ONE continuous 5.0-second shot. Only 1 primary action + reaction.\n"
        "2. SHOW, DON'T TELL: Explicitly write physical expressions and body language in 'scenario' "
        "(e.g. 'Pip holds his empty stomach and looks up sadly; Barnaby notices, smiles warmly, and gently holds out the berry basket').\n"
        "3. CEFR B1 SPOKEN NARRATION (strictly 6 to 10 words per scene): "
        "Every scene has a 'narration_text' in simple, warm, conversational English. "
        "Use active sentences that describe what the audience sees. Link lines naturally like an engaging storybook.\n"
        "4. NARRATOR PERFORMANCE SCRIPT: Include 'narration_voice_style' ('warm_wonder', 'gentle_care', 'joyful_resolution') "
        "and a 'sync_trigger_word' (the exact action verb where visual climax occurs).\n"
        "5. CHARACTER UNIQUENESS & SPATIAL STABILITY: Exactly 1 instance of each character. In two-character shots, "
        "specify screen sides ('left' and 'right') so characters never overlap or spawn duplicates.\n"
        "6. CHARACTER SCALE & RELATIVE HEIGHT INVARIANTS: Define species and relative scale for each character. "
        "In scenes with multiple characters, explicitly lock their relative heights into qwen_scene_prompt and qwen_end_scene_prompt "
        "(e.g. 'Barnaby is a massive brown bear, 3.5x larger and taller than Pip; Pip reaches only to Barnaby\'s knee. "
        "Never shrink the bear or enlarge the rabbit; keep their relative volumetric scale identical across every shot').\n"
        "7. STRICT PROP INVARIANT LOCK: Any featured food, item, or fruit MUST have ONE locked canonical identity. "
        "Spoken narration and visual scene prompts MUST match 100% (e.g. if visuals show red strawberries, "
        "narration MUST talk about strawberries, NEVER blueberries).\n"
        "8. WORLD BIBLE: Define a cohesive 4-color palette, consistent time of day, weather, lighting invariants, and locked_prop.\n"
        "9. Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "title": "Short Film Title",\n'
        '  "art_style": "Target art style",\n'
        '  "theme": "Core theme",\n'
        '  "genre": "' + genre + '",\n'
        '  "world_bible": {\n'
        '    "palette": ["#Hex1", "#Hex2", "#Hex3", "#Hex4"],\n'
        '    "time_of_day": "e.g. Sunny morning",\n'
        '    "weather": "e.g. Gentle breeze, golden sunbeams",\n'
        '    "lighting_style": "e.g. Warm soft storybook rim lighting",\n'
        '    "art_style_invariants": "Locked style tokens",\n'
        '    "locked_prop": {\n'
        '      "canonical_name": "e.g. red strawberries",\n'
        '      "visual_description": "ripe bright red strawberries in a small wicker basket",\n'
        '      "banned_variants": ["blueberries", "blackberries"]\n'
        '    }\n'
        '  },\n'
        '  "characters": [\n'
        "    {\n"
        '      "id": "char_1",\n'
        '      "name": "Character Name",\n'
        '      "species": "bear / rabbit / human_adult",\n'
        '      "scale_category": "giant / tiny / medium",\n'
        '      "scale_factor": 3.5,\n'
        '      "description": "Personality and stature",\n'
        '      "face": "Eyes, expression, markings",\n'
        '      "hair": "Fur or hair texture/color",\n'
        '      "clothing": "Wardrobe, colors, scarf/hat",\n'
        '      "signature_details": "Held items or props",\n'
        '      "z_image_prompt": "Ultra-detailed character sheet prompt for Z-Image Turbo, front view, clean neutral studio background, 4k"\n'
        "    }\n"
        "  ],\n"
        '  "scenes": [\n'
        "    {\n"
        '      "scene_number": 1,\n'
        '      "title": "Scene Title",\n'
        '      "duration_seconds": 5.0,\n'
        '      "narrative_beat": "Beat role",\n'
        '      "causal_connector": "SETUP / THEREFORE / BUT",\n'
        '      "narration_text": "Spoken line in B1 English (6 to 10 words)",\n'
        '      "narration_voice_style": "warm_wonder / gentle_care / joyful_resolution",\n'
        '      "sync_trigger_word": "exact action verb in narration",\n'
        '      "character_placements": [\n'
        '        {"character_id": "char_1", "screen_side": "left", "facing": "right", "pose": "friendly posture"}\n'
        '      ],\n'
        '      "scenario": "Visual action beat with physical show-don\'t-tell cues",\n'
        '      "qwen_scene_prompt": "FIRST FRAME (t=0s) prompt: environment, lighting, starting character poses",\n'
        '      "qwen_end_scene_prompt": "LAST FRAME (t=5s) prompt: same environment, but visibly advanced ending pose",\n'
        '      "wan_motion_prompt": "Continuous 5-second motion description connecting start frame to end frame",\n'
        '      "camera_movement": "Cinematic camera move (e.g. Slow push-in / gentle orbital tracking)",\n'
        '      "character_ids": ["char_1"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Story Premise: {premise}\n"
        f"Art Style: {art_style}\n"
        f"Number of Scenes: {num_scenes}\n"
        f"Approved Beat Sheet:\n{json.dumps(beat_sheet, indent=2)}\n\n"
        "Draft the full production screenplay now."
    )

    return system_prompt, user_prompt


def build_story_editor_audit_prompt(
    story_data: Dict[str, Any],
    genre: str,
) -> Tuple[str, str]:
    """
    Pass 3: Rigorous Story Doctor Audit reviewing:
      - Causal necessity (Pixar THEREFORE/BUT test)
      - Emotional payoff & Show-Don't-Tell cues
      - Tone invariants (zero banned words)
      - CEFR B1 readability
    """
    cfg = GENRE_CONFIGS.get(genre, GENRE_CONFIGS["kids"])

    system_prompt = (
        "You are an exacting animation Story Doctor and Script Editor.\n"
        "Your task is to audit the drafted screenplay against professional Pixar/Studio Ghibli narrative standards.\n\n"
        f"TARGET GENRE: {cfg['name']}\n{cfg['tone_prompt']}\n\n"
        "EVALUATION CRITERIA (Score each from 1.0 to 10.0):\n"
        "1. causality: Are all scenes connected by logical cause-and-effect ('THEREFORE' / 'BUT')?\n"
        "2. emotional_arc: Does the film have an authentic heart-touching progression and satisfying payoff?\n"
        "3. visual_clarity_5s: Can each shot be cleanly animated in 5.0s without visual clutter or impossible teleportations?\n"
        "4. tone_coherence: Is the tone strictly authentic to the genre? (For kids stories: ZERO violence, shockwaves, or combat poses!)\n"
        "5. b1_readability: Is the narration natural, engaging B1 English (6-10 words, accessible vocabulary)?\n\n"
        "Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "overall_score": 1.0-10.0,\n'
        '  "subscores": {\n'
        '    "causality": 1.0-10.0,\n'
        '    "emotional_arc": 1.0-10.0,\n'
        '    "visual_clarity_5s": 1.0-10.0,\n'
        '    "tone_coherence": 1.0-10.0,\n'
        '    "b1_readability": 1.0-10.0\n'
        '  },\n'
        '  "passed": true,\n'
        '  "issues": ["list of specific flaws found, or empty list"],\n'
        '  "critique": "Constructive 2-sentence directorial feedback",\n'
        '  "repaired_scenes": null\n'
        "}"
    )

    user_prompt = (
        f"Story Title: {story_data.get('title')}\n"
        f"Genre: {cfg['name']}\n"
        f"Characters: {json.dumps(story_data.get('characters', []), indent=2)}\n"
        f"Scenes:\n{json.dumps(story_data.get('scenes', []), indent=2)}\n\n"
        "Conduct your story audit now."
    )

    return system_prompt, user_prompt


# ── 4. Narrator Performance Scripting & Heuristic Auditor ─────────────────────

def build_performance_script(
    scene_data: Dict[str, Any],
    genre: str = "kids",
) -> Dict[str, Any]:
    """
    Constructs an expressive Narrator Performance Script for a single scene, specifying:
      - text: Spoken narration text with natural breathing punctuation
      - emotion: Prosody emotion tag for Kokoro-82M (e.g. warm_wonder, joyful_resolution, tense_whisper)
      - speed: Kokoro prosody speed multiplier (e.g. 0.95-1.05)
      - pause_after_sec: Tail pause (default 0.35s - 0.45s)
      - emphasis_word: Key sync verb to emphasize
      - character_quote: Optional short character spoken quote (if present in screenplay)
    """
    text = (scene_data.get("narration_text") or "").strip()
    voice_style = scene_data.get("narration_voice_style")
    if not voice_style:
        if genre == "kids":
            voice_style = "warm_wonder" if scene_data.get("scene_number", 1) == 1 else "joyful_resolution"
        elif genre == "mystery":
            voice_style = "tense_whisper"
        elif genre == "adventure":
            voice_style = "urgent_rising"
        else:
            voice_style = "warm_storyteller"

    speed_map = {
        "warm_wonder": 0.96,
        "gentle_care": 0.95,
        "joyful_resolution": 0.98,
        "tense_whisper": 0.90,
        "urgent_rising": 1.06,
        "epic_climax": 1.08,
        "warm_storyteller": 1.00,
    }
    speed = speed_map.get(voice_style, 1.0)
    trigger = scene_data.get("sync_trigger_word") or ""
    quote = scene_data.get("character_quote") or None

    return {
        "text": text,
        "emotion": voice_style,
        "speed": round(speed, 2),
        "pause_after_sec": 0.40,
        "emphasis_word": trigger,
        "character_quote": quote,
    }


def audit_screenplay_heuristics(
    story_data: Dict[str, Any],
    genre: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic rule-based screenplay audit evaluating:
      1. Causality (THEREFORE / BUT)
      2. 5-Second Feasibility (word count <= 11, atomic action)
      3. Emotional Arc (expressive body language & physical cues)
      4. Tone Coherence (banned words check per genre)
      5. B1 Readability (Flesch score >= 65, accessible vocabulary)
      6. Visual Delta (Start vs End frame prompt divergence)
    """
    from narration_readability import check_b1_readability

    scenes = story_data.get("scenes", [])
    characters = story_data.get("characters", [])
    eff_genre = genre or story_data.get("genre") or detect_story_genre(
        story_data.get("theme", story_data.get("title", ""))
    )
    cfg = GENRE_CONFIGS.get(eff_genre, GENRE_CONFIGS["kids"])
    banned_words = cfg.get("banned_words", [])

    c_score = 10.0
    f_score = 10.0
    e_score = 10.0
    t_score = 10.0
    b_score = 10.0
    v_score = 10.0
    issues: List[str] = []

    proper_nouns = {c.get("name", "") for c in characters if isinstance(c, dict)}

    emotional_keywords = [
        "smile", "smiles", "smiling", "grin", "happy", "joy", "tears", "sad", "droop",
        "hug", "cuddle", "hop", "skip", "wonder", "laugh", "giggle", "relief", "cheer",
        "affection", "kindness", "gentle", "warm", "proud", "sparkle", "gaze", "curiosity"
    ]

    has_emotional_cue = False

    for i, s in enumerate(scenes):
        s_num = s.get("scene_number", i + 1)
        conn = str(s.get("causal_connector", "")).upper()
        if "AND THEN" in conn:
            c_score = max(2.0, c_score - 4.0)
            issues.append(f"Scene {s_num} uses forbidden connector 'AND THEN'.")
        elif i > 0 and not any(k in conn for k in ("BUT", "THEREFORE", "CONSEQUENCE", "RESULT", "SO")):
            c_score = max(5.0, c_score - 1.5)
            issues.append(f"Scene {s_num} lacks explicit 'THEREFORE' or 'BUT' causal link.")

        # Feasibility & Narration Word Count
        narr_text = s.get("narration_text", "")
        words = re.findall(r"\w+", narr_text)
        if len(words) > 11:
            f_score = max(4.0, f_score - 2.0)
            issues.append(f"Scene {s_num} narration has {len(words)} words (max is 11 for 5s clip).")
        elif len(words) < 4:
            f_score = max(6.0, f_score - 1.0)
            issues.append(f"Scene {s_num} narration is too short ({len(words)} words).")

        # B1 Readability
        b1_res = check_b1_readability(narr_text, proper_nouns=proper_nouns)
        if not b1_res.get("passed", True):
            b_score = max(2.0, b_score - 3.0 * len(b1_res.get("issues", [1])))
            issues.extend([f"Scene {s_num} narration: {iss}" for iss in b1_res.get("issues", [])])

        # Tone Coherence & Banned Words
        all_scene_text = (
            f"{s.get('scenario', '')} {s.get('qwen_scene_prompt', '')} "
            f"{s.get('qwen_end_scene_prompt', '')} {s.get('wan_motion_prompt', '')}"
        ).lower()
        found_banned = [w for w in banned_words if re.search(rf"\b{re.escape(w)}\b", all_scene_text)]
        if found_banned:
            t_score = max(1.0, t_score - 3.5 * len(found_banned))
            issues.append(f"Scene {s_num} violates {eff_genre} tone with banned word(s): {', '.join(found_banned)}.")

        # Emotional Arc Cues
        if any(w in all_scene_text for w in emotional_keywords):
            has_emotional_cue = True

        # Visual Delta
        p_start = (s.get("qwen_scene_prompt") or "").strip().lower()
        p_end = (s.get("qwen_end_scene_prompt") or "").strip().lower()
        if p_start and p_end and p_start == p_end:
            v_score = 1.0  # Completely frozen video is fatal
            issues.append(f"Scene {s_num} start and end prompts are identical (causes frozen video).")

    if not has_emotional_cue and eff_genre == "kids":
        e_score = 6.0
        issues.append("Story lacks explicit show-don't-tell physical emotional cues.")

    # 7. Prop Continuity & Narrative-Visual Alignment
    p_score = 10.0
    full_narr = " ".join(str(s.get("narration_text", "")).lower() for s in scenes)
    full_prompts = " ".join(
        f"{s.get('qwen_scene_prompt', '')} {s.get('qwen_end_scene_prompt', '')}".lower() for s in scenes
    )
    for tax_key, tax_info in PROP_CANONICAL_TAXONOMY.items():
        prompts_has_canon = bool(re.search(rf"\b{tax_info['canonical_name']}\b", full_prompts))
        for conflict_word in tax_info["conflicting_variants"]:
            if re.search(rf"\b{conflict_word}\b", full_narr) and prompts_has_canon:
                p_score = max(2.0, p_score - 4.0)
                issues.append(f"Prop contradiction: Narration mentions '{conflict_word}' while visual prompts specify '{tax_info['canonical_name']}'.")
            elif re.search(rf"\b{conflict_word}\b", full_prompts) and re.search(rf"\b{tax_info['canonical_name']}\b", full_narr):
                p_score = max(2.0, p_score - 4.0)
                issues.append(f"Prop contradiction: Visual prompts specify '{conflict_word}' while narration mentions '{tax_info['canonical_name']}'.")

    # 8. Multi-Character Relative Scale Continuity
    s_score = 10.0
    if len(characters) >= 2:
        for i, s in enumerate(scenes):
            c_ids = s.get("character_ids", [])
            if len(c_ids) >= 2:
                scene_text = f"{s.get('qwen_scene_prompt', '')} {s.get('qwen_end_scene_prompt', '')}"
                if "SCALE INVARIANT" not in scene_text and "larger" not in scene_text.lower() and "taller" not in scene_text.lower() and "knee" not in scene_text.lower():
                    s_score = max(5.0, s_score - 1.5)
                    issues.append(f"Scene {i+1} features multiple characters but lacks explicit relative character scale anchors.")

    overall = round((c_score + f_score + e_score + t_score + b_score + v_score + p_score + s_score) / 8.0, 1)
    # Severe flaws cap the overall score so unviable stories cannot proceed to GPU
    if t_score <= 3.0 or v_score <= 2.0 or c_score <= 3.0 or p_score <= 4.0 or s_score <= 4.0:
        overall = min(overall, 5.5)

    passed = overall >= 8.0 and not any(v < 6.0 for v in [c_score, f_score, t_score, b_score, v_score, p_score, s_score])
    critical_fail = overall < 6.0 or t_score < 5.0 or v_score < 4.0 or p_score < 5.0 or s_score < 5.0

    return {
        "passed": passed,
        "critical_fail": critical_fail,
        "overall_score": overall,
        "subscores": {
            "causality": round(c_score, 1),
            "feasibility_5s": round(f_score, 1),
            "emotional_arc": round(e_score, 1),
            "tone_coherence": round(t_score, 1),
            "b1_readability": round(b_score, 1),
            "visual_delta": round(v_score, 1),
            "prop_continuity": round(p_score, 1),
            "scale_continuity": round(s_score, 1),
        },
        "issues": issues,
        "critique": "; ".join(issues) if issues else "Screenplay strictly conforms to causal director grammar and tone invariants.",
    }
