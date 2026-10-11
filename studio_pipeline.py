"""
Studio Automated Multi-Agent Story & Video Production Pipeline
=============================================================
This module orchestrates the complete end-to-end studio pipeline:
1. Story & Character Breakdown (Gemma 4: 31B via Ollama Cloud)
2. Character Portrait Synthesis (Z-Image Turbo with locked identical seed)
3. Scene Keyframe Synthesis (Qwen Image Edit with multi-reference character transfer)
4. Video Clip Animation (Wan 2.2 with 4k+1 temporal constraints & CFG stability)
5. Video Assembly & Stitching (Seamless 30s final movie generation)
"""

import os
import io
import re
import json
import time
import pathlib
import tempfile
import subprocess
import urllib.request
from typing import Callable, Optional, Dict, List, Any, Tuple
from PIL import Image

import imageio_ffmpeg

from backend import (
    generate_z_image_turbo_remote,
    qwen_image_edit_remote,
    run_remote_inference,
    DEFAULT_NEGATIVE_PROMPT,
)

# ── Defaults & Constants ──────────────────────────────────────────────────────

DEFAULT_OLLAMA_API_KEY = "ca8516405cc74a238c9525b490331d3a.HmyaGAbsmLFyYN25P1K8_tG6"
DEFAULT_OLLAMA_MODEL = "gemma4:31b"
DEFAULT_OLLAMA_ENDPOINT = "https://ollama.com/v1/chat/completions"

# Aspect ratio presets for scene synthesis and video animation
ASPECT_RATIO_PRESETS = {
    "9:16 Vertical (Shorts/Reels)": {"qwen_w": 576, "qwen_h": 1024, "wan_w": 480, "wan_h": 832},
    "16:9 Landscape (Cinematic)":   {"qwen_w": 1024, "qwen_h": 576,  "wan_w": 832, "wan_h": 480},
    "1:1 Square (Social Media)":     {"qwen_w": 1024, "qwen_h": 1024, "wan_w": 640, "wan_h": 640},
}


class StudioPipeline:
    """
    Automated Multi-Agent Video Production Studio Coordinator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ):
        self.api_key = api_key or DEFAULT_OLLAMA_API_KEY
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.endpoint_url = endpoint_url or DEFAULT_OLLAMA_ENDPOINT

        if output_base_dir:
            self.base_dir = pathlib.Path(output_base_dir)
        else:
            self.base_dir = pathlib.Path(__file__).parent / "outputs" / "studio"

        self.char_dir = self.base_dir / "characters"
        self.scenes_dir = self.base_dir / "scenes"
        self.clips_dir = self.base_dir / "clips"
        self.movies_dir = self.base_dir / "movies"

        for d in [self.char_dir, self.scenes_dir, self.clips_dir, self.movies_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Story Generation & Character Extraction (Gemma 4: 31B)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_story_and_characters(
        self,
        prompt: str,
        num_scenes: int = 5,
        target_duration: float = 30.0,
        art_style: str = "2D vibrant cartoon animation",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Queries Ollama Cloud (gemma4:31b) to produce a structured short story with
        extracted character visual details and scene-by-scene director instructions.
        """
        if progress_cb:
            progress_cb(0.05, f"Consulting Gemma 4: 31B for story and character architecture...")

        scene_duration = round(target_duration / max(1, num_scenes), 1)

        system_prompt = (
            "You are a master animation director and AI scriptwriter.\n"
            f"The user wants a structured short movie ({target_duration} seconds total, broken into {num_scenes} scenes, ~{scene_duration}s each).\n"
            "You MUST extract exact character visual details (face, hair, clothing, signature traits) and generate dedicated "
            "prompts for: (1) Z-Image Turbo character portraits, (2) Qwen Image Edit scene keyframes, and (3) Wan 2.2 temporal camera motion.\n"
            "CRITICAL: Output ONLY valid JSON matching this schema without any surrounding markdown backticks or commentary:\n"
            "{\n"
            '  "title": "Short Film Title",\n'
            '  "art_style": "Visual style description",\n'
            '  "theme": "Moral or thematic premise",\n'
            '  "characters": [\n'
            "    {\n"
            '      "id": "char_1",\n'
            '      "name": "Character Name",\n'
            '      "description": "Personality and role summary",\n'
            '      "face": "Exact face features: shape, eyes, expression, facial markings",\n'
            '      "hair": "Hair/fur style, color, texture, length",\n'
            '      "clothing": "Complete outfit, fabric colors, style, shoes",\n'
            '      "signature_details": "Unique identifiable items, props, or accessories",\n'
            '      "z_image_prompt": "Ultra-detailed portrait prompt for Z-Image Turbo focusing on upper body and face against clean neutral studio background, 4k, crisp character model sheet",\n'
            '      "negative_prompt": "blurry, distorted, ugly, low quality, artifacts"\n'
            "    }\n"
            "  ],\n"
            '  "scenes": [\n'
            "    {\n"
            '      "scene_number": 1,\n'
            '      "title": "Scene Name",\n'
            f'      "duration_seconds": {scene_duration},\n'
            '      "scenario": "Story beat narrative description",\n'
            '      "qwen_scene_prompt": "Prompt for Qwen Image Edit describing the complete environment and character actions, incorporating referenced characters while maintaining locked identity",\n'
            '      "wan_motion_prompt": "Wan 2.2 motion prompt describing temporal action, lighting shifts, and camera kinematics, preserving subject appearance and facial identity without morphing",\n'
            '      "camera_movement": "Camera motion e.g. Slow Cinematic Push-In / Static Tripod Shot",\n'
            '      "character_ids": ["char_1"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_content = f"Story Premise: {prompt}\nTarget Style: {art_style}\nNumber of Scenes: {num_scenes}\nTotal Duration: {target_duration}s."

        req_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
            "max_tokens": 2500,
        }

        req = urllib.request.Request(
            self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(req_body).encode("utf-8"),
        )

        max_retries = 2
        raw_text = ""
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    raw_text = resp_data["choices"][0]["message"]["content"]
                    break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2.0)
                    continue
                raise RuntimeError(f"Ollama API request failed: {e}")

        parsed_data = self._parse_json_safely(raw_text)

        # Basic schema guarantees
        if "characters" not in parsed_data or not isinstance(parsed_data["characters"], list):
            parsed_data["characters"] = [
                {
                    "id": "char_1",
                    "name": "Hero",
                    "face": "Friendly and expressive",
                    "hair": "Clean styled",
                    "clothing": "Casual animation attire",
                    "signature_details": "Signature badge",
                    "z_image_prompt": f"Character portrait of {prompt}, clean neutral studio background, 4k",
                }
            ]
        if "scenes" not in parsed_data or not isinstance(parsed_data["scenes"], list):
            parsed_data["scenes"] = [
                {
                    "scene_number": 1,
                    "title": "Scene 1",
                    "duration_seconds": scene_duration,
                    "scenario": prompt,
                    "qwen_scene_prompt": prompt,
                    "wan_motion_prompt": "cinematic smooth motion, locked facial identity, photorealistic animation",
                    "camera_movement": "Slow Cinematic Push-In",
                    "character_ids": ["char_1"],
                }
            ]

        if progress_cb:
            progress_cb(0.20, f"Story generated: '{parsed_data.get('title', 'Studio Story')}' ({len(parsed_data.get('characters', []))} characters, {len(parsed_data.get('scenes', []))} scenes)")

        return parsed_data

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Character Portrait Synthesis via Z-Image Turbo (Locked Seed)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_character_portraits(
        self,
        characters: List[Dict[str, Any]],
        shared_seed: int = 42,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes reference portrait images for all extracted characters using
        Z-Image Turbo on Modal. Crucially, uses the SAME LOCKED SEED for all characters
        to establish a coherent base latent structure and visual harmony.
        """
        total = len(characters)
        updated_characters = []

        for idx, char in enumerate(characters):
            name = char.get("name", f"Char_{idx+1}")
            char_id = char.get("id", f"char_{idx+1}")
            char_prompt = char.get("z_image_prompt") or (
                f"Full portrait of {name}, {char.get('face', '')}, {char.get('hair', '')}, "
                f"{char.get('clothing', '')}, {char.get('signature_details', '')}, neutral studio background, 4k, crisp lighting"
            )

            if progress_cb:
                pct = 0.20 + (idx / max(1, total)) * 0.20
                progress_cb(pct, f"Synthesizing character reference {idx+1}/{total}: '{name}' (Seed {shared_seed})...")

            print(f"[StudioPipeline] Generating Z-Image Turbo portrait for '{name}' with locked seed {shared_seed}...")
            pil_img, local_path = generate_z_image_turbo_remote(
                prompt=char_prompt,
                seed=shared_seed,
                width=1024,
                height=1024,
                steps=9,
                save_dir=str(self.char_dir),
                filename_prefix=f"{char_id}_{self._sanitize_filename(name)}",
            )

            char_entry = dict(char)
            char_entry["portrait_pil"] = pil_img
            char_entry["portrait_path"] = local_path
            char_entry["portrait_seed"] = shared_seed
            updated_characters.append(char_entry)

        if progress_cb:
            progress_cb(0.40, f"All {total} character portraits synthesized with locked seed {shared_seed}.")

        return updated_characters

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Scene Frame Synthesis via Qwen Image Edit (Multi-Reference)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_scene_keyframes(
        self,
        scenes: List[Dict[str, Any]],
        characters: List[Dict[str, Any]],
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        shared_seed: int = 42,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes scene keyframes using Qwen Image Edit Rapid AIO.
        Transfers the exact character faces, clothing, and styles from Stage 2 by passing
        the Z-Image Turbo character portraits as multiple reference images into Qwen.
        """
        dims = ASPECT_RATIO_PRESETS.get(aspect_ratio, ASPECT_RATIO_PRESETS["9:16 Vertical (Shorts/Reels)"])
        qwen_w = dims["qwen_w"]
        qwen_h = dims["qwen_h"]

        # Build character lookup map
        char_map = {c.get("id"): c for c in characters}
        default_char = characters[0] if characters else None

        total = len(scenes)
        updated_scenes = []

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            title = scene.get("title", f"Scene {scene_num}")
            char_ids = scene.get("character_ids", [])

            # Gather reference portraits for characters in this scene
            ref_chars = [char_map[cid] for cid in char_ids if cid in char_map]
            if not ref_chars and default_char:
                ref_chars = [default_char]

            # Primary base image is Character 1 portrait
            base_pil = ref_chars[0].get("portrait_pil") if ref_chars else None
            # Additional reference images (Character 2, 3, etc.)
            additional_refs = [rc.get("portrait_pil") for rc in ref_chars[1:] if rc.get("portrait_pil")]

            scene_prompt = scene.get("qwen_scene_prompt") or scene.get("scenario") or "Scene background"
            # Ensure prompt anchors character reference identity
            char_names = ", ".join([rc.get("name", "character") for rc in ref_chars])
            refined_prompt = (
                f"{scene_prompt}, featuring {char_names} with identical faces, hairstyles, "
                f"and outfits matching the reference images, full frame cinematic composition, 4k"
            )

            if progress_cb:
                pct = 0.40 + (idx / max(1, total)) * 0.25
                progress_cb(pct, f"Synthesizing scene frame {idx+1}/{total}: '{title}' via Qwen Multi-Ref...")

            print(f"[StudioPipeline] Generating Qwen keyframe for Scene {scene_num} ({qwen_w}x{qwen_h})...")
            
            # Call Qwen remote
            results = qwen_image_edit_remote(
                base_image=base_pil,
                ref_images=additional_refs,
                prompt=refined_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=shared_seed + idx,
                steps=20,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_images=1,
                custom_width=qwen_w,
                custom_height=qwen_h,
            )

            keyframe_bytes = results[0]
            keyframe_pil = Image.open(io.BytesIO(keyframe_bytes)).convert("RGB")

            # Persist keyframe locally
            timestamp = int(time.time())
            filename = f"scene_{scene_num}_{self._sanitize_filename(title)}_{timestamp}.jpg"
            keyframe_path = self.scenes_dir / filename
            keyframe_pil.save(keyframe_path, format="JPEG", quality=95)
            print(f"[StudioPipeline] Saved Scene {scene_num} keyframe: {keyframe_path}")

            scene_entry = dict(scene)
            scene_entry["keyframe_pil"] = keyframe_pil
            scene_entry["keyframe_path"] = str(keyframe_path)
            updated_scenes.append(scene_entry)

        if progress_cb:
            progress_cb(0.65, f"All {total} scene keyframes synthesized with consistent character references.")

        return updated_scenes

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: Scene Video Animation via Wan 2.2
    # ─────────────────────────────────────────────────────────────────────────

    def generate_scene_videos(
        self,
        scenes: List[Dict[str, Any]],
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        Animates each scene keyframe into a video clip using Wan 2.2 on Modal.
        Adheres to 4k+1 frame rules (17, 33, 57, 81 frames), CFG 2.0 stability,
        and temporal camera motion prompts.
        """
        dims = ASPECT_RATIO_PRESETS.get(aspect_ratio, ASPECT_RATIO_PRESETS["9:16 Vertical (Shorts/Reels)"])
        wan_w = dims["wan_w"]
        wan_h = dims["wan_h"]

        total = len(scenes)
        clip_paths = []

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            title = scene.get("title", f"Scene {scene_num}")
            keyframe_pil = scene.get("keyframe_pil")
            duration_sec = float(scene.get("duration_seconds", 3.5))

            if keyframe_pil is None and "keyframe_path" in scene:
                keyframe_pil = Image.open(scene["keyframe_path"]).convert("RGB")

            # Resize to native Wan 2.2 dimensions (multiple of 16)
            resized_frame = keyframe_pil.resize((wan_w, wan_h), Image.Resampling.LANCZOS)

            # Frame calculation adhering to 4k+1 rule (17 to 81 frames)
            raw_frames = int(round(duration_sec * 16))
            k = max(4, min(20, round(raw_frames / 4)))
            clamped = 1 + (k * 4)

            motion_prompt = scene.get("wan_motion_prompt") or "cinematic natural motion, preserve subject facial identity"
            camera_movement = scene.get("camera_movement", "Slow Cinematic Push-In")
            full_prompt = f"{motion_prompt}, camera: {camera_movement}, locked facial identity, photorealistic fluid animation"

            if progress_cb:
                pct = 0.65 + (idx / max(1, total)) * 0.25
                progress_cb(pct, f"Animating Scene {idx+1}/{total}: '{title}' ({clamped} frames @ 16fps)...")

            print(f"[StudioPipeline] Animating Scene {scene_num} ({wan_w}x{wan_h}, {clamped} frames)...")

            # Wan 2.2 remote inference
            video_path = run_remote_inference(
                resized_image=resized_frame,
                processed_last_image=None,
                prompt=full_prompt,
                steps=25,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_frames=clamped,
                guidance_scale=2.0,  # Strict CFG for facial stability
                guidance_scale_2=1.0,
                current_seed=42 + idx,
                scheduler_name="UniPCMultistep",
                flow_shift=3.0,
                frame_multiplier=16,
                quality="High Quality",
                duration_seconds=duration_sec,
                safe_mode=True,
            )

            # Copy/rename into dedicated clips dir
            timestamp = int(time.time())
            clip_dest = self.clips_dir / f"clip_scene_{scene_num}_{self._sanitize_filename(title)}_{timestamp}.mp4"
            with open(video_path, "rb") as src, open(clip_dest, "wb") as dst:
                dst.write(src.read())

            print(f"[StudioPipeline] Saved Scene {scene_num} clip: {clip_dest}")
            clip_paths.append(str(clip_dest))
            scene["video_path"] = str(clip_dest)

        if progress_cb:
            progress_cb(0.90, f"All {total} scene clips animated successfully.")

        return clip_paths

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Video Assembly & Stitching (Seamless 30s Final Movie)
    # ─────────────────────────────────────────────────────────────────────────

    def stitch_clips(
        self,
        clip_paths: List[str],
        movie_title: str = "Studio Movie",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Concatenates individual scene clips into a single seamless movie using FFmpeg.
        """
        if not clip_paths:
            raise ValueError("No video clips provided for stitching.")

        if progress_cb:
            progress_cb(0.92, "Assembling and stitching scene clips into final movie...")

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        timestamp = int(time.time())
        clean_title = self._sanitize_filename(movie_title)
        output_movie_path = self.movies_dir / f"{clean_title}_{timestamp}.mp4"

        # Create FFmpeg concat demuxer file list
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f_list:
            concat_list_file = f_list.name
            for p in clip_paths:
                # Escape backslashes and single quotes for ffmpeg safe syntax
                safe_path = os.path.abspath(p).replace("\\", "/")
                f_list.write(f"file '{safe_path}'\n")

        try:
            # First try stream copy concatenation (instant, lossless)
            cmd_copy = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_file,
                "-c", "copy",
                str(output_movie_path),
            ]
            print(f"[StudioPipeline] Stitching clips via FFmpeg copy: {' '.join(cmd_copy)}")
            res = subprocess.run(cmd_copy, capture_output=True, text=True)

            # If copy failed (e.g. slight codec/fps variation), perform clean re-encoding
            if res.returncode != 0 or not output_movie_path.exists() or output_movie_path.stat().st_size == 0:
                print(f"[StudioPipeline] FFmpeg copy returned {res.returncode}, re-encoding concat...")
                cmd_reencode = [
                    ffmpeg_bin,
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_file,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    str(output_movie_path),
                ]
                res_re = subprocess.run(cmd_reencode, capture_output=True, text=True)
                if res_re.returncode != 0:
                    raise RuntimeError(f"FFmpeg stitching failed: {res_re.stderr}")
        finally:
            if os.path.exists(concat_list_file):
                os.remove(concat_list_file)

        print(f"[StudioPipeline] Successfully assembled movie ({output_movie_path.stat().st_size} bytes): {output_movie_path}")

        if progress_cb:
            progress_cb(1.0, f"Movie assembly complete: {output_movie_path.name}")

        return str(output_movie_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Full End-to-End Orchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        user_prompt: str,
        shared_seed: int = 42,
        num_scenes: int = 5,
        target_duration: float = 30.0,
        aspect_ratio: str = "9:16 Vertical (Shorts/Reels)",
        art_style: str = "2D vibrant cartoon animation",
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes all 5 stages of the Studio pipeline end-to-end.
        """
        # Stage 1: Story & Character Extraction
        story_data = self.generate_story_and_characters(
            prompt=user_prompt,
            num_scenes=num_scenes,
            target_duration=target_duration,
            art_style=art_style,
            progress_cb=progress_cb,
        )

        # Stage 2: Character Portrait Generation (Z-Image Turbo with locked seed)
        characters = self.generate_character_portraits(
            characters=story_data.get("characters", []),
            shared_seed=shared_seed,
            progress_cb=progress_cb,
        )
        story_data["characters"] = characters

        # Stage 3: Scene Keyframe Generation (Qwen Multi-Reference)
        scenes = self.generate_scene_keyframes(
            scenes=story_data.get("scenes", []),
            characters=characters,
            aspect_ratio=aspect_ratio,
            shared_seed=shared_seed,
            progress_cb=progress_cb,
        )
        story_data["scenes"] = scenes

        # Stage 4: Scene Video Animation (Wan 2.2)
        clip_paths = self.generate_scene_videos(
            scenes=scenes,
            aspect_ratio=aspect_ratio,
            progress_cb=progress_cb,
        )

        # Stage 5: Movie Stitching
        final_movie = self.stitch_clips(
            clip_paths=clip_paths,
            movie_title=story_data.get("title", "Studio_Movie"),
            progress_cb=progress_cb,
        )
        story_data["final_movie_path"] = final_movie
        story_data["clip_paths"] = clip_paths

        return story_data

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_safely(raw_text: str) -> Dict[str, Any]:
        clean = raw_text.strip()
        # Remove markdown code blocks if wrapped
        if "```json" in clean:
            clean = clean.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in clean:
            clean = clean.split("```", 1)[1].split("```", 1)[0]
        clean = clean.strip()

        # Find outer JSON braces
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start : end + 1]

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            # Fallback regex extraction if trailing comma or syntax error
            clean_fixed = re.sub(r",\s*([\]}])", r"\1", clean)
            try:
                return json.loads(clean_fixed)
            except Exception:
                raise ValueError(f"Could not parse valid JSON from Gemma response: {e}\nRaw: {raw_text[:300]}")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        s = re.sub(r"[^\w\-_]", "_", name)
        return re.sub(r"__+", "_", s).strip("_")[:40]
