"""
sprite_cutter.py - Universal 2D Game Sprite Extraction & Sprite Sheet Packer
Supports:
  1. Video frame extraction at game-engine target frame rates (8, 12, 16, 24 FPS)
  2. Background removal & alpha transparency (auto-detect corner chroma / floodfill / HSV color distance)
  3. Dynamic union bounding box cropping (keeps character centered without frame jitter)
  4. Texture atlas sprite sheet packing (transparent PNG)
  5. Standard engine JSON metadata export (compatible with Unity, Godot, Unreal, Pygame)
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter


def auto_detect_background_color(img_np: np.ndarray) -> np.ndarray:
    """
    Samples the outer borders and corners of an image (RGB) to determine the dominant background color.
    Detects pure white studio backgrounds (#FFFFFF) and green chroma key (#00FF00).
    """
    h, w = img_np.shape[:2]
    corner_size = max(2, min(12, min(h, w) // 10))
    corners = [
        img_np[0:corner_size, 0:corner_size],                    # Top-Left
        img_np[0:corner_size, w - corner_size : w],              # Top-Right
        img_np[h - corner_size : h, 0:corner_size],              # Bottom-Left
        img_np[h - corner_size : h, w - corner_size : w],        # Bottom-Right
        img_np[0:2, :],                                          # Top Border strip
        img_np[h - 2 : h, :],                                    # Bottom Border strip
        img_np[:, 0:2],                                          # Left Border strip
        img_np[:, w - 2 : w],                                    # Right Border strip
    ]
    border_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_median = np.median(border_pixels, axis=0).astype(np.uint8)

    # If samples are high-luminance white/near-white (RGB >= 230), snap to pure white #FFFFFF
    if np.mean(bg_median) >= 230 or np.all(bg_median >= 220):
        return np.array([255, 255, 255], dtype=np.uint8)

    # If samples are dominant green chroma
    if bg_median[1] > 175 and bg_median[0] < 120 and bg_median[2] < 120:
        return np.array([0, 255, 0], dtype=np.uint8)

    return bg_median


def remove_background_alpha(
    image: Union[Image.Image, np.ndarray],
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    tolerance: float = 38.0,
    feather_radius: int = 1,
    spill_suppression: bool = True,
    defringe: bool = True,
    preserve_vfx: bool = True,
) -> Image.Image:
    """
    Removes the solid white studio or chroma key background from an image, producing an RGBA
    image with smooth alpha, full VFX preservation (slashes, magic glow, particle trails, sparks),
    border flood-fill protection for white character clothing/eyes, and anti-halo defringing.
    """
    if isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
        img_np = np.array(img_pil)
    else:
        img_np = image.copy()
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        img_pil = Image.fromarray(img_np)

    h, w = img_np.shape[:2]

    # Detect or parse background color
    if bg_color is None:
        target_bg = auto_detect_background_color(img_np).astype(np.float32)
    else:
        target_bg = np.array(bg_color[:3], dtype=np.float32)

    is_white_bg = bool(np.mean(target_bg) >= 220 or np.all(target_bg >= 210))

    if is_white_bg:
        # ─────────────────────────────────────────────────────────────────────
        # High-Fidelity White Studio Background Matting with VFX Preservation
        # ─────────────────────────────────────────────────────────────────────
        # Compute departure from pure white:
        diff_rgb = np.linalg.norm(target_bg - img_np.astype(np.float32), axis=2)
        min_channel = np.min(img_np, axis=2).astype(np.float32)
        whiteness_delta = 255.0 - min_channel  # 0 for pure white, 255 for black or pure primary

        # Combined metric capturing both dark character shapes and vibrant pure-color VFX
        departure = np.maximum(diff_rgb * (255.0 / 441.67), whiteness_delta)

        # Thresholds:
        # - bg_tol: flat white studio background threshold
        # - solid_tol: threshold above which pixel is 100% opaque
        bg_tol = max(8.0, tolerance * 0.45)
        solid_tol = max(tolerance * 4.0, 180.0)

        # 1. Identify pure background candidates
        cand_bg = (departure <= bg_tol).astype(np.uint8)

        # 2. Flood fill from ALL 4 borders to find true exterior background
        flood_map = cand_bg.copy()
        mask_for_flood = np.zeros((h + 2, w + 2), np.uint8)

        for x in range(w):
            if flood_map[0, x] == 1:
                cv2.floodFill(flood_map, mask_for_flood, (x, 0), 2)
            if flood_map[h - 1, x] == 1:
                cv2.floodFill(flood_map, mask_for_flood, (x, h - 1), 2)
        for y in range(h):
            if flood_map[y, 0] == 1:
                cv2.floodFill(flood_map, mask_for_flood, (0, y), 2)
            if flood_map[y, w - 1] == 1:
                cv2.floodFill(flood_map, mask_for_flood, (w - 1, y), 2)

        is_ext_bg = (flood_map == 2)

        # 3. Calculate Alpha:
        # - Exterior background reached from border -> 0.0 alpha
        # - Interior white features (cand_bg is True but NOT reached by border flood fill) -> 255.0 (protected!)
        # - Foreground / VFX (cand_bg is False):
        final_alpha = np.full((h, w), 255.0, dtype=np.float32)
        final_alpha[is_ext_bg] = 0.0

        # Non-background pixels (cand_bg == 0)
        fg_mask = (cand_bg == 0)
        if preserve_vfx:
            # Semi-transparent VFX and soft anti-aliased edges
            soft_mask = fg_mask & (departure < solid_tol)
            proportional_alpha = (departure - bg_tol) / max(1.0, (solid_tol - bg_tol))
            final_alpha[soft_mask] = np.clip(proportional_alpha[soft_mask], 0.0, 1.0) * 255.0
            # Solid core
            solid_mask = fg_mask & (departure >= solid_tol)
            final_alpha[solid_mask] = 255.0
        else:
            final_alpha[fg_mask] = 255.0

        # Edge feathering if requested
        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            final_alpha = cv2.GaussianBlur(final_alpha, (ksize, ksize), 0)

        out_rgb = img_np.copy().astype(np.float32)

        # White defringing (anti-halo / un-premultiply against pure white)
        # Prevents milky/white halos around character silhouettes and glowing energy effects
        if defringe:
            alpha_norm = np.clip(final_alpha / 255.0, 0.0, 1.0)
            edge_or_vfx = (alpha_norm > 0.01) & (alpha_norm < 0.99)

            # Un-premultiply: C_fg = (C_obs - (1 - a) * C_bg) / a
            a_denom = np.maximum(alpha_norm, 0.12)
            for c in range(3):
                bg_val = target_bg[c]
                unpremul = (out_rgb[:, :, c] - (1.0 - alpha_norm) * bg_val) / a_denom
                out_rgb[:, :, c] = np.where(
                    edge_or_vfx,
                    np.clip(unpremul, 0.0, 255.0),
                    out_rgb[:, :, c]
                )

        final_rgb_u8 = np.clip(out_rgb, 0, 255).astype(np.uint8)
        final_alpha_u8 = np.clip(final_alpha, 0, 255).astype(np.uint8)
        rgba = np.dstack([final_rgb_u8, final_alpha_u8])
        return Image.fromarray(rgba, mode="RGBA")

    else:
        # ─────────────────────────────────────────────────────────────────────
        # Universal Chroma Key (Green Screen / Arbitrary Color) Matting
        # ─────────────────────────────────────────────────────────────────────
        # Compute Euclidean distance in RGB color space
        diff_rgb = np.linalg.norm(img_np.astype(np.float32) - target_bg, axis=2)

        # Also compute distance in HSV space to handle lighting gradients on green screen
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        target_bg_u8 = np.clip(target_bg, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        target_hsv = cv2.cvtColor(target_bg_u8, cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)

        hue_diff = np.abs(img_hsv[:, :, 0] - target_hsv[0])
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)

        is_chroma = bool(target_hsv[1] > 60.0)
        if is_chroma:
            match_mask = (hue_diff < 18.0) & (diff_rgb < (tolerance * 2.2))
        else:
            match_mask = diff_rgb < tolerance

        alpha = np.where(match_mask, 0, 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Flood fill from ALL border edges to protect interior green elements
        mask_for_flood = np.zeros((h + 2, w + 2), np.uint8)
        flood_img = (~alpha).copy()
        for x in range(w):
            if flood_img[0, x] > 0:
                cv2.floodFill(flood_img, mask_for_flood, (x, 0), 128)
            if flood_img[h - 1, x] > 0:
                cv2.floodFill(flood_img, mask_for_flood, (x, h - 1), 128)
        for y in range(h):
            if flood_img[y, 0] > 0:
                cv2.floodFill(flood_img, mask_for_flood, (0, y), 128)
            if flood_img[y, w - 1] > 0:
                cv2.floodFill(flood_img, mask_for_flood, (w - 1, y), 128)

        final_alpha = np.where(flood_img == 128, 0, 255).astype(np.uint8)

        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            final_alpha = cv2.GaussianBlur(final_alpha, (ksize, ksize), 0)

        out_rgb = img_np.copy()
        if spill_suppression and is_chroma and target_hsv[0] > 35 and target_hsv[0] < 85:
            edge_zone = (final_alpha > 0) & (final_alpha < 240)
            r = out_rgb[:, :, 0].astype(np.float32)
            g = out_rgb[:, :, 1].astype(np.float32)
            b = out_rgb[:, :, 2].astype(np.float32)
            avg_rb = (r + b) / 2.0
            g_spill = np.maximum(0.0, g - avg_rb)
            out_rgb[:, :, 1] = np.where(edge_zone, np.clip(g - g_spill * 0.7, 0, 255), g).astype(np.uint8)

        rgba = np.dstack([out_rgb, final_alpha])
        return Image.fromarray(rgba, mode="RGBA")


def extract_video_frames(video_path: str, target_fps: int = 12) -> List[Image.Image]:
    """
    Extracts sequential frames from a video file at the requested game frame rate (e.g. 8, 12, 16, 24 FPS).
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    target_fps = max(1, min(60, int(target_fps)))
    step = max(1.0, native_fps / float(target_fps))

    frames: List[Image.Image] = []
    current_float_idx = 0.0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx >= int(round(current_float_idx)):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            current_float_idx += step

        frame_idx += 1

    cap.release()
    if not frames and total_frames > 0:
        # Fallback: grab at least the first frame
        cap = cv2.VideoCapture(video_path)
        ret, f = cap.read()
        cap.release()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

    return frames


def find_union_bounding_box(frames_rgba: List[Image.Image], padding: int = 4) -> Tuple[int, int, int, int]:
    """
    Finds the union bounding box across all frames so the sprite canvas stays centered and stable.
    Returns (min_x, min_y, max_x, max_y).
    """
    if not frames_rgba:
        return (0, 0, 0, 0)

    w, h = frames_rgba[0].size
    global_min_x, global_min_y = w, h
    global_max_x, global_max_y = 0, 0

    has_content = False
    for frame in frames_rgba:
        bbox = frame.getbbox()  # Uses non-zero alpha
        if bbox:
            has_content = True
            global_min_x = min(global_min_x, bbox[0])
            global_min_y = min(global_min_y, bbox[1])
            global_max_x = max(global_max_x, bbox[2])
            global_max_y = max(global_max_y, bbox[3])

    if not has_content:
        return (0, 0, w, h)

    min_x = max(0, global_min_x - padding)
    min_y = max(0, global_min_y - padding)
    max_x = min(w, global_max_x + padding)
    max_y = min(h, global_max_y + padding)

    return (min_x, min_y, max_x, max_y)


def crop_frames_to_union_bbox(
    frames_rgba: List[Image.Image], padding: int = 4
) -> Tuple[List[Image.Image], Tuple[int, int, int, int]]:
    """
    Crops all frames in the animation to the shared union bounding box.
    """
    if not frames_rgba:
        return [], (0, 0, 0, 0)

    bbox = find_union_bounding_box(frames_rgba, padding=padding)
    cropped = [f.crop(bbox) for f in frames_rgba]
    return cropped, bbox


def pack_sprite_sheet(
    frames_rgba: List[Image.Image],
    max_cols: Optional[int] = None,
    border_padding: int = 2,
    shape_padding: int = 2,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Packs a list of RGBA sprite frames into a rectangular grid texture atlas (PNG with alpha).
    Returns (packed_image_pil, frame_metadata_list).
    """
    if not frames_rgba:
        blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        return blank, []

    num_frames = len(frames_rgba)
    frame_w, frame_h = frames_rgba[0].size

    # Compute optimal grid dimensions
    if max_cols is not None and max_cols > 0:
        cols = min(num_frames, int(max_cols))
    else:
        # Default to near-square aspect ratio
        cols = math.ceil(math.sqrt(num_frames))

    rows = math.ceil(num_frames / float(cols))

    sheet_w = border_padding * 2 + cols * frame_w + (cols - 1) * shape_padding
    sheet_h = border_padding * 2 + rows * frame_h + (rows - 1) * shape_padding

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    metadata: List[Dict[str, Any]] = []

    for idx, frame in enumerate(frames_rgba):
        col = idx % cols
        row = idx // cols
        x = border_padding + col * (frame_w + shape_padding)
        y = border_padding + row * (frame_h + shape_padding)

        sheet.paste(frame, (x, y), mask=frame)
        metadata.append({
            "frame_index": idx,
            "x": x,
            "y": y,
            "width": frame_w,
            "height": frame_h,
        })

    return sheet, metadata


def generate_engine_metadata(
    character_name: str,
    animation_name: str,
    genre: str,
    fps: int,
    is_loop: bool,
    sheet_size: Tuple[int, int],
    frame_coords: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Constructs an engine-ready JSON descriptor compatible with Unity, Godot AnimatedSprite2D,
    Unreal Paper2D, and Pygame.
    """
    frame_duration_ms = int(round(1000.0 / max(1, fps)))
    frames_list = []
    for idx, f in enumerate(frame_coords):
        w = f.get("width", f.get("w", 0))
        h = f.get("height", f.get("h", 0))
        x = f.get("x", 0)
        y = f.get("y", 0)
        f_idx = f.get("frame_index", f.get("frame", idx))
        frames_list.append({
            "frame": f_idx,
            "rect": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            },
            "duration_ms": frame_duration_ms,
            "unity": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            },
            "godot": {
                "region": [x, y, w, h],
            },
        })

    return {
        "character": character_name,
        "genre": genre,
        "animation": animation_name,
        "fps": int(fps),
        "is_loop": bool(is_loop),
        "total_frames": len(frame_coords),
        "sheet_dimensions": {
            "width": sheet_size[0],
            "height": sheet_size[1],
        },
        "frames": frames_list,
        "meta": {
            "app": "Wan 2D Sprites Creator",
            "version": "1.0",
            "format": "RGBA8888",
        },
    }


def process_video_to_sprite_sheet(
    video_path: str,
    character_name: str = "Character",
    animation_name: str = "idle",
    genre: str = "Action Platformer",
    target_fps: int = 12,
    is_loop: bool = True,
    bg_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    tolerance: float = 38.0,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete high-level pipeline: Video file -> Extracted Frames -> Alpha Matting ->
    Bounding Box Cropping -> Sprite Sheet Packing -> JSON Metadata.
    """
    # 1. Extract frames from video
    raw_frames = extract_video_frames(video_path, target_fps=target_fps)
    if not raw_frames:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    # 2. Alpha matting / background removal on every frame
    alpha_frames = [
        remove_background_alpha(f, bg_color=bg_color, tolerance=tolerance)
        for f in raw_frames
    ]

    # 3. Crop to union bounding box to eliminate empty canvas border and stabilize
    cropped_frames, union_bbox = crop_frames_to_union_bbox(alpha_frames, padding=4)

    # 4. Pack into sprite sheet texture atlas
    sheet_img, frame_metadata = pack_sprite_sheet(cropped_frames)

    # 5. Build engine JSON manifest
    metadata = generate_engine_metadata(
        character_name=character_name,
        animation_name=animation_name,
        genre=genre,
        fps=target_fps,
        is_loop=is_loop,
        sheet_size=sheet_img.size,
        frame_coords=frame_metadata,
    )

    # 6. Save files if output_dir is provided
    sheet_path = ""
    json_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_char = "".join(c for c in character_name if c.isalnum() or c in "_-").strip() or "char"
        safe_anim = "".join(c for c in animation_name if c.isalnum() or c in "_-").strip() or "anim"
        base_name = f"{safe_char}_{safe_anim}_{target_fps}fps"

        sheet_path = os.path.join(output_dir, f"{base_name}_sheet.png")
        sheet_img.save(sheet_path, format="PNG")

        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return {
        "sheet_image": sheet_img,
        "sheet_path": sheet_path,
        "json_path": json_path,
        "metadata": metadata,
        "frames_rgba": cropped_frames,
        "frame_count": len(cropped_frames),
        "union_bbox": union_bbox,
    }
