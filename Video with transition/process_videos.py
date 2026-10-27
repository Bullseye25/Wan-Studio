import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

# Supported video extensions
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

def get_ffmpeg_exe():
    # 1. System PATH
    exe = shutil.which("ffmpeg")
    if exe and os.path.exists(exe):
        return exe
    
    # 2. imageio_ffmpeg if installed
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    
    # 3. Known local paths in Windows
    possible_paths = [
        Path(r"C:\Users\Ammad Raza\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"),
        Path(sys.executable).parent / "ffmpeg.exe",
        Path(sys.executable).parent / "Scripts" / "ffmpeg.exe",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
            
    # Search python directory for imageio_ffmpeg
    try:
        site_packages = Path(sys.executable).parent / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
        if site_packages.exists():
            for f in site_packages.glob("ffmpeg*.exe"):
                return str(f)
    except Exception:
        pass

    return "ffmpeg"

def natural_sort_key(s):
    """Sort strings containing numbers in human natural order (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def get_video_info(ffmpeg_exe, file_path):
    """Probe video duration, dimensions, and fps using ffmpeg."""
    cmd = [ffmpeg_exe, "-hide_banner", "-i", str(file_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    output = p.stderr
    
    # Duration
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output)
    if dur_match:
        hours, mins, secs = dur_match.groups()
        duration = float(hours) * 3600 + float(mins) * 60 + float(secs)
    else:
        duration = 5.0 # fallback

    # Video stream dimensions and fps
    w, h, fps = 1080, 1920, 30.0 # defaults
    v_match = re.search(r"Stream.*Video:.*?(\d{2,5})x(\d{2,5})", output)
    if v_match:
        w, h = int(v_match.group(1)), int(v_match.group(2))
    
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", output)
    if fps_match:
        fps = float(fps_match.group(1))

    return duration, w, h, fps

def main():
    folder = Path(__file__).resolve().parent
    ffmpeg_exe = get_ffmpeg_exe()
    
    print("=" * 65)
    print(" Video Transition & Combiner Tool (Fade to Black - 1s) ")
    print("=" * 65)
    print(f"Working Directory: {folder}")
    print(f"Using FFmpeg     : {ffmpeg_exe}")

    # Gather video files (excluding outputs like combined_*.mp4)
    all_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS and not f.name.startswith("combined_")
    ]

    if not all_files:
        print("\n[!] No input video files found in this folder.")
        print("    Place your video files here and run this script again.")
        return

    # Natural sorting
    all_files.sort(key=lambda x: natural_sort_key(x.name))

    print(f"\n[+] Found {len(all_files)} video file(s):")
    for idx, f in enumerate(all_files, start=1):
        print(f"    {idx:02d}. {f.name}")

    if len(all_files) == 1:
        print("\n[!] Only 1 video found. Applying 1s fade-in/fade-out and removing audio...")
        v = all_files[0]
        dur, _, _, _ = get_video_info(ffmpeg_exe, v)
        out_name = folder / f"combined_{v.stem}_faded.mp4"
        fade_out_start = max(0.0, dur - 1.0)
        vf = (
            f"fade=t=in:st=0:d=1.0:color=black,"
            f"fade=t=out:st={fade_out_start:.2f}:d=1.0:color=black,"
            f"hqdn3d=1.5:1.5:4:4,unsharp=5:5:0.8:5:5:0.0"
        )
        cmd = [
            ffmpeg_exe, "-y", "-i", str(v),
            "-vf", vf,
            "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
            str(out_name)
        ]
        subprocess.run(cmd)
        print(f"\n[+] Done! Output: {out_name.name}")
        return

    # Probe all videos
    print("\n[+] Analyzing video files and calculating transition offsets...")
    durations = []
    widths = []
    heights = []
    fps_list = []

    for f in all_files:
        dur, w, h, fps = get_video_info(ffmpeg_exe, f)
        durations.append(dur)
        widths.append(w)
        heights.append(h)
        fps_list.append(fps)

    target_w = widths[0]
    target_h = heights[0]
    target_fps = fps_list[0] if fps_list[0] > 0 else 30.0
    trans_dur = 1.0 # 1 second black fade transition

    # Build FFmpeg complex filter graph:
    # 1. Standardize scale and fps across all videos: [0:v]scale=W:H:force_original_aspect_ratio=decrease,pad=...,fps=...[v0]
    # 2. Chain xfade with fadeblack transition:
    #    offset_0 = dur_0 - 1.0
    #    [v0][v1]xfade=transition=fadeblack:duration=1.0:offset=offset_0[x1]
    #    offset_1 = offset_0 + dur_1 - 1.0
    #    [x1][v2]xfade=transition=fadeblack:duration=1.0:offset=offset_1[x2] ...
    # 3. Enhance video quality: mild denoising + subtle edge contrast (hqdn3d + unsharp)

    filter_chains = []
    
    # Scaling and format normalization inputs
    for i in range(len(all_files)):
        pad_filter = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={target_fps},setsar=1"
        )
        filter_chains.append(f"[{i}:v]{pad_filter}[v{i}]")

    # Transitions
    current_offset = max(0.1, durations[0] - trans_dur)
    prev_v = "[v0]"

    for i in range(1, len(all_files)):
        next_v = f"[v{i}]"
        out_v = f"[x{i}]" if i < len(all_files) - 1 else "[xfinal]"
        filter_chains.append(
            f"{prev_v}{next_v}xfade=transition=fadeblack:duration={trans_dur:.2f}:offset={current_offset:.2f}{out_v}"
        )
        if i < len(all_files) - 1:
            current_offset += max(0.1, durations[i] - trans_dur)
            prev_v = out_v

    # Quality enhancer at final node:
    # hqdn3d (cleans compression noise) + unsharp (crisp detail without size blowup)
    filter_chains.append(f"[xfinal]hqdn3d=1.5:1.5:4:4,unsharp=5:5:0.8:5:5:0.0[outv]")

    filter_complex_str = ";".join(filter_chains)

    out_file = folder / f"combined_transition_{len(all_files)}_videos.mp4"

    cmd = [ffmpeg_exe, "-y"]
    for f in all_files:
        cmd.extend(["-i", str(f)])

    cmd.extend([
        "-filter_complex", filter_complex_str,
        "-map", "[outv]",
        "-an", # Remove all audio
        "-c:v", "libx264",
        "-preset", "slow", # Better compression efficiency -> keeps file size compact
        "-crf", "19",       # High quality perceptually lossless without increasing size
        "-pix_fmt", "yuv420p",
        str(out_file)
    ])

    print(f"\n[+] Processing and joining {len(all_files)} videos with 1s fade-to-black transitions...")
    print(f"[+] Audio will be completely removed.")
    print(f"[+] Video enhancement: hqdn3d denoiser + unsharp mask filter enabled.")
    print(f"[+] Output will be saved to: {out_file.name}")
    print("\nEncoding in progress, please wait...")

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    if res.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 65)
        print(f" SUCCESS! Combined video generated successfully! ")
        print("=" * 65)
        print(f"File : {out_file}")
        print(f"Size : {size_mb:.2f} MB")
    else:
        print("\n[!] FFmpeg failed with error:")
        print(res.stderr[-1000:] if res.stderr else "Unknown error")

if __name__ == "__main__":
    main()
