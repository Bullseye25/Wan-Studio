# Video Transition & Combiner Tool

## How to Use:
1. Place all your video clips (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) into this folder:
   `Video with transition`
2. Name them so their order is clear, for example:
   - `01.mp4`, `02.mp4`, `03.mp4`, `04.mp4`...
   - `1.mp4`, `2.mp4`, `3.mp4`, `4.mp4`...
   - `a.mp4`, `b.mp4`, `c.mp4`, `d.mp4`...
   - Or any names with numbers; the script uses natural numerical sorting so `1, 2, 10` are ordered logically.
3. Double-click **`run_transition_combiner.bat`** to run.

## Features:
- **Automatic Alignment**: Detects and sorts all video files in the folder in natural alphanumeric order.
- **Fade to Black (1 Second)**: Uses FFmpeg `fadeblack` transition (smooth fade-out to pure black and fade-in to next clip).
- **Audio Removal**: Strips all audio tracks cleanly (`-an`).
- **Quality Enhancement & Size Control**:
  - Uses `hqdn3d` spatial-temporal denoising to clean noise artifacts and compression grain.
  - Applies subtle `unsharp` mask sharpening for edge definition.
  - Encoded with `libx264 -preset slow -crf 19` for optimal visual fidelity with a compact file size.
- **Output**: Generates `combined_transition_<N>_videos.mp4` in the same folder.
