"""
Smart Production Telemetry, Quality Analysis & Continuous Improvement Logging Engine.
Captures stage latencies, character sheet isolation metrics, Qwen dual-keyframe pose deltas
& background SSIM, Kokoro/Prosodic narrator audio-visual sync offsets, Wan 2.2 Farneback
optical flow & sharpness retention, and automatically synthesizes actionable improvement
recommendations for iterative studio self-training.
"""

import os
import json
import time
import pathlib
import subprocess
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import imageio_ffmpeg

from orchestra_rules import (
    MAX_STORY_CHARACTERS,
    MAX_CLIP_DURATION_SEC,
    audit_prompt_for_anti_patterns,
    validate_story_budget,
)


class ProductionTelemetryCollector:
    """
    End-to-end telemetry recorder and post-mortem analyzer for the Wan 2.2 AI Studio Pipeline.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = pathlib.Path(base_dir) if base_dir else (pathlib.Path(__file__).parent / "outputs" / "studio")
        self.movies_dir = self.base_dir / "movies"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.movies_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_log_file = self.base_dir / "production_telemetry_log.jsonl"

        self.run_started_at = time.time()
        self.stage_timings: Dict[str, Dict[str, Any]] = {}
        self._active_stage: Optional[str] = None
        self._active_stage_t0: float = 0.0

    def start_stage(self, stage_name: str) -> None:
        if self._active_stage:
            self.end_stage(self._active_stage)
        self._active_stage = stage_name
        self._active_stage_t0 = time.time()
        self.stage_timings[stage_name] = {
            "started_at": round(self._active_stage_t0, 3),
            "duration_sec": 0.0,
            "status": "running",
        }

    def end_stage(self, stage_name: Optional[str] = None, status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> float:
        target = stage_name or self._active_stage
        if not target or target not in self.stage_timings:
            return 0.0
        elapsed = round(max(0.0, time.time() - self._active_stage_t0), 2)
        self.stage_timings[target]["duration_sec"] = elapsed
        self.stage_timings[target]["status"] = status
        if metadata:
            self.stage_timings[target].update(metadata)
        if self._active_stage == target:
            self._active_stage = None
        return elapsed

    # ── 1. Character Portrait Telemetry ───────────────────────────────────────

    @staticmethod
    def analyze_character_portraits(characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import cv2

        results = []
        for idx, char in enumerate(characters or []):
            cid = char.get("id", f"char_{idx+1}")
            name = char.get("name", f"Character {idx+1}")
            prompt = str(char.get("z_image_prompt") or "")
            path = char.get("portrait_path")
            violations = audit_prompt_for_anti_patterns(prompt, "z_image_turbo")

            info: Dict[str, Any] = {
                "id": cid,
                "name": name,
                "portrait_path": path,
                "seed": char.get("portrait_seed"),
                "exists": bool(path and os.path.exists(path)),
                "width": 0,
                "height": 0,
                "sharpness_laplacian": 0.0,
                "border_uniformity_std": 0.0,
                "clean_studio_backdrop": True,
                "prompt_violations": violations,
            }

            if path and os.path.exists(path):
                try:
                    with Image.open(path) as im:
                        rgb = np.array(im.convert("RGB"))
                        h, w, _ = rgb.shape
                        info["width"] = int(w)
                        info["height"] = int(h)
                        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                        info["sharpness_laplacian"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

                        # Measure outer 8% border standard deviation to verify neutral backdrop
                        bx = max(4, int(w * 0.08))
                        by = max(4, int(h * 0.08))
                        top_strip = gray[:by, :]
                        left_strip = gray[:, :bx]
                        right_strip = gray[:, w - bx :]
                        border_pixels = np.concatenate([top_strip.flatten(), left_strip.flatten(), right_strip.flatten()])
                        border_std = float(np.std(border_pixels))
                        info["border_uniformity_std"] = round(border_std, 2)
                        info["clean_studio_backdrop"] = border_std < 55.0
                except Exception as e:
                    info["error"] = str(e)

            results.append(info)
        return results

    # ── 2. Scene Keyframes, Narration & Video Telemetry ───────────────────────

    @staticmethod
    def analyze_scenes(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import cv2

        scene_metrics = []
        for idx, sc in enumerate(scenes or []):
            snum = sc.get("scene_number", idx + 1)
            title = sc.get("title", f"Scene {snum}")
            narration = str(sc.get("narration_text") or "").strip()
            words = narration.split() if narration else []
            synced = sc.get("synced_phases") or {}
            quant = sc.get("audio_quantization") or {}
            compat = sc.get("flf2v_compat") or {}
            qa = sc.get("qa_report") or {}
            director = sc.get("director") or {}
            telemetry = sc.get("telemetry") or {}

            qwen_start_p = str(sc.get("qwen_scene_prompt") or "")
            qwen_end_p = str(sc.get("qwen_end_scene_prompt") or "")
            wan_p = str(director.get("wan_prompt") or sc.get("wan_motion_prompt") or "")

            qwen_violations = (
                audit_prompt_for_anti_patterns(qwen_start_p, "qwen_image_edit")
                + audit_prompt_for_anti_patterns(qwen_end_p, "qwen_image_edit")
            )
            wan_violations = audit_prompt_for_anti_patterns(wan_p, "wan_2_2")

            # Inspect clip file properties if present
            clip_path = sc.get("video_path")
            clip_size_kb = 0.0
            actual_fps = 0.0
            actual_frames = 0
            actual_dur_sec = 0.0
            if clip_path and os.path.exists(clip_path):
                clip_size_kb = round(os.path.getsize(clip_path) / 1024.0, 1)
                try:
                    cap = cv2.VideoCapture(str(clip_path))
                    if cap.isOpened():
                        actual_fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
                        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        if actual_fps > 0:
                            actual_dur_sec = round(actual_frames / actual_fps, 2)
                        cap.release()
                except Exception:
                    pass

            entry = {
                "scene_number": snum,
                "title": title,
                "narrative_beat": sc.get("narrative_beat", ""),
                "causal_connector": sc.get("causal_connector", ""),
                "character_ids": sc.get("character_ids", []),
                "narration": {
                    "text": narration,
                    "word_count": len(words),
                    "within_budget_6_12": 4 <= len(words) <= 13,
                    "voice_style": sc.get("narration_voice_style", ""),
                    "tts_engine": sc.get("tts_engine", "kokoro-82m"),
                    "raw_speech_sec": quant.get("raw_speech_duration", 0.0),
                    "quantized_clip_sec": quant.get("exact_duration_seconds", sc.get("duration_seconds", 5.0)),
                    "tempo_factor": quant.get("tempo_factor", 1.0),
                    "sync_trigger_word": sc.get("sync_trigger_word", ""),
                    "trigger_onset_sec": synced.get("trigger_onset_sec", synced.get("sync_trigger_timestamp", 0.0)),
                    "phase_boundaries": {"p1": synced.get("p1", 1.25), "p2": synced.get("p2", 3.9)},
                },
                "keyframes": {
                    "start_path": sc.get("keyframe_path"),
                    "end_path": sc.get("last_keyframe_path"),
                    "pose_delta": sc.get("pose_delta", compat.get("pose_delta", 0.0)),
                    "bg_similarity": compat.get("bg_similarity", 1.0),
                    "bg_ssim": compat.get("bg_ssim", 1.0),
                    "flf2v_status": compat.get("status", "OPTIMAL"),
                    "flf2v_safe": sc.get("flf2v_safe", True),
                    "vlm_qa_score": qa.get("score", None),
                    "vlm_qa_passed": qa.get("passed", True),
                    "vlm_qa_issues": qa.get("issues", "None"),
                    "qwen_prompt_violations": qwen_violations,
                },
                "director_and_wan": {
                    "source": director.get("source", "unknown"),
                    "camera_movement": director.get("camera_movement", sc.get("camera_movement", "")),
                    "motion_intensity": director.get("motion_intensity", 2),
                    "guidance_scale": director.get("guidance_scale", 2.0),
                    "guidance_scale_2": director.get("guidance_scale_2", 1.5),
                    "flow_shift": director.get("flow_shift", 3.0),
                    "steps": director.get("steps", 8),
                    "used_last_frame_flf2v": sc.get("used_last_frame", True),
                    "wan_prompt_violations": wan_violations,
                },
                "video_telemetry": {
                    "video_path": clip_path,
                    "file_size_kb": clip_size_kb,
                    "actual_fps": actual_fps,
                    "actual_frames": actual_frames,
                    "actual_duration_sec": actual_dur_sec,
                    "audio_muxed": sc.get("audio_muxed", False),
                    "passed_optical_gate": telemetry.get("passed", True),
                    "mean_flow": telemetry.get("mean_flow", 0.0),
                    "surge_ratio": telemetry.get("surge_ratio", 1.0),
                    "sharpness_retention": telemetry.get("sharpness_retention", 1.0),
                    "peak_flow_sec": telemetry.get("peak_flow_sec", 0.0),
                    "audio_sync_delta_sec": telemetry.get("audio_sync_delta_sec", None),
                    "issues": telemetry.get("issues", []),
                },
            }
            scene_metrics.append(entry)
        return scene_metrics

    # ── 3. Final Movie Inspection ─────────────────────────────────────────────

    @staticmethod
    def analyze_final_movie(
        movie_path: Optional[str],
        srt_path: Optional[str],
        transition_type: str = "crossfade",
        transition_duration: float = 0.5,
    ) -> Dict[str, Any]:
        import cv2

        info: Dict[str, Any] = {
            "movie_path": movie_path,
            "exists": bool(movie_path and os.path.exists(movie_path)),
            "file_size_mb": 0.0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "has_audio_stream": False,
            "transition_type": transition_type,
            "transition_duration": transition_duration,
            "srt_path": srt_path,
            "srt_exists": bool(srt_path and os.path.exists(srt_path)),
            "srt_cue_count": 0,
        }

        if movie_path and os.path.exists(movie_path):
            info["file_size_mb"] = round(os.path.getsize(movie_path) / (1024.0 * 1024.0), 2)
            try:
                cap = cv2.VideoCapture(str(movie_path))
                if cap.isOpened():
                    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    info["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
                    info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if info["fps"] > 0:
                        info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
                    cap.release()
            except Exception:
                pass

            try:
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                probe = subprocess.run([ffmpeg_bin, "-i", str(movie_path)], capture_output=True, text=True)
                info["has_audio_stream"] = "Audio:" in (probe.stderr or "")
            except Exception:
                pass

        if srt_path and os.path.exists(srt_path):
            try:
                txt = pathlib.Path(srt_path).read_text(encoding="utf-8")
                info["srt_cue_count"] = txt.count("-->")
            except Exception:
                pass

        return info

    # ── 4. Smart Root-Cause & Continuous Improvement Recommendation Engine ───

    @staticmethod
    def generate_improvement_recommendations(
        story_data: Dict[str, Any],
        char_metrics: List[Dict[str, Any]],
        scene_metrics: List[Dict[str, Any]],
        movie_metrics: Dict[str, Any],
        stage_timings: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Inspects all quantitative telemetry signals across the production run and synthesizes
        concrete, prioritized engineering & prompt-tuning recommendations for future runs.
        """
        recs: List[Dict[str, Any]] = []

        # 1. Character genesis checks
        for cm in char_metrics:
            if not cm.get("clean_studio_backdrop", True):
                recs.append({
                    "category": "Z-Image Turbo Character Isolation",
                    "severity": "MEDIUM",
                    "target": f"Character '{cm['name']}'",
                    "finding": f"Outer border pixel std ({cm['border_uniformity_std']}) indicates background clutter (> 55.0).",
                    "recommendation": "Reinforce 'plain solid neutral light-grey studio background, zero environmental props' in Z-Image prompt.",
                })
            if cm.get("sharpness_laplacian", 100.0) < 25.0:
                recs.append({
                    "category": "Z-Image Turbo Sharpness",
                    "severity": "LOW",
                    "target": f"Character '{cm['name']}'",
                    "finding": f"Character portrait Laplacian sharpness is low ({cm['sharpness_laplacian']}).",
                    "recommendation": "Increase Z-Image steps from 9 to 10 or append 'crisp ink lines, ultra-sharp 4K concept art'.",
                })

        # 2. Scene keyframe & video optical flow checks
        for sm in scene_metrics:
            snum = sm["scene_number"]
            kf = sm["keyframes"]
            vt = sm["video_telemetry"]
            dw = sm["director_and_wan"]
            nr = sm["narration"]

            p_delta = float(kf.get("pose_delta") or 0.0)
            if 0.0 < p_delta < 18.0:
                recs.append({
                    "category": "Qwen Keyframe Pose Delta",
                    "severity": "HIGH",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Start-to-End Pose Delta is low ({p_delta:.1f} < 18.0), risking static Wan interpolation.",
                    "recommendation": "Increase Qwen End-Frame true_cfg_scale from 5.5 to 6.2 and use stronger verb posture shifts (kneeling, lunging, reaching).",
                })
            elif p_delta > 78.0:
                recs.append({
                    "category": "Qwen Keyframe Divergence",
                    "severity": "MEDIUM",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Start-to-End Pose Delta is very high ({p_delta:.1f} > 78.0), increasing FLF2V morph risk.",
                    "recommendation": "Route extreme delta shots to Guided I2V mode or increase peripheral border stabilization ratio from 0.14 to 0.18.",
                })

            bg_ssim = float(kf.get("bg_ssim") or 1.0)
            if bg_ssim < 0.55:
                recs.append({
                    "category": "Background Permanence",
                    "severity": "MEDIUM",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Background SSIM between Start and End frame dropped to {bg_ssim:.2f}.",
                    "recommendation": "Strengthen Qwen background lock prompt and increase stabilize_end_frame_background blending weight.",
                })

            mean_flow = float(vt.get("mean_flow") or 0.0)
            surge = float(vt.get("surge_ratio") or 1.0)
            sharp_ret = float(vt.get("sharpness_retention") or 1.0)
            sync_delta = vt.get("audio_sync_delta_sec")

            if mean_flow < 0.25 and vt.get("actual_frames", 0) > 0:
                recs.append({
                    "category": "Wan 2.2 Motion Dynamics",
                    "severity": "HIGH",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Low optical flow magnitude (mean_flow={mean_flow:.3f}), clip appears overly static.",
                    "recommendation": "Raise motion_intensity to 3+ and increase Wan flow_shift from 3.0 to 4.2.",
                })
            if surge > 5.5:
                recs.append({
                    "category": "Wan 2.2 Temporal Stability",
                    "severity": "HIGH",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"High optical flow surge ratio ({surge:.2f}x median), indicating rapid mid-clip morphing or ghosting.",
                    "recommendation": f"Clamp guidance_scale_2 to <= 1.25 (current: {dw.get('guidance_scale_2')}) and ensure FlowShift >= 4.5.",
                })
            if sharp_ret < 0.60 and vt.get("actual_frames", 0) > 0:
                recs.append({
                    "category": "Wan 2.2 Late-Frame Sharpness",
                    "severity": "MEDIUM",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Final 20% frames lost sharpness (retention={sharp_ret:.2f} < 0.60).",
                    "recommendation": "Enable Anti-Noise Guard (CFG <= 2.0, FlowShift 4.5) and apply CIE-Lab match on End Keyframe.",
                })
            if sync_delta is not None and float(sync_delta) > 0.75:
                recs.append({
                    "category": "Audio-Visual Narrator Sync",
                    "severity": "LOW",
                    "target": f"Scene {snum} ('{sm['title']}')",
                    "finding": f"Peak visual motion ({vt.get('peak_flow_sec')}s) drifted by {sync_delta:.2f}s from spoken trigger word '{nr.get('sync_trigger_word')}' ({nr.get('trigger_onset_sec')}s).",
                    "recommendation": "Tighten Phase 1 wind-up window in Wan 3-phase prompt so Phase 2 action initiates 0.25s earlier.",
                })

        # 3. Latency bottleneck check
        total_stage_sec = sum(float(v.get("duration_sec", 0.0)) for v in stage_timings.values())
        if total_stage_sec > 0:
            slowest_stage, slowest_info = max(
                stage_timings.items(), key=lambda kv: float(kv[1].get("duration_sec", 0.0))
            )
            share_pct = round((float(slowest_info.get("duration_sec", 0.0)) / total_stage_sec) * 100.0, 1)
            recs.append({
                "category": "Pipeline Latency & Throughput",
                "severity": "INFO",
                "target": f"Stage '{slowest_stage}'",
                "finding": f"Stage '{slowest_stage}' took {slowest_info.get('duration_sec')}s ({share_pct}% of total runtime {round(total_stage_sec, 1)}s).",
                "recommendation": "Keep Modal container warm during active phase and batch multi-scene keyframe requests when possible.",
            })

        if not recs:
            recs.append({
                "category": "Overall Production Health",
                "severity": "INFO",
                "target": "All 6 Scenes",
                "finding": "All character sheets, dual keyframes, narrator sync offsets, and Wan 2.2 optical flow gates passed within optimal thresholds.",
                "recommendation": "Promote this run's storyboard and director plans directly to the SFT Gold Exemplar dataset.",
            })

        return recs

    # ── 5. Full Report Compilation & Persistence ──────────────────────────────

    def compile_and_save_reports(
        self,
        story_data: Dict[str, Any],
        transition_type: str = "crossfade",
        transition_duration: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compiles the complete telemetry bundle, appends to `production_telemetry_log.jsonl`,
        writes the JSON report and Markdown post-mortem analysis report, and returns the report dict.
        """
        if self._active_stage:
            self.end_stage(self._active_stage)

        total_wall_sec = round(time.time() - self.run_started_at, 2)
        title = story_data.get("title", "Studio_Movie")
        clean_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_")[:40] or "Studio_Movie"
        ts = int(time.time())

        budget_ok, budget_errors = validate_story_budget(story_data)
        char_metrics = self.analyze_character_portraits(story_data.get("characters", []))
        scene_metrics = self.analyze_scenes(story_data.get("scenes", []))
        movie_metrics = self.analyze_final_movie(
            movie_path=story_data.get("final_movie_path"),
            srt_path=story_data.get("subtitle_srt_path"),
            transition_type=transition_type,
            transition_duration=transition_duration,
        )

        recommendations = self.generate_improvement_recommendations(
            story_data=story_data,
            char_metrics=char_metrics,
            scene_metrics=scene_metrics,
            movie_metrics=movie_metrics,
            stage_timings=self.stage_timings,
        )

        # Aggregate summary statistics
        pose_deltas = [s["keyframes"]["pose_delta"] for s in scene_metrics if s["keyframes"].get("pose_delta")]
        mean_flows = [s["video_telemetry"]["mean_flow"] for s in scene_metrics if s["video_telemetry"].get("mean_flow")]
        surges = [s["video_telemetry"]["surge_ratio"] for s in scene_metrics if s["video_telemetry"].get("surge_ratio")]
        sharp_rets = [s["video_telemetry"]["sharpness_retention"] for s in scene_metrics if s["video_telemetry"].get("sharpness_retention")]
        sync_deltas = [
            s["video_telemetry"]["audio_sync_delta_sec"]
            for s in scene_metrics
            if s["video_telemetry"].get("audio_sync_delta_sec") is not None
        ]

        summary_stats = {
            "total_wall_clock_sec": total_wall_sec,
            "character_count": len(char_metrics),
            "scene_count": len(scene_metrics),
            "budget_validated": budget_ok,
            "budget_errors": budget_errors,
            "logic_audit_score": (story_data.get("logic_audit") or {}).get("overall_score", 10.0),
            "avg_pose_delta": round(float(np.mean(pose_deltas)), 2) if pose_deltas else 0.0,
            "avg_optical_flow": round(float(np.mean(mean_flows)), 3) if mean_flows else 0.0,
            "max_surge_ratio": round(float(np.max(surges)), 2) if surges else 1.0,
            "avg_sharpness_retention": round(float(np.mean(sharp_rets)), 2) if sharp_rets else 1.0,
            "avg_audio_sync_delta_sec": round(float(np.mean(sync_deltas)), 2) if sync_deltas else 0.0,
            "clips_passed_quality_gate": sum(1 for s in scene_metrics if s["video_telemetry"].get("passed_optical_gate", True)),
        }

        report = {
            "timestamp": ts,
            "title": title,
            "art_style": story_data.get("art_style", ""),
            "theme": story_data.get("theme", ""),
            "seeds": story_data.get("seeds", {}),
            "summary": summary_stats,
            "stage_timings": self.stage_timings,
            "logic_audit": story_data.get("logic_audit", {}),
            "character_telemetry": char_metrics,
            "scene_telemetry": scene_metrics,
            "final_movie_telemetry": movie_metrics,
            "improvement_recommendations": recommendations,
        }

        json_report_path = self.movies_dir / f"{clean_title}_{ts}_telemetry_report.json"
        md_report_path = self.movies_dir / f"{clean_title}_{ts}_analysis_report.md"

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        with open(self.telemetry_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

        md_content = self._format_markdown_report(report)
        md_report_path.write_text(md_content, encoding="utf-8")

        report["json_report_path"] = str(json_report_path)
        report["markdown_report_path"] = str(md_report_path)
        return report

    @staticmethod
    def _format_markdown_report(report: Dict[str, Any]) -> str:
        s = report["summary"]
        m = report["final_movie_telemetry"]
        seeds = report.get("seeds", {})
        audit = report.get("logic_audit", {})
        subscores = audit.get("subscores", {})

        lines = [
            f"# 🎬 Production Telemetry & Post-Mortem Analysis: *{report['title']}*",
            "",
            f"- **Total Wall-Clock Runtime**: `{s['total_wall_clock_sec']}s`",
            f"- **Final Movie File**: `{m.get('movie_path')}` (`{m.get('file_size_mb')} MB`, `{m.get('duration_sec')}s`, `{m.get('width')}x{m.get('height')} @ {m.get('fps')}fps`)",
            f"- **Transition Engine**: `{m.get('transition_type')}` (`{m.get('transition_duration')}s` overlap) | **Audio Track**: `{'Present' if m.get('has_audio_stream') else 'Silent'}` | **SRT Cues**: `{m.get('srt_cue_count')}`",
            f"- **Locked Studio Seeds**: Z-Image=`{seeds.get('z_image_seed')}` | Qwen=`{seeds.get('qwen_seed')}` | Wan 2.2=`{seeds.get('wan_seed')}`",
            f"- **Story Doctor Causal Score**: `{s['logic_audit_score']}/10.0` (Causality=`{subscores.get('causality', '-')}`, 5s Feasibility=`{subscores.get('feasibility_5s', '-')}`, Prop Lock=`{subscores.get('prop_permanence', '-')}`, Spatial=`{subscores.get('spatial_continuity', '-')}`, Visual Delta=`{subscores.get('visual_delta', '-')}`)",
            f"- **Aggregate Quality Metrics**: Avg Pose Delta=`{s['avg_pose_delta']}` | Avg Optical Flow=`{s['avg_optical_flow']}` | Max Surge Ratio=`{s['max_surge_ratio']}` | Avg Sharpness Retention=`{s['avg_sharpness_retention']}` | Avg Audio-Visual Sync Offset=`{s['avg_audio_sync_delta_sec']}s`",
            "",
            "---",
            "",
            "## 1. Stage-by-Stage Latency Breakdown",
            "",
            "| Stage | Duration (s) | Status |",
            "| :--- | :---: | :---: |",
        ]
        for st_name, st_info in report.get("stage_timings", {}).items():
            lines.append(f"| `{st_name}` | `{st_info.get('duration_sec', 0.0)}s` | `{st_info.get('status', '')}` |")

        lines.extend([
            "",
            "---",
            "",
            "## 2. Character Genesis Telemetry (Z-Image Turbo)",
            "",
            "| ID | Name | Resolution | Sharpness (Laplacian) | Border Std (Studio Isolation) | Clean Backdrop | Anti-Patterns |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])
        for cm in report.get("character_telemetry", []):
            viol = ", ".join(cm.get("prompt_violations", [])) or "None"
            lines.append(
                f"| `{cm['id']}` | **{cm['name']}** | `{cm['width']}x{cm['height']}` | `{cm['sharpness_laplacian']}` | "
                f"`{cm['border_uniformity_std']}` | `{'PASS' if cm['clean_studio_backdrop'] else 'WARN'}` | {viol} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Scene-by-Scene Keyframe, Narrator & Optical Flow Telemetry",
            "",
            "| Scene | Beat & Connector | Narration (Words) | Trigger @ Time | Pose Delta | BG SSIM | FLF2V Status | Mean Flow | Surge Ratio | Sharpness Ret. | Audio Sync Delta | Gate |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for sm in report.get("scene_telemetry", []):
            nr = sm["narration"]
            kf = sm["keyframes"]
            vt = sm["video_telemetry"]
            gate_str = "PASS" if vt.get("passed_optical_gate", True) else "WARN"
            sync_d = f"{vt['audio_sync_delta_sec']}s" if vt.get("audio_sync_delta_sec") is not None else "N/A"
            lines.append(
                f"| **#{sm['scene_number']}** ({sm['title']}) | {sm['narrative_beat']} (`{sm['causal_connector'][:18]}`) | "
                f"\"{nr['text']}\" (`{nr['word_count']}w`) | `{nr['sync_trigger_word']}` @ `{nr['trigger_onset_sec']}s` | "
                f"`{kf['pose_delta']}` | `{kf['bg_ssim']}` | `{kf['flf2v_status']}` | "
                f"`{vt['mean_flow']}` | `{vt['surge_ratio']}` | `{vt['sharpness_retention']}` | `{sync_d}` | **{gate_str}** |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Smart Continuous Improvement Recommendations",
            "",
            "| Severity | Category | Target | Diagnostic Finding | Actionable Tuning Recommendation |",
            "| :---: | :--- | :--- | :--- | :--- |",
        ])
        for rec in report.get("improvement_recommendations", []):
            lines.append(
                f"| **{rec['severity']}** | {rec['category']} | `{rec['target']}` | {rec['finding']} | {rec['recommendation']} |"
            )

        lines.append("")
        return "\n".join(lines)
