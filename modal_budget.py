"""
Modal Container Budget Controller & Enforcer.
============================================
Guarantees that active cloud GPU/CPU containers on Modal never exceed the hard limit of 8,
prevents excessive costs, and calculates precise per-stage dollar cost estimates.
"""

import os
import re
import pathlib
import threading
from typing import Dict, Any, Tuple

# Hard global limit requested: no more than 8 simultaneous containers on Modal
MAX_GLOBAL_MODAL_CONTAINERS = 8

# Hourly & per-second pricing estimates (approx standard Modal rates)
HOURLY_RATES = {
    "A100-80GB": 3.67,       # Wan 2.2 FLF2V
    "L40S": 1.50,            # Qwen Image Edit & Z-Image Turbo
    "CPU": 0.15,             # Kokoro-82M Narrator
}

PER_SECOND_RATES = {
    k: v / 3600.0 for k, v in HOURLY_RATES.items()
}


def get_configured_modal_containers(base_dir: str = None) -> Dict[str, int]:
    """
    Parses the deploy_*.py files to extract the configured max_containers for each service.
    """
    if base_dir is None:
        root_dir = pathlib.Path(__file__).parent
    else:
        root_dir = pathlib.Path(base_dir)

    deploy_files = {
        "wan": root_dir / "deploy_wan.py",
        "qwen": root_dir / "deploy_qwen.py",
        "z_image": root_dir / "deploy_z_image.py",
        "narrator": root_dir / "deploy_narrator.py",
    }

    results = {}
    for name, path in deploy_files.items():
        if path.exists():
            content = path.read_text(encoding="utf-8")
            m = re.search(r"max_containers\s*=\s*(\d+)", content)
            if m:
                results[name] = int(m.group(1))
            else:
                results[name] = 1
        else:
            # Fallback default if file not found
            results[name] = 1

    return results


def verify_modal_container_budget(hard_cap: int = MAX_GLOBAL_MODAL_CONTAINERS, base_dir: str = None) -> Tuple[bool, int, Dict[str, int]]:
    """
    Verifies that the sum of all configured max_containers does not exceed hard_cap.
    Returns: (is_valid, total_containers, breakdown)
    """
    configured = get_configured_modal_containers(base_dir=base_dir)
    total = sum(configured.values())
    is_valid = total <= hard_cap
    if not is_valid:
        raise ValueError(
            f"Configured Modal containers ({total}) exceeds the hard budget cap of {hard_cap}! "
            f"Breakdown: {configured}. Please adjust deploy_*.py files."
        )
    return is_valid, total, configured


class ModalBudgetController:
    """
    Thread-safe concurrency coordinator for client-side task dispatching.
    Ensures parallel calls do not queue more workers than permitted.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, hard_cap: int = MAX_GLOBAL_MODAL_CONTAINERS):
        if getattr(self, "_initialized", False):
            return
        self.hard_cap = hard_cap
        self._active_tasks = 0
        self._task_lock = threading.Lock()
        self._semaphore = threading.Semaphore(hard_cap)
        self._initialized = True

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """Acquires a container slot under the global cap."""
        return self._semaphore.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        """Releases an acquired container slot."""
        self._semaphore.release()

    @staticmethod
    def estimate_production_cost(
        num_scenes: int,
        detected_chars: int = 2,
        wan_seconds_per_clip: float = 250.0,
        qwen_seconds_per_img: float = 85.0,
        z_image_seconds_per_char: float = 20.0,
        narrator_seconds_total: float = 10.0,
        end_frames_skipped: int = 0,
    ) -> Dict[str, Any]:
        """
        Estimates dollar cost for a film run based on Modal GPU execution seconds.
        """
        # Character portraits on L40S
        char_gpu_sec = detected_chars * z_image_seconds_per_char
        char_cost = char_gpu_sec * PER_SECOND_RATES["L40S"]

        # Keyframes on L40S: Start frames (num_scenes) + End frames (num_scenes - end_frames_skipped)
        total_keyframes = max(1, (num_scenes * 2) - end_frames_skipped)
        qwen_gpu_sec = total_keyframes * qwen_seconds_per_img
        qwen_cost = qwen_gpu_sec * PER_SECOND_RATES["L40S"]

        # Wan video on A100-80GB: num_scenes clips
        wan_gpu_sec = num_scenes * wan_seconds_per_clip
        wan_cost = wan_gpu_sec * PER_SECOND_RATES["A100-80GB"]

        # Narrator on CPU / lightweight
        narr_cost = narrator_seconds_total * PER_SECOND_RATES["CPU"]

        total_cost = char_cost + qwen_cost + wan_cost + narr_cost
        return {
            "estimated_cost_usd": round(total_cost, 2),
            "breakdown_usd": {
                "wan_a100": round(wan_cost, 2),
                "qwen_l40s": round(qwen_cost, 2),
                "z_image_l40s": round(char_cost, 2),
                "narrator_cpu": round(narr_cost, 3),
            },
            "total_gpu_seconds": round(char_gpu_sec + qwen_gpu_sec + wan_gpu_sec, 1),
        }
