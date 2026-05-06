import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Resource discovery helper (used across train.py / inference.py too)
# ─────────────────────────────────────────────────────────────────────────────
def detect_hw(reserved_cpus: int = 2) -> dict:
    """Detect CPU and GPU resources available to this process."""
    info = {
        "cpu_count":      os.cpu_count() or 4,
        "workers":        max(1, (os.cpu_count() or 4) - reserved_cpus),
        "gpu_count":      0,
        "gpu_name":       None,
        "vram_gb":        0.0,
        "bf16_supported": False,
    }
    try:
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"]       = props.name
            info["vram_gb"]        = props.total_memory / 1e9
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        pass
    return info