import os
import torch
import subprocess

def detect_hw(verbose=False):
    """Detect and display per-GPU hardware & utilization specs, matching the output style:
    [rank X] GPU N (NAME): Y GB free / Z GB total, U% util / M% mem
    """
    hw_info = {}

    # CPU info
    try:
        hw_info['cpu_count'] = os.cpu_count()
    except Exception:
        hw_info['cpu_count'] = None

    # GPU info
    gpu_count = torch.cuda.device_count()
    hw_info['gpu_count'] = gpu_count

    if gpu_count > 0:
        gpu_summary = []
        for device in range(gpu_count):
            name = torch.cuda.get_device_name(device)
            stats = torch.cuda.get_device_properties(device)
            # VRAM (total & free)
            try:
                total_mem = stats.total_memory / 1024**3  # GB
                torch.cuda.set_device(device)
                free_mem = torch.cuda.mem_get_info(device)[0] / 1024**3  # GB
            except Exception:
                total_mem, free_mem = None, None
            # Utilization via nvidia-smi
            gpu_util = None
            mem_util = None
            try:
                smi_out = subprocess.check_output([
                    'nvidia-smi',
                    f'--id={device}',
                    f'--query-gpu=utilization.gpu,utilization.memory',
                    '--format=csv,noheader,nounits'
                ])
                util_str = smi_out.decode().strip().split(",")
                if len(util_str) == 2:
                    gpu_util = int(util_str[0].strip())
                    mem_util = int(util_str[1].strip())
            except Exception:
                gpu_util, mem_util = None, None
            gpu_summary.append({
                "id": device,
                "name": name,
                "vram_total_gb": total_mem,
                "vram_free_gb": free_mem,
                "gpu_util_%": gpu_util,
                "mem_util_%": mem_util,
            })
            # Print per-GPU info as in output.txt (4-6)
            if verbose:
                print(
                    f"GPU {device} ({name}): "
                    f"{free_mem:.1f} GB free / {total_mem:.1f} GB total"
                    f"{f', {gpu_util}% util' if gpu_util is not None else ''}"
                    f"{f' / {mem_util}% mem' if mem_util is not None else ''}"
                )
        # Pick "main" GPU info (first GPU)
        first = gpu_summary[0]
        hw_info['gpu_name'] = first["name"]
        hw_info['vram_gb'] = first["vram_total_gb"]
        hw_info['vram_free_gb'] = first["vram_free_gb"]
        hw_info['gpu_util_%'] = first["gpu_util_%"]
        hw_info['mem_util_%'] = first["mem_util_%"]
        hw_info['per_gpu'] = gpu_summary
    else:
        hw_info['gpu_name'] = None
        hw_info['vram_gb'] = None
        hw_info['vram_free_gb'] = None
        hw_info['gpu_util_%'] = None
        hw_info['mem_util_%'] = None
        hw_info['per_gpu'] = []

    # BF16 supported?
    hw_info['bf16_supported'] = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    # RAM info (optional, requires psutil)
    try:
        import psutil
        mem = psutil.virtual_memory()
        hw_info['ram_total_gb'] = mem.total / 1024**3
        hw_info['ram_used_gb'] = (mem.total - mem.available) / 1024**3
        hw_info['ram_used_%'] = (mem.total - mem.available) / mem.total * 100.0
    except ImportError:
        hw_info['ram_total_gb'] = None
        hw_info['ram_used_gb'] = None
        hw_info['ram_used_%'] = None

    return hw_info

def print_hw_summary(hw_info):
    # Print CPU
    print(f"CPU count: {hw_info.get('cpu_count', 'N/A')}")
    # Print RAM
    if hw_info.get('ram_total_gb') is not None:
        print(
            f"RAM: {hw_info['ram_used_gb']:.1f} GB used / {hw_info['ram_total_gb']:.1f} GB total"
            f" ({hw_info['ram_used_%']:.1f}%)"
        )
    # Print GPUs
    if hw_info.get("gpu_count", 0) > 0:
        for g in hw_info['per_gpu']:
            print(f"GPU {g['id']} ({g['name']}): "
                  f"{g['vram_free_gb']:.1f} GB free / {g['vram_total_gb']:.1f} GB total"
                  f"{f', {g['gpu_util_%']}% util' if g['gpu_util_%'] is not None else ''}"
                  f"{f' / {g['mem_util_%']}% mem' if g['mem_util_%'] is not None else ''}")
    else:
        print("No GPUs detected.")
    # BF16 support
    print(f"BF16 supported: {hw_info['bf16_supported']}")

if __name__ == "__main__":
    hw = detect_hw(verbose=False)
    print_hw_summary(hw)