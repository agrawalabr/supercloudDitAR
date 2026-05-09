import torch
import time

def dummy_gpus_hold():
    """
    Keeps all available GPUs (e.g., 8 on LS140 nodes) busy until interrupted by the user.
    Each GPU will have its own busy loop.
    Prints activity at regular intervals to show progress.
    """
    import threading

    def keep_one_gpu_busy(device, gpu_idx):
        print(f"Keeping GPU busy on device: {device} until interrupted (Ctrl+C)")
        size = 32768  # Large enough to keep the GPU occupied
        x = torch.rand(size, size, device=device)
        y = torch.rand(size, size, device=device)

        iter_count = 0
        while True:
            z = torch.mm(x, y)
            x = torch.relu(z)
            iter_count += 1
            if iter_count % 100 == 0:
                print(f"[Device {gpu_idx}] Still busy... iterations: {iter_count//100}s")
            time.sleep(0.1)

    if not torch.cuda.is_available():
        print("CUDA is not available, running on CPU (will not occupy multiple GPUs).")
        keep_one_gpu_busy(torch.device("cpu"), "cpu")
        return

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs. Keeping all occupied until interrupted (Ctrl+C).")

    threads = []
    for i in range(n_gpus):
        device = torch.device(f"cuda:{i}")
        t = threading.Thread(target=keep_one_gpu_busy, args=(device, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == "__main__":
    keep_all_gpus_busy()