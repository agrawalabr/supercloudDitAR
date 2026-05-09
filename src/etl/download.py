import subprocess
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm.notebook import tqdm

BUCKET = "mit-supercloud-dataset"
PREFIX = "datacenter-challenge/202201"
OUT_DIR = "./mit"
GPU_BATCH_SIZE = 1000
MAX_WORKERS = os.cpu_count()

os.makedirs(f"{OUT_DIR}/gpu", exist_ok=True)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_key(key):
    rel = key[len(PREFIX)+1:]
    local = os.path.join(OUT_DIR, rel)
    
    # Skip if already downloaded
    if os.path.exists(local):
        return key, "skipped"
    
    os.makedirs(os.path.dirname(local), exist_ok=True)
    s3.download_file(BUCKET, key, local)
    return key, "downloaded"

print("Listing gpu/ keys...")
paginator = s3.get_paginator("list_objects_v2")
gpu_keys = []
for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/gpu/"):
    for obj in page.get("Contents", []):
        gpu_keys.append(obj["Key"])

print(f"Found {len(gpu_keys)} gpu files. Downloading in batches of {GPU_BATCH_SIZE} with {MAX_WORKERS} workers...")

batches = [gpu_keys[i:i+GPU_BATCH_SIZE] for i in range(0, len(gpu_keys), GPU_BATCH_SIZE)]
total_downloaded = 0
total_skipped = 0

for batch_num, batch in enumerate(batches):
    print(f"Batch {batch_num+1}/{len(batches)} ({len(batch)} files)")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_key, k): k for k in batch}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num+1}"):
            try:
                _, status = f.result()
                if status == "skipped":
                    total_skipped += 1
                else:
                    total_downloaded += 1
            except Exception as e:
                print(f"Failed: {futures[f]} — {e}")

print(f"All done. Downloaded: {total_downloaded} | Skipped (already existed): {total_skipped}")