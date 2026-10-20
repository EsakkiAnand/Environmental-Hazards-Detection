import os
import sys
import zipfile
import urllib.request
import shutil
import subprocess

TARGET_RAW_DIR = r"C:\Users\esakk\dataset\raw"

DATASET_DOWNLOADS = [
    {
        "id": "dfire",
        "name": "D-Fire (Fire & Smoke)",
        "type": "github_zip",
        "url": "https://github.com/sayedgamal99/DFireDataset/archive/refs/heads/master.zip",
        "target_dir": os.path.join(TARGET_RAW_DIR, "dfire")
    },
    {
        "id": "mpid",
        "name": "MPID (Exposed Powerlines & Insulators)",
        "type": "github_zip",
        "url": "https://github.com/phd-benel/MPID/archive/refs/heads/main.zip",
        "target_dir": os.path.join(TARGET_RAW_DIR, "mpid")
    },
    {
        "id": "uav_centre_itc",
        "name": "UAV-Centre-ITC (Damaged Structures)",
        "type": "github_zip",
        "url": "https://github.com/UAV-Centre-ITC/yolo-damaged-building-opening-detection/archive/refs/heads/main.zip",
        "target_dir": os.path.join(TARGET_RAW_DIR, "uav_centre_itc")
    }
]

def download_and_extract(url, target_dir, dataset_name):
    print(f"[*] Downloading {dataset_name}...")
    zip_path = os.path.join(target_dir, "dataset.zip")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        print(f"[+] Downloaded {dataset_name}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove(zip_path)
        print(f"[SUCCESS] Extracted {dataset_name} into {target_dir}\n")
    except Exception as e:
        print(f"[!] Error downloading {dataset_name}: {e}\n")

def run_auto_downloads():
    print("=" * 65)
    print(f"  Auto-Downloading Open Datasets into: {TARGET_RAW_DIR}")
    print("=" * 65)
    
    for item in DATASET_DOWNLOADS:
        os.makedirs(item["target_dir"], exist_ok=True)
        download_and_extract(item["url"], item["target_dir"], item["name"])

if __name__ == "__main__":
    run_auto_downloads()
