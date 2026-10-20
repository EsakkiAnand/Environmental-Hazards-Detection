import os
import sys
import json
import urllib.request
import zipfile
import shutil

TARGET_DATASET_DIR = r"C:\Users\esakk\dataset"

DATASET_CONFIG = {
    "raw_sources": [
        "dfire",
        "uavs_ffdb",
        "drone_for_flood",
        "uav_centre_itc",
        "rescuenet",
        "mpid",
        "cas_landslide",
        "chem_spill"
    ],
    "unified_dir": os.path.join(TARGET_DATASET_DIR, "unified_hazards"),
    "metadata": {
        "dfire": {
            "name": "D-Fire Dataset",
            "url": "https://github.com/sayedgamal99/DFireDataset",
            "description": "Ground-level fire and smoke dataset (~21k images). Sourced strictly for TRAIN split.",
            "notes": "Clone repo or download release zip into C:\\Users\\esakk\\dataset\\raw\\dfire\\"
        },
        "uavs_ffdb": {
            "name": "UAVs-FFDB (Forest Fire Aerial)",
            "url": "https://openaccess.hku.edu.tr/items/30214f7b-6b8e-4a4d-b221-bb9aedf17547",
            "description": "Aerial drone-captured fire imagery (5-15m altitude). Sourced for VAL split.",
            "notes": "Extract aerial fire images into C:\\Users\\esakk\\dataset\\raw\\uavs_ffdb\\"
        },
        "drone_for_flood": {
            "name": "Drone-for-flood (Roboflow)",
            "url": "https://universe.roboflow.com/drone-7nzsq/drone-for-flood",
            "description": "Drone-captured flood imagery in YOLO format.",
            "notes": "Export in YOLOv8 format from Roboflow Universe into C:\\Users\\esakk\\dataset\\raw\\drone_for_flood\\"
        },
        "uav_centre_itc": {
            "name": "UAV-Centre-ITC Damaged Building Opening",
            "url": "https://github.com/UAV-Centre-ITC/yolo-damaged-building-opening-detection",
            "description": "UAV damaged building opening dataset for search & rescue.",
            "notes": "Clone repo or download data into C:\\Users\\esakk\\dataset\\raw\\uav_centre_itc\\"
        },
        "rescuenet": {
            "name": "RescueNet Debris Dataset",
            "url": "https://www.nature.com/articles/s41597-023-02799-4",
            "description": "UAV post-disaster debris dataset with segmentation masks.",
            "notes": "Download images & masks into C:\\Users\\esakk\\dataset\\raw\\rescuenet\\"
        },
        "mpid": {
            "name": "Merged Public Insulator Dataset (MPID)",
            "url": "https://github.com/phd-benel/MPID",
            "description": "Exposed powerlines and insulator inspection dataset.",
            "notes": "Clone repo into C:\\Users\\esakk\\dataset\\raw\\mpid\\"
        },
        "cas_landslide": {
            "name": "CAS Landslide Dataset",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10762236/",
            "description": "Satellite & UAV landslide dataset (20k+ images).",
            "notes": "Extract into C:\\Users\\esakk\\dataset\\raw\\cas_landslide\\"
        },
        "chem_spill": {
            "name": "Chemical Spill Dataset (Roboflow)",
            "url": "https://universe.roboflow.com/k-d113o/chem-spill",
            "description": "Chemical/oil spill ground dataset (56 images).",
            "notes": "Export in YOLOv8 format from Roboflow Universe into C:\\Users\\esakk\\dataset\\raw\\chem_spill\\"
        }
    }
}

def init_directory_structure():
    print("=" * 65)
    print(f" Initializing Dataset Storage at: {TARGET_DATASET_DIR}")
    print("=" * 65)
    
    # Create target root
    os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
    
    # Create raw folders
    raw_root = os.path.join(TARGET_DATASET_DIR, "raw")
    for source in DATASET_CONFIG["raw_sources"]:
        source_path = os.path.join(raw_root, source)
        os.makedirs(source_path, exist_ok=True)
        print(f"[+] Created raw folder: {source_path}")
        
    # Create unified dataset structure
    unified_root = DATASET_CONFIG["unified_dir"]
    for split in ["train", "val"]:
        os.makedirs(os.path.join(unified_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(unified_root, split, "labels"), exist_ok=True)
        print(f"[+] Created unified split: {os.path.join(unified_root, split)}")
        
    # Write dataset_summary.json
    summary_path = os.path.join(TARGET_DATASET_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_CONFIG, f, indent=4)
        
    print("\n[SUCCESS] All directory structures created cleanly!")
    print(f"[+] Summary saved to: {summary_path}\n")

def print_ingestion_guide():
    print("=" * 65)
    print(" Dataset Download & Placement Guide")
    print("=" * 65)
    for source_id, info in DATASET_CONFIG["metadata"].items():
        print(f"\n-> [{info['name']}]")
        print(f"  Target Directory: C:\\Users\\esakk\\dataset\\raw\\{source_id}\\")
        print(f"  Source URL:       {info['url']}")
        print(f"  Description:      {info['description']}")
        print(f"  Instructions:     {info['notes']}")
    print("=" * 65)

if __name__ == "__main__":
    init_directory_structure()
    print_ingestion_guide()
