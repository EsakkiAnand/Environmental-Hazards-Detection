"""
Unified Hazard Dataset Preparation Entry Point
Executes the refined end-to-end dataset cleaning, connected-components mask conversion,
altitude filtering, scene-aware splitting, and verification pipeline.
"""
import prepare_unified_dataset
import verify_dataset

if __name__ == "__main__":
    prepare_unified_dataset.main()
    verify_dataset.verify_dataset()
