#!/usr/bin/env python3
"""
Data Download Script for CameraBench Videos
Downloads videos from HuggingFace repository for camera motion understanding.
"""

import os
from pathlib import Path

# Install huggingface_hub if not available
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import snapshot_download

def main():
    repo_id = "chancharikm/cambench_train_videos"
    output_dir = "data/videos"
    
    print("🎥 Downloading CameraBench videos...")
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download videos
        print("Downloading... This may take a while.")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir="data",
            allow_patterns="videos/*"
        )
        
        print(f"✅ Videos downloaded successfully to ./{output_dir}/")
        print("\nNext steps:")
        print("1. Go to data/cam_motion directory")
        print(f"2. Update video paths to point to ./videos/")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()