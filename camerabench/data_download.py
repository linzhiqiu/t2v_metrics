#!/usr/bin/env python3
"""
Data Download Script for CameraBench Videos
Downloads videos from HuggingFace repository for camera motion understanding.
"""

import os
import time
from pathlib import Path

# Install huggingface_hub if not available
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system("pip install huggingface_hub")
    from huggingface_hub import snapshot_download

def main():
    repo_id = "syCen/Videos4CameraBench"
    output_dir = "data/videos"
    
    print("Downloading CameraBench videos...")
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    max_retries = 3
    base_delay = 2  # Base delay in seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)

            # Download videos with rate limiting considerations
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir="data/videos",
                # allow_patterns="videos/*",
                tqdm_class=None,  # Disable individual file progress bars
                max_workers=1,    # Use single thread to avoid overwhelming the server
                token=None        # Use default token handling
            )
            
            print(f"Videos downloaded successfully to ./{output_dir}/")
            print("\nNext steps:")
            print("1. Go to data/cam_motion directory")
            print(f"2. Update video paths to point to ./videos/")
            return  # Success, exit the function
            
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print("This might be due to rate limiting. Retrying with delay...")
            else:
                print("All download attempts failed.")
                print("\nTroubleshooting:")
                print("1. The server may be rate limiting downloads")
                print("2. Try running the script again later")
                print("3. Check your internet connection")
                print("4. Consider downloading in smaller batches if the issue persists")

if __name__ == "__main__":
    main()
