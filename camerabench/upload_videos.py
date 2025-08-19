#!/usr/bin/env python3
"""
Script to upload videos from local directory to Hugging Face repository
"""

import os
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

def upload_videos_to_hf():
    # Configuration
    local_video_dir = "/data3/zhiqiul/video_annotation/videos"
    repo_id = "chancharikm/cambench_train_videos"
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} created or already exists")
    except Exception as e:
        print(f"Error creating repository: {e}")
        # Continue anyway in case repo already exists
        print("Continuing with upload attempt...")
    
    # Get all video files from the local directory
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    local_path = Path(local_video_dir)
    
    if not local_path.exists():
        print(f"Error: Local directory {local_video_dir} does not exist")
        return
    
    video_files = []
    for file_path in local_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print("No video files found in the directory")
        return
    
    print(f"Found {len(video_files)} video files to upload")
    
    # Upload each video file with rate limiting strategy
    for i in tqdm(range(len(video_files)), desc="Uploading videos"):
        # Pause every 600 videos to reset rate limit
        if i % 100 == 0 and i > 0:
            print(f"\nUploaded {i} videos. Pausing for 1 hour to reset rate limit...")
            time.sleep(3600)  # 1 hour pause
        
        video_file = video_files[i]
        try:
            # Get relative path from the base directory
            relative_path = video_file.relative_to(local_path)
            
            # Create the path in the repo (under videos/ folder)
            repo_path = f"videos/{relative_path}"
            
            print(f"Uploading {video_file.name} -> {repo_path}")
            
            # First try as dataset repo
            try:
                api.upload_file(
                    path_or_fileobj=str(video_file),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
            except Exception as dataset_error:
                print(f"Dataset upload failed, trying as model repo: {dataset_error}")
                # If dataset fails, try as model repo
                api.upload_file(
                    path_or_fileobj=str(video_file),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
            
            print(f"✓ Successfully uploaded {video_file.name}")
            
            # Keep short delay for rate limiting (same as reference script)
            time.sleep(1.0)
            
        except Exception as e:
            print(f"✗ Error uploading {video_file.name}: {e}")
            # Still wait even on error to be safe
            time.sleep(1.0)
    
    print(f"Upload process completed. Repository: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Check if HF_TOKEN is set
    if not os.getenv('HF_TOKEN'):
        print("Error: HF_TOKEN environment variable is not set")
        print("Please set it with: export HF_TOKEN=your_token_here")
        exit(1)
    
    upload_videos_to_hf()