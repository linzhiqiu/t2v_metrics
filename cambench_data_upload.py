# import json
# import os
# import pandas as pd
# import shutil
# from huggingface_hub import HfApi, create_repo, login
# from pathlib import Path

# # Configuration
# DATA_FILE = "cambench_vqa.json"
# REPO_ID = "chancharikm/camerabench_vqa_lmms_eval"
# VIDEO_DIR = "/data3/zhiqiul/video_annotation/videos/"

# def upload_with_large_folder_correct():
#     # Setup
#     api = HfApi()
#     create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    
#     # Load your data
#     with open(DATA_FILE, 'r') as f:
#         data = json.load(f)
    
#     # Create local repo structure
#     local_repo = "./temp_repo"
#     os.makedirs(f"{local_repo}/videos", exist_ok=True)
    
#     # Copy videos to local structure
#     print("Copying videos to local repo structure...")
#     for item in data:
#         video_path = item["Video"]
#         if os.path.exists(video_path):
#             filename = os.path.basename(video_path)
#             shutil.copy2(video_path, f"{local_repo}/videos/{filename}")
#             item["Video"] = f"videos/{filename}"
    
#     # Create train.csv in local repo
#     df = pd.DataFrame(data)
#     df.to_csv(f'{local_repo}/train.csv', index=False)
    
#     # Upload entire local repo structure
#     print("Uploading entire repo...")
#     api.upload_large_folder(
#         folder_path=local_repo,
#         repo_id=REPO_ID,
#         repo_type="dataset",
#     )
    
#     print(f"✅ Dataset ready! Use: load_dataset('{REPO_ID}')")
    
#     # Cleanup
#     shutil.rmtree(local_repo)

# upload_with_large_folder_correct()

import json
import os
import pandas as pd
from huggingface_hub import HfApi, create_repo, login
import time
from tqdm import tqdm

# Configuration
DATA_FILE = "cambench_vqa.json"
REPO_ID = "chancharikm/camerabench_vqa_lmms_eval"

def upload_with_rate_limiting():
    # Setup (no need for explicit login if token is in env)
    api = HfApi()
    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    
    # Load your data
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    print(f"Total videos to upload: {len(data)}")
    
    # Upload videos with progress bar
    for i in tqdm(range(len(data)), desc="Uploading videos"):
        # Pause every 600 videos 
        if i % 600 == 0:  # Much cleaner
            print(f"Uploaded {i} videos. Pausing for 1 hour to reset rate limit...")
            time.sleep(3600)
            
        item = data[i]
        video_path = item["Video"]
        
        if os.path.exists(video_path):
            filename = os.path.basename(video_path)
            
            try:
                api.upload_file(
                    repo_id=REPO_ID,
                    path_or_fileobj=video_path,
                    repo_type="dataset",
                    path_in_repo=f"videos/{filename}"
                )
                item["Video"] = f'videos/{filename}'
                
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
                item["Video"] = f'videos/{filename}'
            
            time.sleep(1.0)  # Keep short delay for rate limiting
        

    # Create and upload CSV with updated paths
    print("Creating and uploading metadata CSV...")
    df = pd.DataFrame(data)
    df.to_csv('train.csv', index=False)
    
    api.upload_file(
        repo_id=REPO_ID,
        path_or_fileobj="train.csv",
        repo_type="dataset",
        path_in_repo="train.csv"
    )
 
    print(f"✅ Dataset ready! Use: load_dataset('{REPO_ID}')")

upload_with_rate_limiting()