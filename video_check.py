import json
import os

import pandas as pd

# missing_videos = []
# with open('cambench_vqa_complex.json', 'r') as compfile, open('cambench_vqa.json', 'r') as regfile:
#     comp_json = json.load(compfile)
#     reg_json = json.load(regfile)
#     for comp_item in comp_json:
#         comp_video = comp_item['Video']
#         missing = True
#         for reg_item in reg_json:
#             reg_video = reg_item['Video']
#             if reg_video == comp_video:
#                 missing = False
#                 print(f'comp_video  {comp_video} reg_video {reg_video}')
        
#         if missing:
#             missing_videos.append(comp_video)

# print(len(missing_videos))

with open('cambench_vqa_complex.json', 'r') as compfile:
    # Fix file paths:
    new_video_paths = []
    complex_data = json.load(compfile)
    for item in complex_data:
        filename = os.path.basename(item["Video"])
        new_video_paths.append(os.path.join('videos', filename))
    dataframe_regular = pd.read_csv('train.csv')
    print(len(dataframe_regular))
    dataframe_complex = pd.DataFrame(complex_data)
    print(len(dataframe_complex))
    complex_indices = list(range(9888, 14172))
    dataframe_complex['Index'] = complex_indices
    dataframe_complex['Video'] = new_video_paths

    combined = pd.concat([dataframe_regular, dataframe_complex], axis=0)
    combined.to_csv('test.csv', index=False)
    print(len(combined))