import t2v_metrics
import sys

print(sys.path)
# print(f"📦 t2v_metrics location: {t2v_metrics.__file__}")
# print(f"📦 t2v_metrics version: {getattr(t2v_metrics, '__version__', 'no version')}")
# from llm.base import LLM
# from typing import List
# import os
# import inspect

# class OpenSource(LLM):
#     def __init__(self, model='perception-lm-1b'):
#         self.model = t2v_metrics.get_score_model(model=model).model
#         # print(f"📦 Model class location: {self.model.__class__.__module__}")
#         # print(f"📦 Model file: {self.model.__class__.__module__.__file__ if hasattr(model.__class__.__module__, '__file__') else 'built-in'}")
#         # print(f"📦 Model source file: {inspect.getfile(self.model.__class__)}")
    
#     def generate(self,
#                  prompt: str,
#                 #  images: List[str] = [],
#                  video: str = "",
#                 #  extracted_frames: List[int] = [],
#                  **kwargs) -> str:
#         """Generate text from a prompt. Optionally provide images or video."""
#         output = self.model.generate(images=[video], texts=[prompt])[0]
#         return output

# if __name__ == "__main__": 
#     print(f"Current working directory: {os.getcwd()}")
#     video_path = 'videos/baby.mp4'
#     # print(f"Looking for video at: {os.path.abspath(video_path)}")
#     # print(f"Video file exists: {os.path.exists(video_path)}")
    
#     # List what's actually in the current directory
#     # print(f"Contents of current directory: {os.listdir('.')}")
#     # if os.path.exists('videos'):
#     #     print(f"Contents of videos directory: {os.listdir('videos')}")
#     # model = OpenSource(model="tarsier-recap-7b")
#     # model = OpenSource(model="tarsier2-7b")
#     # model = OpenSource(model="qwen2.5-vl-7b")
#     # print(os.getcwd())
#     model = OpenSource(model='perception-lm-8b')

#     print(
#         model.generate(
#             prompt="Describe the subject motion in this video.",
#             # video="https://huggingface.co/datasets/zhiqiulin/video_captioning/resolve/main/f4ZzHtww6Tc.2.2.mp4",
#             # video="https://huggingface.co/datasets/zhiqiulin/video_captioning/resolve/main/d_T0KPYgqMA.0.2.mp4",
#             video='videos/baby.mp4'
#         )
#     )
