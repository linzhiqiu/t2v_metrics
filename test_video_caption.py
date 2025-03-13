import t2v_metrics
tested_models = [
    "tarsier-recap-7b",
    # "llava-video-7b",
    # "llava-video-72B",
    # "internvl2.5-8b",
    # "internvl2.5-26b",
    # "internvl2.5-38b", # Not working
    # "internvl2.5-78b", # Not working
    # "internvideo2-chat-8b-hd", # Not working
    "qwen2.5-vl-7b",
    "qwen2.5-vl-72b",
]

video_path = "videos/baby.mp4"
question = "Describe this video."
for model in tested_models:
    score = t2v_metrics.get_score_model(model=model)
    print(f"Model: {model}")
    caption = score.model.generate([video_path], [question])[0]
    print(f"Caption: {caption}")
    print()
