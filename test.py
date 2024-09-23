import t2v_metrics

# llava_ov_score = t2v_metrics.VQAScore(model='llava-onevision-qwen2-7b-ov')
# image = "images/0.png" # an image path in string format
# text = "someone talks on the phone angrily while another person sits happily"
# score = llava_ov_score(images=[image], texts=[text])
# print(score)

# mplug_score = t2v_metrics.VQAScore(model='internvl2-8b')
# image = "images/0.png" # an image path in string format
# text = "someone talks on the phone angrily while another person sits happily"
# score = mplug_score(images=[image], texts=[text])
# print(score)

# import t2v_metrics
# import torch
# from transformers import AutoModelForCausalLM

# from transformers import AutoTokenizer, AutoProcessor
# from decord import VideoReader, cpu    # pip install decord


# model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
# model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='flash_attention_2', trust_remote_code=True, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# processor = model.init_processor(tokenizer)

# messages = [
#     {"role": "user", "content": """<|video|>
# Describe this video."""},
#     {"role": "assistant", "content": ""}
# ]

# videos = ['/nas-mmu-data/examples/car_room.mp4']

# MAX_NUM_FRAMES=16

# def encode_video(video_path):
#     def uniform_sample(l, n):
#         gap = len(l) / n
#         idxs = [int(i * gap + gap / 2) for i in range(n)]
#         return [l[i] for i in idxs]

#     vr = VideoReader(video_path, ctx=cpu(0))
#     sample_fps = round(vr.get_avg_fps() / 1)  # FPS
#     frame_idx = [i for i in range(0, len(vr), sample_fps)]
#     if len(frame_idx) > MAX_NUM_FRAMES:
#         frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
#     frames = vr.get_batch(frame_idx).asnumpy()
#     frames = [Image.fromarray(v.astype('uint8')) for v in frames]
#     print('num frames:', len(frames))
#     return frames
# video_frames = [encode_video(_) for _ in videos]
# inputs = processor(messages, images=None, videos=video_frames)

# inputs.to('cuda')
# inputs.update({
#     'tokenizer': tokenizer,
#     'max_new_tokens':100,
#     'decode_text':True,
# })


# g = model.generate(**inputs)
# print(g)
# # add icecream and flash-attn
# # model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
# # config = mPLUGOwl3Config.from_pretrained(model_path)
# # print(config)
# # # model = mPLUGOwl3Model(config).cuda().half()
# # model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# # model.eval().cuda()

internvideo_score = t2v_metrics.VQAScore('internvideo2-chat-8b-hd') 
# image = "images/0.png" # an image path in string format
# text = "someone talks on the phone angrily while another person sits happily"
video = "videos/1_yes.mp4" # an image path in string format
text = "a garage door opening"
score = internvideo_score(videos=[video], texts=[text], num_frames=8)
print(score)