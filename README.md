## **VQAScore for Evaluating Text-to-Visual Models [[Project Page]](https://linzhiqiu.github.io/papers/vqascore/)**  

*VQAScore allows researchers to automatically evaluate text-to-image/video/3D models using one-line of Python code!*

[[VQAScore Page](https://linzhiqiu.github.io/papers/vqascore/)] [[VQAScore Demo](https://huggingface.co/spaces/zhiqiulin/VQAScore)]  [[GenAI-Bench Page](https://linzhiqiu.github.io/papers/genai_bench/)] [[GenAI-Bench Demo](https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer)] [[CLIP-FlanT5 Model Zoo](https://github.com/linzhiqiu/CLIP-FlanT5/blob/master/docs/MODEL_ZOO.md)]

**VQAScore: Evaluating Text-to-Visual Generation with Image-to-Text Generation** (ECCV 2024) [[Paper](https://arxiv.org/pdf/2404.01291)] [[HF](https://huggingface.co/zhiqiulin/clip-flant5-xxl)] <br>
[Zhiqiu Lin](https://linzhiqiu.github.io/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/), Baiqi Li, Jiayao Li, [Xide Xia](https://scholar.google.com/citations?user=FHLTntIAAAAJ&hl=en), [Graham Neubig](https://www.phontron.com/), [Pengchuan Zhang*](https://scholar.google.com/citations?user=3VZ_E64AAAAJ&hl=en), [Deva Ramanan*](https://www.cs.cmu.edu/~deva/) (*Co-First and co-senior authors)

**GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation** (CVPR 2024, **Best Short Paper @ SynData Workshop**) [[Paper](https://arxiv.org/abs/2406.13743)] [[HF](https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer)] <br>
Baiqi Li*, [Zhiqiu Lin*](https://linzhiqiu.github.io/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/), Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, [Xide Xia*](https://scholar.google.com/citations?user=FHLTntIAAAAJ&hl=en), [Pengchuan Zhang*](https://scholar.google.com/citations?user=3VZ_E64AAAAJ&hl=en), [Graham Neubig*](https://www.phontron.com/), [Deva Ramanan*](https://www.cs.cmu.edu/~deva/) (*Co-First and co-senior authors)

**CameraBench: Towards Understanding Camera Motions in Any Video** (arXiv 2025) \[[Paper](https://arxiv.org/abs/2504.15376)] \[[Site](https://linzhiqiu.github.io/papers/camerabench/)] <br>
[Zhiqiu Lin\*](https://linzhiqiu.github.io/), Siyuan Cen\*, Daniel Jiang, Jay Karhade, Hewei Wang, [Chancharik Mitra](https://x.com/chancharikm), Tiffany Yu Tong Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, Rushikesh Zawar, Xue Bai, Yilun Du, Chuang Gan, [Deva Ramanan](https://www.cs.cmu.edu/~deva/) (\*Co-First Authors)

## News
- [2025/09/03] 🚀 **VQAScore** gets a **major upgrade** with support for **20+ state-of-the-art video-language models** for [video-based VQAScore](#video-text-alignment-scores) (e.g., Qwen2.5-VL, LLaVA-Video, etc.), along with full integration of the new benchmark [CameraBench](https://linzhiqiu.github.io/papers/camerabench/) for evaluating camera-motion understanding in text-to-video models like Kling and Runway. Huge thanks to our collaborator **Chancharik Mitra** for leading this milestone update!
- [2025/09/03] ✨ **VQAScore** has become the **go-to evaluation choice for generative models**: **GenAI-Bench** is now adopted by **Google DeepMind** (Imagen3 & Imagen4), **Bytedance Seed**, **NVIDIA**, and others. Meanwhile, our **open-source CLIP-FlanT5 models** have been downloaded over **2 million times** on Hugging Face!
- [2024/08/13] 🔥 **VQAScore** is highlighted in Google's [Imagen3 report](https://arxiv.org/abs/2408.07009) as the strongest replacement of CLIPScore for automated evaluation! **GenAI-Bench** was chosen as one of the key benchmarks to showcase Imagen3's superior prompt-image alignment. Kudos to Google for this achievement! [[Paper](https://arxiv.org/abs/2408.07009)]
- [2024/07/01] 🔥 **VQAScore** has been accepted to ECCV 2024!
- [2024/06/20] 🔥 **GenAI-Bench** won Best Short Paper at the CVPR'24 SynData Workshop! [[Workshop Site](https://syndata4cv.github.io/)].

<img src="images/example.png" width=600> 

VQAScore significantly outperforms previous metrics such as CLIPScore and PickScore on compositional text prompts, and it is much simpler than prior art (e.g., HPSv2, TIFA, Davidsonian, VPEval, VIEScore) making use of human feedback or proprietary models like ChatGPT and GPT-4Vision. 

## Available Models:

### VQAScore
| Model Family Name | Image | Video | Models |
| --------------------- | :---: | :---: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CLIP-FlanT5 | :heavy_check_mark: | | clip-flant5-xxl, clip-flant5-xl |
| LLaVA-1.5 | :heavy_check_mark: | | llava-v1.5-13b, llava-v1.5-7b |
| ShareGPT4V | :heavy_check_mark: | | sharegpt4v-7b, sharegpt4v-13b |
| LLaVA-1.6 | :heavy_check_mark: | | llava-v1.6-13b |
| InstructBLIP-FlanT5 | :heavy_check_mark: | | instructblip-flant5-xxl, instructblip-flant5-xl |
| GPT-4 | :heavy_check_mark: | :heavy_check_mark: | gpt-4-turbo, gpt-4o, gpt-4.1 |
| LLaVA-OneVision | :heavy_check_mark: | :heavy_check_mark: | llava-onevision-qwen2-7b-si, llava-onevision-qwen2-7b-ov |
| mPLUG-Owl3 | :heavy_check_mark: | :heavy_check_mark: | mplug-owl3-7b |
| PaliGemma | :heavy_check_mark: | | paligemma-3b-mix-224, paligemma-3b-mix-448, paligemma-3b-mix-896 |
| InternVL2 | :heavy_check_mark: | :heavy_check_mark: | internvl2-1b, internvl2-2b, internvl2-4b, internvl2-8b, internvl2-26b, internvl2-40b, internvl2-llama3-76b |
| InternVL2.5 | :heavy_check_mark: | :heavy_check_mark: | internvl2.5-1b, internvl2.5-2b, internvl2.5-4b, internvl2.5-8b, internvl2.5-26b, internvl2.5-38b, internvl2.5-78b |
| InternVL3 | :heavy_check_mark: | :heavy_check_mark: | internvl3-8b, internvl3-14b, internvl3-78b |
| InternVideo2-Chat | :heavy_check_mark: | :heavy_check_mark: | internvideo2-chat-8b, internvideo2-chat-8b-hd, internvideo2-chat-8b-internlm |
| InternLM-XComposer2.5 | :heavy_check_mark: | :heavy_check_mark: | internlmxcomposer25-7b |
| Llama-3.2 | :heavy_check_mark: | | llama-3.2-1b, llama-3.2-3b, llama-3.2-1b-instruct, llama-3.2-3b-instruct, llama-3.2-11b-vision, llama-3.2-11b-vision-instruct, llama-3.2-90b-vision, llama-3.2-90b-vision-instruct |
| Llama-Guard-3 | :heavy_check_mark: | | llama-guard-3-1b, llama-guard-3-11b-vision |
| Molmo | :heavy_check_mark: | | molmo-72b-0924, molmo-7b-d-0924, molmo-7b-o-0924, molmoe-1b-0924 |
| Gemini | :heavy_check_mark: | :heavy_check_mark: | gemini-1.5-pro, gemini-1.5-flash, gemini-2.5-pro-preview-03-25 |
| Qwen2-VL | :heavy_check_mark: | :heavy_check_mark: | qwen2-vl-2b, qwen2-vl-7b, qwen2-vl-72b |
| Qwen2.5-VL | :heavy_check_mark: | :heavy_check_mark: | qwen2.5-vl-3b, qwen2.5-vl-7b, qwen2.5-vl-32b, qwen2.5-vl-72b |
| LLaVA-Video | | :heavy_check_mark: | llava-video-7b, llava-video-72B |
| Tarsier | | :heavy_check_mark: | tarsier-recap-7b, tarsier2-7b |
| Perception-LM | | :heavy_check_mark: | perception-lm-1b, perception-lm-3b, perception-lm-8b |
---
### ITMScore
| Model Family Name | Image | Video | Models |
| ----------------- | :---: | :---: | ----------------------------------------- |
| BLIP2-ITM | :heavy_check_mark: | | blip2-itm, blip2-itm-vitL, blip2-itm-coco |
| UMT-ITM | :heavy_check_mark: | :heavy_check_mark: | umt-b16-25m-itm, umt-l16-25m-itm |
| InternVideo2-ITM | :heavy_check_mark: | :heavy_check_mark: | internvideo2-1b-stage2-itm |
---
### CLIPScore
| Model Family Name | Image | Video | Models |
| ------------------- | :---: | :---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All OpenCLIP Models | :heavy_check_mark: | | openai\:ViT-B-32, openai\:ViT-L-14, laion2b\_s32b\_b82k\:ViT-L-14, datacomp\_xl\_s13b\_b90k\:ViT-B-16, webli\:ViT-B-16-SigLIP, metaclip\_400m\:ViT-B-32, ... (200+ models available - use `t2v_metrics.list_all_clipscore_models()` to see all exact model names) |
| BLIP2-ITC | :heavy_check_mark: | | blip2-itc, blip2-itc-vitL, blip2-itc-coco |
| HPSv2 | :heavy_check_mark: | | hpsv2 |
| PickScore | :heavy_check_mark: | | pickscore-v1 |
| UMT-CLIP | :heavy_check_mark: | :heavy_check_mark: | umt-b16-25m-clip, umt-l16-25m-clip |
| InternVideo2-CLIP | :heavy_check_mark: | :heavy_check_mark: | internvideo2-1b-stage2-clip |
| LanguageBind Video | :heavy_check_mark: | :heavy_check_mark: | languagebind-video-v1.5-ft, languagebind-video-ft, languagebind-video-v1.5, languagebind-video |

## Quick start

Install the package in editable mode via:
```bash
git clone https://github.com/linzhiqiu/t2v_metrics
cd t2v_metrics

conda create -n t2v python=3.10 -y
conda activate t2v
conda install pip -y

conda install ffmpeg -c conda-forge
pip install -e . # local pip install
```

Or you can do a standard install via `pip install t2v-metrics`.

**Note**: Certain models have additional requirements for full usability that may conflict with other model requirements. For these rare cases, please install the dependencies in the corresponding requirements.txt folder.

Now, the following Python code is all you need to compute the VQAScore for image-text alignment (higher scores indicate greater similarity):

```python
import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

### For a single (image, text) pair
image = "images/0.png" # an image path in string format
text = "someone talks on the phone angrily while another person sits happily"
score = clip_flant5_score(images=[image], texts=[text])

### Alternatively, if you want to calculate the pairwise similarity scores 
### between M images and N texts, run the following to return a M x N score tensor.
images = ["images/0.png", "images/1.png"]
texts = ["someone talks on the phone angrily while another person sits happily",
         "someone talks on the phone happily while another person sits angrily"]
scores = clip_flant5_score(images=images, texts=texts) # scores[i][j] is the score between image i and text j
```

### Notes on GPU and cache
- **GPU usage**: By default, this code uses the first cuda device on your machine. We recommend 40GB GPUs for the largest VQAScore models such as `clip-flant5-xxl` and `llava-v1.5-13b`. If you have limited GPU memory, consider smaller models such as `clip-flant5-xl` and `llava-v1.5-7b`.
- **Cache directory**: You can change the cache folder which saves all model checkpoints (default is `./hf_cache/`) by updating `HF_CACHE_DIR` in [t2v_metrics/constants.py](t2v_metrics/constants.py).


## Benchmarking text-to-image models on GenAI-Bench

### 1. Generate Images
To generate images using a specified model, run:
```bash
python -m genai_bench.generate --output_dir ./outputs/ --gen_model runwayml/stable-diffusion-v1-5
```

The generated images will be saved in `./outputs/<model>/`. You may want to modify this script to generate images using your own models.

### 2. Evaluate VQAScore Performance

You can evaluate your model using VQAScore based on clip-flant5-xxl:
```bash
python -m genai_bench.evaluate --model clip-flant5-xxl --output_dir ./outputs --gen_model runwayml/stable-diffusion-v1-5
```

Or you can use GPT-4o based VQAScore:
```bash
python -m genai_bench.evaluate --model gpt-4o --api_key INPUT_YOUR_KEY_HERE --output_dir ./outputs --gen_model runwayml/stable-diffusion-v1-5
```

For comparative VQAScore results (based on clip-flant5-xxl and GPT-4o) against state-of-the-art models like DALLE-3 and Midjourney v6, please refer to the [VQAScore results](https://github.com/linzhiqiu/t2v_metrics/blob/main/genai_bench/model_performance_vqacore.md)!


## **Advanced Usage**  

- [Batch processing for more image-text pairs](#batch-processing-for-more-image-text-pairs)
- [Check all supported models](#check-all-supported-models)
- [Customizing the question and answer template (for VQAScore)](#customizing-the-question-and-answer-template-for-vqascore)
- [Reproducing VQAScore paper results](#reproducing-vqascore-paper-results)
- [Reproducing GenAI-Bench paper results](#reproducing-genai-bench-paper-results)
- [Using GPT-4o for VQAScore](#using-gpt-4o-for-vqascore)
- [Implementing your own scoring metric](#implementing-your-own-scoring-metric)
- [Text generation (VQA) using CLIP-FlanT5](#text-generation-vqa-using-clip-flant5)
- [Video-text alignment scores](#video-text-alignment-scores)

### Batch processing for more image-text pairs
With a large batch of M images x N texts, you can speed up using the ``batch_forward()`` function. 
```python
import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')

# The number of images and texts per dictionary must be consistent.
# E.g., the below example shows how to evaluate 4 generated images per text
dataset = [
  {'images': ["images/0/DALLE3.png", "images/0/Midjourney.jpg", "images/0/SDXL.jpg", "images/0/DeepFloyd.jpg"], 'texts': ["The brown dog chases the black dog around the tree."]},
  {'images': ["images/1/DALLE3.png", "images/1/Midjourney.jpg", "images/1/SDXL.jpg", "images/1/DeepFloyd.jpg"], 'texts': ["Two cats sit at the window, the blue one intently watching the rain, the red one curled up asleep."]},
  #...
]
scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=16) # (n_sample, 4, 1) tensor
```

### Check all supported models
We currently support running VQAScore with CLIP-FlanT5, LLaVA-1.5, and InstructBLIP as well as SOTA video-language models like Qwen2.5-VL, InternVL3, GPT-4o, and Gemini-2.5-pro:
```python
gpt4o_score = t2v_metrics.VQAScore(model='gpt-4o', api_key="YOUR_API_KEY") # Using OpenAI Key
gemini25_score = t2v_metrics.VQAScore(model='gemini-2.5-pro', api_key="YOUR_API_KEY") # This is using your Gemini API key, which is the recommended method. If you would like to use your Vertex AI project, please make a request on Github.
qwen25vl_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b')
internvl3_score = t2v_metrics.VQAScore(model='internvl3-8b')
```
You can check all supported models by running the below commands:

```python
print("VQAScore models:")
t2v_metrics.list_all_vqascore_models()

print("ITMScore models:")
t2v_metrics.list_all_itmscore_models()

print("CLIPScore models:")
t2v_metrics.list_all_clipscore_models()
```

### Customizing the question and answer template (for VQAScore)
The question and answer slightly affect the final score, as shown in the Appendix of our paper. We provide a simple default template for each model and do not recommend changing it for the sake of reproducibility. However, we do want to point out that the question and answer can be easily modified. For example, CLIP-FlanT5 and LLaVA-1.5 use the following template, which can be found at [t2v_metrics/models/vqascore_models/clip_t5_model.py](t2v_metrics/models/vqascore_models/clip_t5_model.py):

```python
# {} will be replaced by the caption
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'
```

You can customize the template by passing the `question_template` and `answer_template` parameters into the `forward()` or `batch_forward()` functions:

```python
# Use a different question for VQAScore
scores = clip_flant5_score(images=images,
                           texts=texts,
                           question_template='Is this figure showing "{}"? Please answer yes or no.',
                           answer_template='Yes')
```

You may also compute P(caption | image) ([VisualGPTScore](https://linzhiqiu.github.io/papers/visual_gpt_score)) instead of P(answer | image, question):
```python
scores = clip_flant5_score(images=images,
                           texts=texts,
                           question_template="", # no question
                           answer_template="{}") # this computes P(caption | image)
```

### Reproducing VQAScore paper results

Our [eval.py](eval.py) allows you to easily run 10 image/vision/3D alignment benchmarks (e.g., Winoground/TIFA160/SeeTrue/StanfordT23D/T2VScore):
```bash
python eval.py --model clip-flant5-xxl # for VQAScore
python eval.py --model openai:ViT-L-14 # for CLIPScore

# You can optionally specify question/answer template, for example:
python eval.py --model clip-flant5-xxl --question "Is the figure showing '{}'?" --answer "Yes"
```

### Reproducing GenAI-Bench paper results

Our [genai_image_eval.py](genai_image_eval.py) and [genai_video_eval.py](genai_video_eval.py) can reproduce the GenAI-Bench results. In additional [genai_image_ranking.py](genai_image_ranking.py) can reproduce the GenAI-Rank results:
```bash
# GenAI-Bench
python genai_image_eval.py --model clip-flant5-xxl
python genai_video_eval.py --model clip-flant5-xxl

# GenAI-Rank
python genai_image_ranking.py --model clip-flant5-xxl --gen_model DALLE_3
python genai_image_ranking.py --model clip-flant5-xxl --gen_model SDXL_Base
```

### Using GPT-4o for VQAScore!
We implemented VQAScore using GPT-4o to achieve a new state-of-the-art performance. Please see [gpt4_eval.py](gpt4_eval.py) for an example. Here is how to use it in command line:
```python
api_key = # Your OpenAI key
score_func = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", api_key=openai_key, top_logprobs=20) # We find top_logprobs=20 to be sufficient for most (image, text) samples. Consider increase this number if you get errors (the API cost will not increase).
```

### Video-Text Alignment Scores

We now support video-text alignment scores, including video-CLIPScore (InternVideo2, Unmasked Teacher, and more) and video-VQAScore (LLaVA-OneVision, Qwen2.5-VL, and more). 

For single-image and CLIP-like models, video frames are concatenated. For all other native video models (we recommend Qwen2.5-VL at the time of writing), video frames are passed directly to the model.

```python
import t2v_metrics

### For a single (video, text) pair:
qwen_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b') 
video = "videos/baby.mp4"
text = "a baby crying"
score = qwen_score(images=[video], texts=[text]) 

### Pairwise similarity scores between M videos and N texts:
videos = ["videos/baby.mp4", "videos/ducks.mp4"]
texts = ["a baby crying", "a group of ducks standing in the water"]
score = qwen_score(images=videos, texts=texts, fps=8.0)  # M x N score tensor

# For Qwen models, specify fps:
# score = qwen_score(images=[video], texts=[text], fps=8.0) # We default to 8.0 FPS for balancing computationally reasonable inference and performance. To switch to Qwen's dynamic FPS sampling, you must explicitly set it to "dynamic"
# score = qwen_score(images=[video], texts=[text], fps="dynamic")

# For other models like LLaVA, use num_frames
# llava_score = t2v_metrics.VQAScore(model='llava-onevision-qwen2-7b-ov')
# score = llava_score(images=[video], texts=[text], num_frames=8) # We did our best to align with the default num_frames for each model, but to be certain, please check each respective model's spec or paper for confirmation.
```

### Text generation (VQA)
To generate texts (captioning or VQA tasks) for any of our models, please use the below code:
```python
import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')

images = ["images/0.png", "images/0.png"] # A list of images
texts = ["Please describe this image: ", "Does the image show 'someone talks on the phone angrily while another person sits happily'?"] # Corresponding prompts
clip_flant5_score.model.generate(images=images, texts=prompts)
```
The generate method for CLIP-FlanT5 may require downgrading to 4.36.1:
```
pip install transformers==4.36.1
```
### Implementing your own scoring metric
You can easily implement your own scoring metric. For example, if you have a VQA model that you believe is more effective, you can incorporate it into the directory at [t2v_metrics/models/vqascore_models](t2v_metrics/models/vqascore_models/). For guidance, please refer to our example implementations of [LLaVA-1.5](t2v_metrics/models/vqascore_models/llava_model.py) and [InstructBLIP](t2v_metrics/models/vqascore_models/instructblip_model.py) as starting points.


## Contributions

- **[Zhiqiu Lin](https://x.com/ZhiqiuLin)**, **[Jean de Nyandwi](https://x.com/Jeande_d)**, **[Chancharik Mitra](https://x.com/chancharikm)**  
  Implemented image-based **CLIPScore** and **VQAScore** for:  
  CLIP-FlanT5, GPT-4o, LLaVA-1.5, InstructBLIP, OpenCLIP, HPSv2, PickScore.

- **Baiqi Li**  
  Implemented **GenAI-Bench** and **GenAI-Rank** benchmarks.

- **[Chancharik Mitra](https://x.com/chancharikm)**  
  Implemented CameraBench and video-based **VQAScore** for:  
  LLaVA-OneVision, Qwen2.5-VL, InternVideo2, InternVL2, InternVL3, InternLMXC2.5, etc.

## Citation

If you find this repository useful for your research, please cite the following papers:

```
@article{lin2024evaluating,
  title   = {Evaluating Text-to-Visual Generation with Image-to-Text Generation},
  author  = {Lin, Zhiqiu and Pathak, Deepak and Li, Baiqi and Li, Jiayao and Xia, Xide and Neubig, Graham and Zhang, Pengchuan and Ramanan, Deva},
  journal = {arXiv preprint arXiv:2404.01291},
  year    = {2024}
}

@article{li2024genaibench,
  title     = {GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation},
  author    = {Li, Baiqi and Lin, Zhiqiu and Pathak, Deepak and Li, Jiayao and Fei, Yixin and Wu, Kewen and Ling, Tiffany and Xia, Xide and Zhang, Pengchuan and Neubig, Graham and Ramanan, Deva},
  journal   = {arXiv preprint arXiv:2406.13743},
  year      = {2024}
}

@article{camerabench,
  title     = {Towards Understanding Camera Motions in Any Video},
  author    = {Lin, Zhiqiu and Cen, Siyuan and Jiang, Daniel and Karhade, Jay and Wang, Hewei and Mitra, Chancharik and Ling, Yu Tong Tiffany and Huang, Yuhan and Liu, Sifan and Chen, Mingyu and Zawar, Rushikesh and Bai, Xue and Du, Yilun and Gan, Chuang and Ramanan, Deva},
  journal   = {arXiv preprint arXiv:2504.15376},
  year      = {2025}
}
```

## Acknowledgements
This repository is inspired from the [Perceptual Metric (LPIPS)](https://github.com/richzhang/PerceptualSimilarity) repository by Richard Zhang for automatic evaluation of image quality.
