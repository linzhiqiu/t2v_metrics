# Evaluation Scripts

CameraBench uses a method-agnostic evaluation setup that separates result generation (Script 1) from metric computation (Script 2). Any method can plug in by swapping Script 1—e.g., VLMs like Qwen-2.5-VL or SfM systems like MegaSAM. We will cover how to run evaluation for (1) binary classification of camera-centric frame motion, (2) yes-or-no VQA tasks, (3) video-text retrieval, and (4) camera-motion captioning.

## Two-Stage Evaluation Process

### Script 1: Score/Prediction Generation (Method-Specific)
Generate scores or captions using your chosen method and save them in standardized formats.

### Script 2: Evaluation (Method-Agnostic) 
Compute metrics from the standardized files using evaluation scripts that work with any method.

To use another method (e.g., SfMs), simply implement your own Script 1 in the standardized format—Script 2 remains unchanged.

## Setup

Setup is identical to what is shown in the main `t2v_metrics` README, but it is reproduced here for convenience. Setup `t2v_metrics` either by:

In the project root:
```
git clone https://github.com/linzhiqiu/t2v_metrics.git
cd t2v_metrics

conda create -n t2v python=3.10 -y
conda activate t2v
conda install pip -y

conda install ffmpeg -c conda-forge
pip install -e .
```
or from anywhere via standard package install:

```
pip install t2v-metrics
```

Note that your `HF_TOKEN` may be needed when running some models, so it is best to set that beforehand.

## Data Download

Download the videos from the following HuggingFace [repo](https://huggingface.co/datasets/syCen/Videos4CameraBnech) into the directory `data/videos`. You can simply use our script:

```python
python data_download.py
```

## 1. Binary Classification Evaluation

Below, we show how to run evaluation for the 7B Qwen-2.5-VL model reported in our paper.

### Score Generation
```bash
# Generate scores using VQAScore models for all splits
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data/binary_classification --output_dir scores/

# Generate scores for specific splits only
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --splits Move_Down Move_Up Pan_Left Pan_Right --output_dir scores/
```

**Available Binary Classification Splits:**
- Movement: `Move_Up`, `Move_Down`, `Move_In`, `Move_Out`, `Move_Left`, `Move_Right`
- Rotation: `Pan_Left`, `Pan_Right`, `Roll_Clockwise`, `Roll_Counterclockwise`, `Tilt_Down`, `Tilt_Up`
- Zooming: `Zoom_In`, `Zoom_Out`
- Static: `Static`

### Evaluation
```bash

# Auto-discover files in a specific directory
python binary_classification_evaluation.py --score_dir scores/ --plots --output_dir evaluation_results

# Evaluate specific score files explicitly
python binary_classification_evaluation.py scores/vqa_scores_qwen2.5-vl-7b_*_Move_Down_*.json scores/vqa_scores_qwen2.5-vl-7b_*_Move_Up_*.json --plots --output_file movement_results.json
```

**Auto-Discovery:** When no score files are provided, automatically finds `vqa_scores_*.json` files in the specified directory.

**Required Output Format for Custom Methods:** 
```json
{
  "metadata": {
    "method_type": "Your_Method_Name",
    "model_name": "your_model_name",
    "checkpoint": "optional_checkpoint_path",
    "split_name": "Move_Down",
    "generation_timestamp": "2025-01-XX"
  },
  "scores": [
    {
      "sample_id": "0",
      "video_path": "path/to/video.mp4",
      "question": "Is the camera moving downward?",
      "ground_truth_label": "yes",
      "method": "your_method_identifier",
      "score": 0.85,
      "error": null
    }
  ]
}
```

---

## 2. VQA and Retrieval Evaluation

### Score Generation
```bash
# Generate scores for all skills using VQAScore models
python vqa_and_retrieval_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data --combine_tasks --output_dir scores


# Generate scores for a specific skill
python vqa_and_retrieval_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --skill "has_motion" --output_dir scores
```

**Available VQA/Retrieval Skills:** `complex_description`, `confusable_motion`, `has_motion`, `motion_and_steadiness`, `motion_direction`, `motion_speed`, `only_motion`, `scene_dynamics`, `tracking_shot`.

### Evaluation
```bash

# Auto-discoverand evaluate all VQA/retrieval score files in a specific directory
python vqa_and_retrieval_evaluation.py --score_dir scores/ --mode both --output_dir evaluation_results

# Evaluate only VQA metrics with auto-discovery
python vqa_and_retrieval_evaluation.py  --score_dir scores/ --mode vqa

# Evaluate only retrieval metrics with auto-discovery
python vqa_and_retrieval_evaluation.py  --score_dir scores/ --mode retrieval

# Evaluate specific score files explicitly
python vqa_and_retrieval_evaluation.py scores/vqa_retrieval_scores_model1_*.json scores/vqa_retrieval_scores_model2_*.json  --score_dir scores/ --mode both --output_file comparison.json
```

**Auto-Discovery:** When no score files are provided, automatically finds `vqa_retrieval_scores_*.json` files in the specified directory.

**Required Output Format for Custom Methods:** 
```json
{
  "metadata": {
    "method_type": "Your_Method_Name", 
    "model_name": "your_model_name",
    "checkpoint": "optional_checkpoint_path",
    "skill_name": "confusable_motion",
    "task_name": "backward_camera_only_vs_backward_ground_only",
    "split_name": "confusable_motion",
    "generation_timestamp": "2025-01-XX"
  },
  "scores": [
    {
      "sample_id": "0",
      "pos_video": "path/to/positive_video.mp4",
      "neg_video": "path/to/negative_video.mp4",
      "pos_question": "Is the camera moving backward?", 
      "neg_question": "Is the ground moving backward?",
      "method": "your_method_identifier",
      "yes_scores": {
        "pos_text_pos_image": 0.85,
        "pos_text_neg_image": 0.23, 
        "neg_text_pos_image": 0.31,
        "neg_text_neg_image": 0.78
      },
      "no_scores": {
        "pos_text_pos_image": 0.15,
        "pos_text_neg_image": 0.77,
        "neg_text_pos_image": 0.69, 
        "neg_text_neg_image": 0.22
      },
      "error": null
    }
  ]
}
```

---

## 3. Caption Generation and Evaluation

### Caption Generation
```bash
# Generate captions using VQA models for all samples
python caption_generation.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data --output_dir scores/

# Generate captions for a limited number of samples
python caption_generation.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data --sample_size 10 --output_dir scores/
```

### Evaluation
```bash
# Auto-discover and evaluate all caption files in a specific directory
python caption_evaluation.py --score_dir scores/ --output_dir evaluation_results

# Evaluate with GPT-4o judge (requires OpenAI API key)
python caption_evaluation.py --score_dir scores/ --api_key YOUR_OPENAI_API_KEY --output_dir evaluation_results

# Skip GPT-4o judge evaluation
python caption_evaluation.py --score_dir scores/ --no_gpt --output_dir evaluation_results

# Evaluate specific caption files explicitly
python caption_evaluation.py scores/caption_results_qwen2.5-vl-7b_*.json --output_file caption_comparison.json


```

**Auto-Discovery:** When no caption files are provided, automatically finds `caption_results_*.json` files in the specified directory.

**Required Output Format for Custom Methods:** 
```json
{
  "metadata": {
    "method_type": "Your_Method_Name",
    "model_name": "your_model_name", 
    "checkpoint": "optional_checkpoint_path",
    "generation_timestamp": "2025-01-XX"
  },
  "captions": [
    {
      "sample_id": "0",
      "video_path": "path/to/video.mp4",
      "question": "Describe the camera motion in this video.",
      "reference_answer": "The camera pans left while moving forward",
      "method": "your_method_identifier", 
      "generated_caption": "The camera is panning to the left and moving forward smoothly",
      "error": null 
    }
  ]
}
```

---
## Using Custom Methods

To evaluate your own method (classical CV, different LMMs, human evaluation, etc.):

1. **Modify only the generation script (Script 1)s** to output the standardized JSON formats shown above for each task (i.e. as long as the output scores of your method are in the required format, the inputs will be compatible with our *evaluation* script) 
2. **Keep the evaluation scripts (Script 2) unchanged** - they work with any method
3. **Ensure your output includes**:
   - All required fields for each sample (`sample_id`, `error`, etc.)
   - Correct data types (scores as floats, captions as strings, etc.). Note that the error field is used to keep track of failed samples, so you should catch any exceptions and save them as a string in this field.
   - The `metadata` section with `model_name`, `method_type`, and optional `checkpoint`
   - Unique identifiers are automatically generated from `model_name`, `checkpoint`, and `split_name`
4. **File naming conventions for auto-discovery**:
   - Binary classification: `classification_scores_*.json`
   - VQA/Retrieval: `vqa_retrieval_scores_*.json`
   - Caption generation: `caption_results_*.json`

The evaluation scripts will automatically compute all metrics and generate timestamped output files with model and file counts included in the filename, regardless of how the scores/captions were generated. **Note:** The evaluation scripts automatically calculate sample counts (total, successful, failed) from the data, so these fields are no longer required in the output format.

---

## Data Location
All evaluation data is located in the `data/` folder with the following structure:
- `data/binary_classification/` - Binary classification tasks
- `data/vqa_and_retrieval/` - VQA and retrieval tasks organized by skills
- `data/caption_data.json` - Samples and Prompts for caption generation

---