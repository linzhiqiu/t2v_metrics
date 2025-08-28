# Evaluation Scripts

CameraBench uses a **method-agnostic evaluation framework** that separates score/prediction generation (Script 1) from evaluation logic (Script 2). This allows any method to be evaluated on CameraBench (VLMs, task specific models, SfMs, and even human evaluation) by simply replacing the functionality of our Script 1. The method we use in the paper is VQAScore with VLMs for binary classification and retrieval and standard text generation with VLMs for captioning. 

## Two-Stage Evaluation Process

### Script 1: Score/Prediction Generation (Method-Specific)
Generate scores or captions using your chosen method and save them in standardized formats.

### Script 2: Evaluation (Method-Agnostic) 
Compute metrics from the standardized files using evaluation scripts that work with any method.

To use a different method (classical CV, other LMMs, or even human evaluators), you only need to modify/recreate the first script in each pair to output the standardized format. The evaluation logic in the 2nd script remains unchanged.

---

## Data Location
All evaluation data is located in the `data/` folder with the following structure:
- `data/binary_classification/` - Binary classification tasks
- `data/vqa_and_retrieval/` - VQA and retrieval tasks organized by skills
- `caption_data.json` - Samples and Prompts for caption generation

---

## 1. Binary Classification Evaluation

### Score Generation
```bash
# Generate scores using VQAScore models for all splits
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data/binary_classification --output_dir scores

# Generate scores for specific splits only
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --splits Move_Down Move_Up Pan_Left Pan_Right --output_dir scores
```

**Available Binary Classification Splits:**
- Movement: `Move_Down`, `Move_In`, `Move_Left`, `Move_Out`, `Move_Right`, `Move_Up`
- Panning: `Pan_Left`, `Pan_Right`  
- Rotation: `Roll_Clockwise`, `Roll_Counterclockwise`
- Tilting: `Tilt_Down`, `Tilt_Up`
- Zooming: `Zoom_In`, `Zoom_Out`
- Static: `Static`

### Evaluation
```bash

# Auto-discover files in a specific directory
python binary_classification_evaluation.py --score_dir path/to/scores --plots --output_dir evaluation_results

# Evaluate specific score files explicitly
python binary_classification_evaluation.py scores/vqa_scores_qwen2.5-vl-7b_*_Move_Down_*.json scores/vqa_scores_qwen2.5-vl-7b_*_Move_Up_*.json --plots --output_file movement_results.json
```

**Auto-Discovery:** When no score files are provided, automatically finds `vqa_scores_*.json` files in the specified directory.

**Required Output Format for Custom Methods:** (i.e. as long as the output scores of your method are in the following format, the inputs will be compatible with our *evaluation* script)
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
  ],
  "total_samples": 1000,
  "successful_samples": 995,
  "failed_samples": 5
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

**Available VQA/Retrieval Skills:**
- `complex_description` - Complex scene descriptions and captions
- `confusable_motion` - Confusable camera motion patterns
- (Additional skills based on your directory structure)

### Evaluation
```bash

# Auto-discoverand evaluate all VQA/retrieval score files in a specific directory
python vqa_and_retrieval_evaluation.py --score_dir path/to/scores --mode both --output_dir evaluation_results

# Evaluate only VQA metrics with auto-discovery
python vqa_and_retrieval_evaluation.py  --score_dir path/to/scores --mode vqa

# Evaluate only retrieval metrics with auto-discovery
python vqa_and_retrieval_evaluation.py  --score_dir path/to/scores --mode retrieval

# Evaluate specific score files explicitly
python vqa_and_retrieval_evaluation.py scores/vqa_retrieval_scores_model1_*.json scores/vqa_retrieval_scores_model2_*.json  --score_dir path/to/scores --mode both --output_file comparison.json
```

**Auto-Discovery:** When no score files are provided, automatically finds `vqa_retrieval_scores_*.json` files in the specified directory.

**Required Output Format for Custom Methods:** (i.e. as long as the output scores of your method are in the following format, the inputs will be compatible with our *evaluation* script)
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
  ],
  "total_samples": 500,
  "successful_samples": 498,
  "failed_samples": 2
}
```

---

## 3. Captioning Evaluation

### Caption Generation
```bash
# Generate captions using VLM models
python caption_generation.py --models 'qwen2.5-vl-7b:chancharikm/qwen2.5-vl-7b-cam-motion' 'gpt-4o' --input data/caption_data.json --output_dir captions --sample_size 100

# Generate captions for single model (backwards compatibility)
python caption_generation.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --input data/caption_data.json --output_dir captions --sample_size 100
```

### Evaluation
```bash
# Evaluate all caption files with GPT-4o judge
python caption_evaluation.py captions/*_captions_*.json --api_key YOUR_OPENAI_API_KEY --output_dir evaluation_results

# Evaluate without GPT-4o judge (faster, but fewer metrics)
python caption_evaluation.py captions/*_captions_*.json --no_gpt --output_dir evaluation_results

# Generate detailed Excel report
python caption_evaluation.py captions/*_captions_*.json --detailed_excel --output_dir evaluation_results
```

**Required Output Format for Custom Methods:**
```json
{
  "metadata": {
    "method_type": "Your_Caption_Method",
    "model_name": "your_model_name",
    "checkpoint": "optional_checkpoint_path",
    "generation_timestamp": "2025-01-XX"
  },
  "captions": [
    {
      "sample_id": "0",
      "video": "path/to/video.mp4",
      "question": "Describe the camera motion in this video.",
      "method": "your_method_identifier",
      "caption": "The camera pans smoothly from left to right",
      "reference": "Smooth left-to-right panning motion",
      "error": null
    }
  ],
  "total_samples": 100,
  "successful_samples": 98,
  "failed_samples": 2
}
```

---

## Output File Formats

Both evaluation scripts now produce consistent output formats with top-level metrics for easy access:

### Binary Classification Output
```json
{
  "evaluation_timestamp": "2025-01-XX",
  "overall_average_precision": 0.85,
  "overall_roc_auc": 0.92,
  "total_splits": 12,
  "evaluated_splits": 12,
  "overall_statistics": {
    "mean_average_precision": 0.85,
    "std_average_precision": 0.03,
    "mean_roc_auc": 0.92,
    "std_roc_auc": 0.02,
    "evaluated_splits": 12
  },
  "results_by_split": { ... }
}
```

### VQA/Retrieval Output
```json
{
  "evaluation_timestamp": "2025-01-XX",
  "evaluation_mode": "both",
  "overall_binary_acc": 0.78,
  "overall_question_acc": 0.82,
  "overall_retrieval_text": 0.75,
  "overall_retrieval_image": 0.80,
  "overall_retrieval_group": 0.68,
  "skill_based_retrieval_text": 0.77,
  "skill_based_retrieval_image": 0.83,
  "skill_based_retrieval_group": 0.71,
  "total_splits": 5,
  "evaluated_splits": 5,
  "overall_statistics": { ... },
  "results_by_split": { ... }
}
```

---

## Using Custom Methods

To evaluate your own method (classical CV, different LMMs, human evaluation, etc.):

1. **Keep the evaluation scripts unchanged** - they work with any method
2. **Modify only the generation scripts** to output the standardized JSON format shown above
3. **Ensure your output includes**:
   - All required fields for each sample (`sample_id`, `error`, etc.)
   - Correct data types (scores as floats, captions as strings, and errors as strings or nulls)
   - The `metadata` section with `model_name`, `method_type`, and optional `checkpoint`
   - Unique identifiers are automatically generated from `model_name`, `checkpoint`, and `split_name`

4. **File naming conventions for auto-discovery**:
   - Binary classification: `vqa_scores_*.json`
   - VQA/Retrieval: `vqa_retrieval_scores_*.json`

The evaluation scripts will automatically compute all metrics and generate timestamped output files with model and file counts included in the filename, regardless of how the scores/captions were generated.
