# Evaluation Scripts

CameraBench uses a **method-agnostic evaluation framework** that separates score/prediction generation (Script 1) from evaluation logic (Script 2). This allows any method to be evaluated on CameraBench (VLMs, task specific models, SfMs, and even human evaluation) by simply replacing the functionality of our Script 1. The method we use in the paper is VQAScore with VLMs for binary classification and retrieval and standard text generation with VLMs for captioning. 

## Two-Stage Evaluation Process

### Script 1: Score/Prediction Generation (Method-Specific)
Generate scores or captions using your chosen method and save them in standardized formats.

### Script 2: Evaluation (Method-Agnostic) 
Compute metrics from the standardized files using evaluation scripts that work with any method.

To use a different method (classical CV, other LMMs, or even human evaluators), you only need to modify the first script in each pair to output the standardized format. The evaluation logic remains unchanged.

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
# Generate scores using VQAScore models
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --data_dir data/binary_classification --output_dir scores

# Generate scores for specific splits only
python binary_classification_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --splits motion_coherence lighting_quality --output_dir scores
```

### Evaluation
```bash
# Evaluate all score files with plots
python binary_classification_evaluation.py scores/*_binary_scores_*.json --plots --output_dir evaluation_results

# Evaluate specific score files
python binary_classification_evaluation.py scores/binary_scores_qwen2.5-vl-7b_motion_coherence_*.json --output_file motion_results.json
```

**Required Output Format for Custom Methods:**
```json
{
  "metadata": {
    "method_type": "Your_Method_Name",
    "model_name": "your_model_name",
    "split_name": "motion_coherence",
    "generation_timestamp": "2025-01-XX"
  },
  "scores": [
    {
      "sample_id": "0",
      "video_path": "path/to/video.mp4",
      "question": "Is the motion smooth?",
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

# Generate scores for specific skill
python vqa_and_retrieval_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --skill "Motion & Steadiness" --output_dir scores

# Generate with custom question template
python vqa_and_retrieval_vlm_scores.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' --question_template "Answer with only Yes or No: {}" --combine_tasks
```

### Evaluation
```bash
# Evaluate both VQA and retrieval metrics
python vqa_and_retrieval_evaluation.py scores/*_vqa_retrieval_scores_*.json --mode both --output_dir evaluation_results

# Evaluate only VQA metrics
python vqa_and_retrieval_evaluation.py scores/*_vqa_retrieval_scores_*.json --mode vqa

# Evaluate only retrieval metrics  
python vqa_and_retrieval_evaluation.py scores/*_vqa_retrieval_scores_*.json --mode retrieval

# Compare specific methods
python vqa_and_retrieval_evaluation.py scores/model1_*.json scores/model2_*.json --mode both --output_file comparison.json
```

**Required Output Format for Custom Methods:**
```json
{
  "metadata": {
    "method_type": "Your_Method_Name", 
    "model_name": "your_model_name",
    "skill_name": "Motion & Steadiness",
    "task_name": "smooth_motion",
    "generation_timestamp": "2025-01-XX"
  },
  "scores": [
    {
      "sample_id": "0",
      "pos_video": "path/to/positive_video.mp4",
      "neg_video": "path/to/negative_video.mp4",
      "pos_question": "Is the motion smooth?", 
      "neg_question": "Is the motion choppy?",
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

## Using Custom Methods

To evaluate your own method (classical CV, different LMMs, human evaluation, etc.):

1. **Keep the evaluation scripts unchanged** - they work with any method
2. **Modify only the generation scripts** to output the standardized JSON format shown above
3. **Ensure your output includes**:
   - All required fields for each sample (`sample_id`, `error`, etc.)
   - Correct data types (scores as floats, captions as strings, and errors as strings or nulls)
   - The `metadata` section, which is there to help with organizing outputs

The evaluation scripts will automatically compute all metrics regardless of how the scores/captions were generated.
