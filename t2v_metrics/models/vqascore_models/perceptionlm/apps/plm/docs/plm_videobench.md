# PLM-VideoBench
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20PLM&#8209;VideoBench-BenchMark-blue)](https://huggingface.co/datasets/facebook/PLM-VideoBench)

As part of our PLM-release, we are releasing a comprehensive set of video benchmarks (grouped as `PLM-VideoBench`) for detailed video understanding. PLM-VideoBench includes the following sub-benchmarks,
1. **Fine-Grained Question Answering (FGQA):** In this task, a model must answer a multiple-choice question (MCQ)
that probes fine-grained activity understanding.
2. **Smart Glasses Question Answering (SGQA):** In this task, a model must answer open-ended questions about
activities and objects visible in an egocentric video stream recorded by a Meta VR Glasses.
3. **Video Region Captioning (RCap):** In this task, the model must generate a detailed description of an event
involving a subject of interest in the video. 
4. **Region Temporal Localization (RTLoc):** In this task, the model must identify the precise time interval within the video when the specified event takes place for the given subject.
5. **Region Dense Video Captioning (RDCap):** In this task, a model must generate a detailed description of all events involving a specific subject of interest in a video.

> [!TIP]
> We have added all `PLM-VideoBench` tasks to [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/plm_videobench). This makes it easy to reproduce PLM results and also allows other models to be tested on the benchmarks.

You can use the following command to evaluate PLM on PLM-VideoBench.

```shell

# Use facebook/Perception-LM-1B for 1B parameters model and facebook/Perception-LM-8B for 8B parameters model.
CHECKPOINTS_PATH=facebook/Perception-LM-3B.

# PLM-VideoBench Tasks
SELECTED_TASK=fgqa_test,sgqa_test,rtloc_test,rcap_test,rdcap_test
OUTPUT_PATH="plm_videobench_evaluation"

accelerate launch --num_processes=8 \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH \
--tasks $TASKS \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path $OUTPUT_PATH
```

## Results

We evaluate PLM against baselines on PLM-VideoBench and
report breakdowns. We report human performance in the first row.
| Model            | FGQA (MBacc) | SGQA (Acc) | RDCap (SODA) | RCap (Score) | RTLoc (meanR) | Avg. |
|------------------|------|------|------------|------------|-------------|------|
| <font color="blue">Human perf.</font>      | <font color="blue">90.9</font>   | <font color="blue">67.9</font>   | <font color="blue">66.6</font>  | <font color="blue">53.9</font>       | <font color="blue">67.8</font>       | <font color="blue">73.9</font>  |
| GPT-4o           | 61.2 | **63.7** | 20.9       | 35.7       | 33.1        | 51.6 |
| Gemini 1.5 Pro   | 57.1 | 49.9 | 14.4       | 33.1       | 27.6        | 44.0 |
| Gemini 2.0 Flash | 58.7 | 44.8 | 13.2       | 30.9       | 27.6        | 42.5 |
| LLaVA-OV-7B      | 40.2 | 41.5 | 4.7        | 24.4       | 13.9        | 32.0 |
| Qwen2VL-7B       | 49.2 | 44.5 | 4.1        | 17.6       | 15.1        | 35.3 |
| Qwen2.5VL-7B     | 49.8 | 43.0 | 2.5        | 21.5       | 10.7        | 34.8 |
| InternVL2-8B     | 47.7 | 45.9 | 1.2        | 21.5       | 11.6        | 35.0 |
| InternVL2.5-8B   | 53.7 | 48.3 | 5.7        | 26.1       | 8.8         | 38.5 |
| PLM-8B           | **67.7** | 46.2 | **52.8**   | **46.6**   | **59.1**    | **55.6** |
