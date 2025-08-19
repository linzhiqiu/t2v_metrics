# Guide to Finetuning Qwen2.5-VL

This guide documents the process for finetuning and evaluating Qwen2.5-VL, primarily for video, but the steps outlinde are also largely the same for images too.

## 1. Setting up LLaMA-Factory (one-time):

We will be using the LLaMA-Factory library, which is set up to allow DPO, LoRA, and full finetuning of Qwen2.5-VL.

To set up the LLaMA-Factory code and environment, follow the directions in this [link](https://github.com/QwenLM/Qwen2.5-VL/tree/35ba6e18636510de4bf8d4a7caaca3f4f5163a84?tab=readme-ov-file#training)

## 2. Preparing your data (for each run):