# CameraBench Video Annotation Scoring

Evaluate Large Multimodal Models on camera motion understanding using [CameraBench](https://linzhiqiu.github.io/papers/camerabench/) - a comprehensive benchmark for understanding camera motion in videos, designed and validated by experts.

## Data Download:

The videos can be downloaded from HuggingFace [here](https://huggingface.co/datasets/syCen/CameraBench). Please download them into the `videos` folder.

## Evaluation Scripts

Run these three scripts to generate CameraBench results:

### 1. LMM Binary Classification
```bash
python binary_classification.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' 
```

### 2. LMM VQA and Retrieval

```bash
python cam_motion_vqa_and_retrieval.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' 
```

### 3. Captioning 

```bash
python captioning.py --model 'qwen2.5-vl-7b' --checkpoint 'chancharikm/qwen2.5-vl-7b-cam-motion' 
```

```bibtex
@article{lin2025camerabench,
  title={Towards Understanding Camera Motions in Any Video},
  author={Lin, Zhiqiu and Cen, Siyuan and Jiang, Daniel and Karhade, Jay and Wang, Hewei and Mitra, Chancharik and Ling, Tiffany and Huang, Yuhan and Liu, Sifan and Chen, Mingyu and Zawar, Rushikesh and Bai, Xue and Du, Yilun and Gan, Chuang and Ramanan, Deva},
  journal={arXiv preprint arXiv:2504.15376},
  year={2025},
}
```

For more details, visit: https://linzhiqiu.github.io/papers/camerabench/