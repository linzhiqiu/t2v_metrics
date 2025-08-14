# CameraBench Video Annotation Scoring

Evaluate Large Multimodal Models on camera motion understanding using [CameraBench](https://linzhiqiu.github.io/papers/camerabench/) - a comprehensive benchmark for understanding camera motion in videos, designed and validated by experts.

## Data Download:

The videos can be downloaded from HuggingFace [here](https://huggingface.co/datasets/syCen/CameraBench). Please download them into the `videos` folder.

## Evaluation Scripts

Run these three scripts to generate complete CameraBench results:

### 1. LMM Binary Classification
```bash
python cam_motion_binary_classification.py --score_model 'qwen2.5-vl-7b-cambench' # 32B and 72B versions available
```

### 2. LMM VQA and Retrieval
To obtain our VQA and Retrieval results, please run **both** of the scripts that follow.
```bash
python cam_motion_vqa_and_retrieval.py --score_model 'qwen2.5-vl-7b-cambench'
python cam_motion_vqa_and_retrieval_complex_caption.py --score_model 'qwen2.5-vl-7b-cambench'
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