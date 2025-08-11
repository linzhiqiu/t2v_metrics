# CameraBench Video Annotation Scoring

Evaluate Large Multimodal Models on camera motion understanding using [CameraBench](https://linzhiqiu.github.io/papers/camerabench/) - a comprehensive benchmark for understanding camera motion in videos, designed and validated by experts.

## Quick Setup

```bash
git clone https://github.com/linzhiqiu/video_annotation/
cd video_annotation
pip install -e .
python download.py --json_path video_data/20250227_0324ground_only/videos.json --label_collections cam_motion
```

## Configuration

Update the data paths in scripts to match your setup:

```python
ROOT = Path("/path/to/your/video_annotation")  # Update this path
VIDEO_ROOT = Path("/path/to/your/video_annotation/videos")  # Update this path
```

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
### 3. SfM Evaluation (TBD)

## Citation

```bibtex
@article{lin2025camerabench,
  title={Towards Understanding Camera Motions in Any Video},
  author={Lin, Zhiqiu and Cen, Siyuan and Jiang, Daniel and Karhade, Jay and Wang, Hewei and Mitra, Chancharik and Ling, Tiffany and Huang, Yuhan and Liu, Sifan and Chen, Mingyu and Zawar, Rushikesh and Bai, Xue and Du, Yilun and Gan, Chuang and Ramanan, Deva},
  journal={arXiv preprint arXiv:2504.15376},
  year={2025},
}
```

For more details, visit: https://linzhiqiu.github.io/papers/camerabench/