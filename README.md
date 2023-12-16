
## **VQAScore for Text-to-Image Evaluation**  [[Project Page]](https://linzhiqiu.github.io/papers/vqascore/)

TODO:
1. pip install
2. evaluate on Winoground
3. advanced tutorial on how to add your own models
4. pick better teaser images because VQAScore still fail on first Winoground sample


## **VQAScore for Text-to-Image Evaluation**  
<!-- [Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Oliver Wang](http://www.oliverwang.info/). In [CVPR](https://arxiv.org/abs/1801.03924), 2018.

<img src='https://richzhang.github.io/PerceptualSimilarity/index_files/fig1_v2.jpg' width=1200> -->

### Quick start

**I haven't set up the pip package yet!! Please follow the steps below first to install environment**:
```
conda create -n t2i python=3.10 -y
conda activate t2i
conda install pip -y
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Run `pip install t2i_metrics`. The following Python code is all you need. 

```python
import t2i_metrics
score_func_clip_flant5 = t2i_metrics.VQAScore(model='clip-flant5-xxl') # CLIP-FlanT5 is our best scoring metric
score_func_llava = t2i_metrics.VQAScore(model='llava-v1.5-13b') # LLaVA-1.5 is the second best
score_func_clip = t2i_metrics.CLIPScore(model='openai:ViT-L-14') # we include clipscore for easier ablation

# For a single (image, text) pair
image = "images/test0.jpg" # an image path in string format
text = "a young person kisses an old person"
score = score_func_clip_flant5(images=[image], texts=[text])

# Alternatively, if you want to calculate the pairwise similarity scores 
# between M images and N texts, run the following to return a M x N score tensor.
images = ["images/test0.jpg", "images/test1.jpg"]
texts = ["an old person kisses a young person", "a young person kisses an old person"]
scores = score_func_clip_flant5(images=images, texts=texts) # scores[i][j] is the score between image i and text j
```

### Notes on GPU and cache
- **GPU usage**: The above scripts will by default use the first cuda device on your machine. We recommend using 40GB GPU for the largest VQA models such as `clip-flant5-xxl` and `llava-v1.5-13b`. If you have limited GPU memory, consider using smaller models such as `clip-flant5-xl` and `llava-v1.5-7b`.
- **Cache directory**: You can change the cache folder (default is `~/.cache/`) by updating `HF_CACHE_DIR` in [t2i_metrics/constants.py](t2i_metrics/constants.py).

### Development mode please run
```
pip install -e .
```


### Batch processing for massive image-text pairs
While the above script can be applied to most scenarios, if you have a large dataset of M images x N texts, then you can optionally speed up inference using the following batch processing script. 
```python
import t2i_metrics
score_func_clip_flant5 = t2i_metrics.VQAScore(model='clip-flant5-xxl')

# Each dictionary must have the same number of images and texts
dataset = [
  {'images': ["images/test0.jpg", "images/test1.jpg"], 'texts': ["an old person kisses a young person", "a young person kisses an old person"]},
  {'images': ["images/test0.jpg", "images/test1.jpg"], 'texts': ["an old person kissing a young person", "a young person kissing an old person"]},
  #...
]
scores = score_func_clip_flant5.batch_forward(dataset=dataset, batch_size=16) # will return n_data x 2 x 2 score tensor

# Each dictionary must have the same number of images and texts
dataset = [
  {'images': ["images/sdxl_0.jpg", "images/dalle3_0.jpg", "images/deepfloyd_0.jpg", "images/imagen2_0.jpg"], 'texts': ["an old person kisses a young person"]},
  {'images': ["images/sdxl_1.jpg", "images/dalle3_1.jpg", "images/deepfloyd_1.jpg", "images/imagen2_1.jpg"], 'texts': ["a young person kissing an old person"]},
  #...
]
scores = score_func_clip_flant5.batch_forward(dataset=dataset, batch_size=16) # will return n_data x 4 x 1 score tensor
```

### Advanced usage: Specifying your own question and answer 
For VQAScore, the question and answer can affect the final performance. We provide a simple default template for each model by default. For example, CLIP-FlanT5 and LLaVA-1.5 uses the below template which can be found at [t2i_metrics/models/vqascore_models/clip_t5_model.py](t2i_metrics/models/vqascore_models/clip_t5_model.py) (we ignored the prepended system message for simplicity):

```python
default_question_template = "Is the image showing '{}'? Please answer yes or no."
default_answer_template = "Yes"
```

You can specify your own template by passing in `question_template` and `answer_template` to `forward()` or `batch_forward()` function:

```python
# An alternative template for VQAScore
question_template = "Does the image show '{}'? Please answer yes or no."
answer_template = "Yes"

images = ["images/test0.jpg", "images/test1.jpg"]
texts = ["an old person kisses a young person", "a young person kisses an old person"]
scores = score_func_clip_flant5(images=images, texts=texts,
                                question_template=question_template,
                                answer_template=answer_template)

# If you want to instead compute P(caption|image) (VisualGPTScore), then you can use the below template
vgpt_question_template = "" # no question
vgpt_answer_template = "{}" # simply calculate the P(caption)

dataset = [
  {'images': ["images/sdxl_0.jpg", "images/dalle3_0.jpg", "images/deepfloyd_0.jpg", "images/imagen2_0.jpg"], 'texts': ["an old person kisses a young person"]},
  {'images': ["images/sdxl_1.jpg", "images/dalle3_1.jpg", "images/deepfloyd_1.jpg", "images/imagen2_1.jpg"], 'texts': ["a young person kissing an old person"]},
  #...
]
scores = score_func_clip_flant5.batch_forward(dataset=dataset,
                                              batch_size=16,
                                              question_template=vgpt_question_template,
                                              answer_template=vgpt_answer_template)
```

# TODO: Update below
More thorough information about variants is below. This repository contains our **perceptual metric (LPIPS)** and **dataset (BAPPS)**. It can also be used as a "perceptual loss". This uses PyTorch; a Tensorflow alternative is [here](https://github.com/alexlee-gk/lpips-tensorflow).

1. 

**Table of Contents**<br>
1. [Learned Perceptual Image Patch Similarity (LPIPS) metric](#1-learned-perceptual-image-patch-similarity-lpips-metric)<br>
   a. [Basic Usage](#a-basic-usage) If you just want to run the metric through command line, this is all you need.<br>
   b. ["Perceptual Loss" usage](#b-backpropping-through-the-metric)<br>
   c. [About the metric](#c-about-the-metric)<br>
2. [Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset](#2-berkeley-adobe-perceptual-patch-similarity-bapps-dataset)<br>
   a. [Download](#a-downloading-the-dataset)<br>
   b. [Evaluation](#b-evaluating-a-perceptual-similarity-metric-on-a-dataset)<br>
   c. [About the dataset](#c-about-the-dataset)<br>
   d. [Train the metric using the dataset](#d-using-the-dataset-to-train-the-metric)<br>

## (0) Dependencies/Setup

### Installation
- Install PyTorch 1.0+ and torchvision fom http://pytorch.org

```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/richzhang/PerceptualSimilarity
cd PerceptualSimilarity
```

## (1) Learned Perceptual Image Patch Similarity (LPIPS) metric

Evaluate the distance between image patches. **Higher means further/more different. Lower means more similar.**

### (A) Basic Usage

#### (A.I) Line commands

Example scripts to take the distance between 2 specific images, all corresponding pairs of images in 2 directories, or all pairs of images within a directory:

```
python lpips_2imgs.py -p0 imgs/ex_ref.png -p1 imgs/ex_p0.png --use_gpu
python lpips_2dirs.py -d0 imgs/ex_dir0 -d1 imgs/ex_dir1 -o imgs/example_dists.txt --use_gpu
python lpips_1dir_allpairs.py -d imgs/ex_dir_pair -o imgs/example_dists_pair.txt --use_gpu
```

#### (A.II) Python code

File [test_network.py](test_network.py) shows example usage. This snippet is all you really need.

```python
import lpips
loss_fn = lpips.LPIPS(net='alex')
d = loss_fn.forward(im0,im1)
```

Variables ```im0, im1``` is a PyTorch Tensor/Variable with shape ```Nx3xHxW``` (```N``` patches of size ```HxW```, RGB images scaled in `[-1,+1]`). This returns `d`, a length `N` Tensor/Variable.

Run `python test_network.py` to take the distance between example reference image [`ex_ref.png`](imgs/ex_ref.png) to distorted images [`ex_p0.png`](./imgs/ex_p0.png) and [`ex_p1.png`](imgs/ex_p1.png). Before running it - which do you think *should* be closer?

**Some Options** By default in `model.initialize`:
- By default, `net='alex'`. Network `alex` is fastest, performs the best (as a forward metric), and is the default. For backpropping, `net='vgg'` loss is closer to the traditional "perceptual loss".
- By default, `lpips=True`. This adds a linear calibration on top of intermediate features in the net. Set this to `lpips=False` to equally weight all the features.

### (B) Backpropping through the metric

File [`lpips_loss.py`](lpips_loss.py) shows how to iteratively optimize using the metric. Run `python lpips_loss.py` for a demo. The code can also be used to implement vanilla VGG loss, without our learned weights.

### (C) About the metric

**Higher means further/more different. Lower means more similar.**

We found that deep network activations work surprisingly well as a perceptual similarity metric. This was true across network architectures (SqueezeNet [2.8 MB], AlexNet [9.1 MB], and VGG [58.9 MB] provided similar scores) and supervisory signals (unsupervised, self-supervised, and supervised all perform strongly). We slightly improved scores by linearly "calibrating" networks - adding a linear layer on top of off-the-shelf classification networks. We provide 3 variants, using linear layers on top of the SqueezeNet, AlexNet (default), and VGG networks.

If you use LPIPS in your publication, please specify which version you are using. The current version is 0.1. You can set `version='0.0'` for the initial release.

## (2) Berkeley Adobe Perceptual Patch Similarity (BAPPS) dataset

### (A) Downloading the dataset

Run `bash ./scripts/download_dataset.sh` to download and unzip the dataset into directory `./dataset`. It takes [6.6 GB] total. Alternatively, run `bash ./scripts/download_dataset_valonly.sh` to only download the validation set [1.3 GB].
- 2AFC train [5.3 GB]
- 2AFC val [1.1 GB]
- JND val [0.2 GB]  

### (B) Evaluating a perceptual similarity metric on a dataset

Script `test_dataset_model.py` evaluates a perceptual model on a subset of the dataset.

**Dataset flags**
- `--dataset_mode`: `2afc` or `jnd`, which type of perceptual judgment to evaluate
- `--datasets`: list the datasets to evaluate
    - if `--dataset_mode 2afc`: choices are [`train/traditional`, `train/cnn`, `val/traditional`, `val/cnn`, `val/superres`, `val/deblur`, `val/color`, `val/frameinterp`]
    - if `--dataset_mode jnd`: choices are [`val/traditional`, `val/cnn`]
    
**Perceptual similarity model flags**
- `--model`: perceptual similarity model to use
    - `lpips` for our LPIPS learned similarity model (linear network on top of internal activations of pretrained network)
    - `baseline` for a classification network (uncalibrated with all layers averaged)
    - `l2` for Euclidean distance
    - `ssim` for Structured Similarity Image Metric
- `--net`: [`squeeze`,`alex`,`vgg`] for the `net-lin` and `net` models; ignored for `l2` and `ssim` models
- `--colorspace`: choices are [`Lab`,`RGB`], used for the `l2` and `ssim` models; ignored for `net-lin` and `net` models

**Misc flags**
- `--batch_size`: evaluation batch size (will default to 1)
- `--use_gpu`: turn on this flag for GPU usage

An example usage is as follows: `python ./test_dataset_model.py --dataset_mode 2afc --datasets val/traditional val/cnn --model lpips --net alex --use_gpu --batch_size 50`. This would evaluate our model on the "traditional" and "cnn" validation datasets.

### (C) About the dataset

The dataset contains two types of perceptual judgements: **Two Alternative Forced Choice (2AFC)** and **Just Noticeable Differences (JND)**.

**(1) 2AFC** Evaluators were given a patch triplet (1 reference + 2 distorted). They were asked to select which of the distorted was "closer" to the reference.

Training sets contain 2 judgments/triplet.
- `train/traditional` [56.6k triplets]
- `train/cnn` [38.1k triplets]
- `train/mix` [56.6k triplets]

Validation sets contain 5 judgments/triplet.
- `val/traditional` [4.7k triplets]
- `val/cnn` [4.7k triplets]
- `val/superres` [10.9k triplets]
- `val/deblur` [9.4k triplets]
- `val/color` [4.7k triplets]
- `val/frameinterp` [1.9k triplets]

Each 2AFC subdirectory contains the following folders:
- `ref`: original reference patches
- `p0,p1`: two distorted patches
- `judge`: human judgments - 0 if all preferred p0, 1 if all humans preferred p1

**(2) JND** Evaluators were presented with two patches - a reference and a distorted - for a limited time. They were asked if the patches were the same (identically) or different. 

Each set contains 3 human evaluations/example.
- `val/traditional` [4.8k pairs]
- `val/cnn` [4.8k pairs]

Each JND subdirectory contains the following folders:
- `p0,p1`: two patches
- `same`: human judgments: 0 if all humans thought patches were different, 1 if all humans thought patches were same

### (D) Using the dataset to train the metric

See script `train_test_metric.sh` for an example of training and testing the metric. The script will train a model on the full training set for 10 epochs, and then test the learned metric on all of the validation sets. The numbers should roughly match the **Alex - lin** row in Table 5 in the [paper](https://arxiv.org/abs/1801.03924). The code supports training a linear layer on top of an existing representation. Training will add a subdirectory in the `checkpoints` directory.

You can also train "scratch" and "tune" versions by running `train_test_metric_scratch.sh` and `train_test_metric_tune.sh`, respectively. 

## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
```

## Acknowledgements

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository. The average precision (AP) code is borrowed from the [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py) repository. [Angjoo Kanazawa](https://github.com/akanazawa), [Connelly Barnes](http://www.connellybarnes.com/work/), [Gaurav Mittal](https://github.com/g1910), [wilhelmhb](https://github.com/wilhelmhb), [Filippo Mameli](https://github.com/mameli), [SuperShinyEyes](https://github.com/SuperShinyEyes), [Minyoung Huh](http://people.csail.mit.edu/minhuh/) helped to improve the codebase.
