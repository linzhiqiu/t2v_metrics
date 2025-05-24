# Example to Finetune PLM on New Data

We provide a step-by-step walkthrough for finetuning PLM on a custom dataset based on the high-level instructions in [training.md](training.md). For this example, we will finetune PLM-8B on a specific domain ([Radiology images](https://huggingface.co/datasets/unsloth/Radiology_mini)) and compare model performance before and after finetuning.

### Setup
Install required packages:
```bash
pip install datasets tqdm
```


### 1. Download dataset and prepare for training

``` python
import json
import os
import tqdm
from datasets import load_dataset

def convert_to_training_jsonl(dataset, split):

    out_dir = "apps/plm/dummy_datasets/Radiology_mini"
    os.makedirs(f"{out_dir}/images", exist_ok=True)

    parsed_data = []
    for entry in tqdm.tqdm(dataset[split]):

        # save image
        image_path = f"{out_dir}/images/{entry["image_id"]}.png"
        entry["image"].save(image_path)

        # create training conversation template
        conversations = [
            {"from": "human", "value": "You are an expert radiographer. Describe accurately what you see in this image."},
            {"from": "assistant", "value": entry["caption"]}
        ]

        parsed_data.append({
            "image": f"{entry["image_id"]}.png",
            "conversations": conversations,
        })

    # Write jsonl for training / evaluation
    with open(f"{out_dir}/{split}.jsonl", "w") as f:
        for entry in parsed_data:
            f.write(json.dumps(entry) + "\n")


dataset = load_dataset("unsloth/Radiology_mini")
convert_to_training_jsonl(dataset, "train")
convert_to_training_jsonl(dataset, "test")
```

After running this code, the training data will be ready for use with the codebase:
```
apps/plm/dummy_datasets/Radiology_mini
├── train.jsonl
├── test.jsonl
├── images
│   ├── ROCOv2_2023_test_000022.png
│   ├── ROCOv2_2023_train_059888.png
│   ├── ...
```

where each data jsonl will contain data in the required training format.
```
# train.jsonl
{"image": "ROCOv2_2023_train_054311.png", "conversations": [{"from": "human", "value": "You are an expert radiographer. Describe accurately what you see in this image."}, {"from": "assistant", "value": "Panoramic radiography shows an osteolytic lesion in the right posterior maxilla with resorption of the floor of the maxillary sinus (arrows)."}]}
{"image": "ROCOv2_2023_train_058916.png", "conversations": [{"from": "human", "value": "You are an expert radiographer. Describe accurately what you see in this image."}, {"from": "assistant", "value": "ERCP showing distal CBD compression. ERCP - endoscopic retrograde cholangiopancreatography; CBD - common bile duct"}]}
...
```


### 2. Add dataset config to configs/datasets.yaml
Point to the newly created data in [configs/datasets.yaml](../configs/datasets.yaml) by adding these lines at the bottom.
```
radiology_finetune:
    annotation: apps/plm/dummy_datasets/Radiology_mini/train.jsonl
    root_dir: apps/plm/dummy_datasets/Radiology_mini/images
```

### 3. Copy and modify the provided finetuning config
The stage # 3 configs can be used to further finetune PLM [configs/stage_3](../configs/stage_3). 
```bash
cp apps/plm/configs/stage_3/plm_8b.yaml apps/plm/configs/finetune/plm_8b_custom.yaml 
```

Copy the config and modify the fields below.
```yaml
# Set the path to save checkpoints to
dump_dir: checkpoints/finetune_example/

# Total number of training iterations
steps: 500

# Pointer to previously created datamix. Ideally, you would incorporate the new data into a larger datamix
# but for now, we finetune only on this data
data:
    datamix: radiology_finetune:1

# Pointer to the initial model weights
checkpoint:
    init_ckpt_path: facebook/Perception-LM-8B
```

Various other parameters can be changed such as learning rate, batch_size, etc. See comments in [configs/stage_3/plm_8b.yaml](../configs/stage_3/plm_8b.yaml) for details.

### 4. Finetune the model
Finetune a model on a single node. For multi-node training, refer to the main [training.md](training.md) doc.
```
torchrun --nproc-per-node 8 -m apps.plm.train \
    config=apps/plm/configs/finetune/plm_8b_custom.yaml 
```

This will start training and save checkpoints, logs and configs in the previously specified `dump_dir`.
```
checkpoints/finetune_example/
├── checkpoints
│   └── 0000000500
│       ├── __0_0.distcp
│       ├── __1_0.distcp
│       ├── ...
│       ├── params.json
│       ├── train_state_00000.json
│       ├── train_state_00001.json
│       ├── ...
├── config.yaml
├── metrics.jsonl
└── train.log
```

### 5. Consolidate the checkpoint
Models trained with FSDP require their weights to be consolidated before inference to create `consolidated.pth`.
```bash
python apps/plm/consolidate.py --ckpt checkpoints/finetune_example/checkpoints/0000000500/
```

### 6. Test and compare model generation
Use the provided generate helper script to compare the base model (before finetuning) to the finetuned version on an unseen test image from the same dataset. 

```bash
python apps/plm/generate.py \
    --ckpt facebook/Perception-LM-8B \
    --media_type image \
    --media_path apps/plm/dummy_datasets/Radiology_mini/images/ROCOv2_2023_test_000022.png \
    --question 'You are an expert radiographer. Describe accurately what you see in this image.'

# Generation:
# The image is a medical scan of a person's abdomen, likely an MRI or CT scan. The scan shows the internal organs of the abdomen, including the liver, stomach, and intestines. The liver is located on the left side of the image, and it appears to be slightly enlarged. The stomach is located in the center of the image, and it appears to be normal in size. The intestines are located on the right side of the image, and they appear to be normal in size and shape. There are no visible abnormalities or tumors in the image. The scan is in black and white, with the organs appearing in shades of gray. The background of the image is black, which helps to highlight the details of the organs. Overall, the image suggests that the person's abdominal organs are healthy and normal.
```


```bash
python apps/plm/generate.py \
    --ckpt checkpoints/finetune_example/checkpoints/0000000500/ \
    --media_type image \
    --media_path apps/plm/dummy_datasets/Radiology_mini/images/ROCOv2_2023_test_000022.png \
    --question 'You are an expert radiographer. Describe accurately what you see in this image.'

# Generation:
# CT scan of the abdomen demonstrating a large liver metastasis (yellow arrow) in segment VII.
```

Comparing the two, we see the finetuned model provide concise descriptions following the style of the training set. Note that we use the same prompt as training since the dataset is small and the model has likely overfit to it. For robust training, include the new data in a large data mix (e.g., our provided [SFT blend](../configs/stage_3/plm_8b.yaml)).


### Wrap up
From here, the model is trained and ready for evaluation. The [generation script](../generate.py) can be modified to directly evaluate the model on the radiology image captioning task (test set) using captioning metrics (e.g., CIDEr). Alternately, if trained with a larger SFT blend, it can be used for domain-specific QA (e.g., [VQA-Radiology](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)).