import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from open_clip import image_to_device
from tqdm import tqdm


def evaluate(model, dataloader, tokenizer, device, amp=True, args=None):
    """
    Evaluate the model on the given dataset.
    The task has N instances, each instance has I images and C captions.
    For each instance, the goal is to find the correct image for each caption and the correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    This procedure is used to evaluate the models on Winoground and SugarCrepe.

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    Returns
    -------

    dict of accuracy metrics
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    image_score = []
    text_score = []
    score = []
    for batch_images, batch_texts in tqdm(dataloader):
        # assert(len(batch_images.shape) == 4)
        batch_images = image_to_device(
            batch_images,
            device,
            torch.float32,
            mean=args.image_mean,
            std=args.image_std,
        )
        # Because of the packing collate function we cannot support multi-image to caption selection
        nim = 1

        # tokenize all texts in the batch
        nt = len(batch_texts[0])
        batch_texts_tok_ = tokenizer(
            [text for i, texts in enumerate(batch_texts) for text in texts]
        ).to(device)

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(
                model.encode_image(batch_images), dim=-1
            ).unsqueeze(1)
            B, _, emb_dim = batch_images_emb.shape
            batch_texts_emb = F.normalize(
                model.encode_text(batch_texts_tok_), dim=-1
            ).view(B, nt, -1)

        gt = torch.arange(min(nim, nt)).to(device)
        for i in range(B):
            # iteratve over instances

            # compute similarities between each image and each text
            images_emb = batch_images_emb[i]
            texts_emb = batch_texts_emb[i]
            scores = images_emb @ texts_emb.t()

            # i-th image should be matched to the i-th text
            image_closest_text = scores.argmax(dim=1)[: len(gt)]
            text_closest_image = scores.argmax(dim=0)[: len(gt)]
            pred_text_is_correct = (image_closest_text == gt).all().item()
            pred_image_is_correct = (text_closest_image == gt).all().item()
            all_correct = pred_text_is_correct and pred_image_is_correct
            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            score.append(all_correct)
    metrics = {}
    metrics["image_acc"] = torch.Tensor(image_score).float().mean().item()
    metrics["text_acc"] = torch.Tensor(text_score).float().mean().item()
    metrics["acc"] = torch.Tensor(score).float().mean().item()
    return metrics
