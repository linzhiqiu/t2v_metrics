import json
import logging
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F
from clip_benchmark.metrics.zeroshot_retrieval import (dataloader_with_indices,
                                                       recall_at_k)
from tqdm import tqdm


def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    amp=True,
    recall_k_list=[1],
    args=None,
    retrieval_template=None,
):
    """
    Evaluate the model on the given dataset

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

    recall_k_list: list of int
        recall@k k's to use

    retrieval_template:
        dict of retrieval templates for each class. Retrieval templates should contain lists of image/text indexes. The model will performed retrieval accross the examples in each list.

    Returns
    -------

    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress

    for batch_images, batch_texts, inds in tqdm(dataloader):
        # move the batch to the device
        batch_images = image_to_device(
            batch_images,
            device,
            torch.float32,
            mean=args.image_mean,
            std=args.image_std,
        )

        # tokenize all texts in the batch
        batch_texts_tok = tokenizer(
            [text for i, texts in enumerate(batch_texts) for text in texts]
        ).to(device)

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    assert images_emb.shape[0] == texts_emb.shape[0]

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    metrics = {}
    multiclass_image_retrieval = []
    multiclass_text_retrieval = []
    for c in retrieval_template.keys():

        image_retrieval = []
        text_retrieval = []
        for indexes in retrieval_template[c]:
            retrieved = scores[np.ix_(indexes, indexes)]
            positive_pairs = torch.zeros_like(retrieved, dtype=bool)
            positive_pairs[
                torch.arange(len(retrieved)), torch.arange(len(retrieved))
            ] = True

            image_retrieval.append(recall_at_k(retrieved, positive_pairs, k=1))
            text_retrieval.append(recall_at_k(retrieved.T, positive_pairs, k=1))

        average_image_retrieval = torch.cat(image_retrieval).float().mean().item()
        average_text_retrieval = torch.cat(text_retrieval).float().mean().item()

        metrics[f"image_retrieval_recall@1_{c}"] = average_image_retrieval
        metrics[f"text_retrieval_recall@1_{c}"] = average_text_retrieval

        multiclass_image_retrieval.append(average_image_retrieval)
        multiclass_text_retrieval.append(average_text_retrieval)

    metrics["image_retrieval_recall@1_multiclass"] = (
        torch.tensor(multiclass_image_retrieval).float().mean().item()
    )
    metrics["text_retrieval_recall@1_multiclass"] = (
        torch.tensor(multiclass_text_retrieval).float().mean().item()
    )

    return metrics
