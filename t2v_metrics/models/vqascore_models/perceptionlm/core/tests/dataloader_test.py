import os
from unittest import TestCase

import torch

from apps.plm.dataset_conf import dataset_config as DATASET_CONFIGS
from apps.plm.tokenizer import PLMTokenizer
from core.data.dataloader import DataloadArgs, get_dataloader


# TOKENIZER_PATH=facebook/Perception-LM-1B/tokenizer.model python -m unittest core/tests/dataloader_test.py
class DataloaderTest(TestCase):
    def setUp(self):
        self.seq_len = 8196
        self.patch_size = 14
        self.pooling_ratio = 2
        self.max_num_tiles = 9
        self.image_res = 448
        self.mllm_tokenizer = PLMTokenizer(
            os.environ["TOKENIZER_PATH"],
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            pooling_ratio=self.pooling_ratio,
        )

    def test_jsonl_image_text_dataloader(self):
        dataloader_args = DataloadArgs(
            datamix="dummy_image:1",
            num_workers=1,
            vision_input_type="thumb+tile",
            image_res=self.image_res,
            max_num_tiles=self.max_num_tiles,
            batch_size=1,
        )
        dataloader = get_dataloader(
            dataloader_args,
            dp_rank=0,
            dp_world_size=1,
            dataset_configs=DATASET_CONFIGS,
            tokenizer=self.mllm_tokenizer,
        )
        batch_iterator = iter(dataloader)
        expected_num_image_tokens = (
            self.image_res // self.patch_size // self.pooling_ratio
        ) ** 2 * (self.max_num_tiles + 1)
        print(f"expected_num_image_tokens: {expected_num_image_tokens}")
        for i in range(3):
            mllm_batch = next(batch_iterator)

            image_token_mask = mllm_batch.x == self.mllm_tokenizer.image_token_id
            num_image_tokens = image_token_mask.sum(dim=1)
            self.assertTrue((mllm_batch.image_pos_index[~image_token_mask] == -1).all())

            for i in range(mllm_batch.x.shape[0]):
                print(f"num_image_tokens in example {i}", num_image_tokens[i])
                self.assertEqual(
                    num_image_tokens[i],
                    expected_num_image_tokens,
                )
                cur_x_is_image = image_token_mask[i]
                cur_image_pos_index = mllm_batch.image_pos_index[i]
                self.assertTrue(
                    torch.equal(
                        cur_image_pos_index[cur_x_is_image],
                        torch.arange(num_image_tokens[i]),
                    )
                )
