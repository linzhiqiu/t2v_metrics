from typing import List, Union
from PIL import Image

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

logger = logging.get_logger(__name__)


class TarsierProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {},
        "images_kwargs": {},
    }


class TarsierProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_token", "patch_size", "merge_size", "temporal_patch_size", "max_seq_len"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
                self,
                image_processor=None,
                tokenizer=None,
                chat_template=None,
                image_token="<image>",
                patch_size=None,
                merge_size=1,
                temporal_patch_size=1,
                max_seq_len=8192,
                **kwargs,
            ) -> None:
        
        self.image_token = image_token
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.max_seq_len = max_seq_len
        self.max_pixels_per_sample = 128 * 384 * 384

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
            self, 
            messages,
            image_processing_config=None,
            is_training=True,
        ) -> torch.Tensor:

        output_kwargs = self._merge_kwargs(
            TarsierProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        # 【图片处理】
        pixel_values, image_grid_thw = [], []
        num_images = 0
        for msg in messages:
            for content in msg['content']:
                if content['type'] == 'image':
                    num_images += self.temporal_patch_size
                elif content['type'] == 'video':
                    num_images += len(content['video'])
        if num_images > 0 and self.max_pixels_per_sample // num_images < image_processing_config['max_pixels']:
            image_processing_config['max_pixels'] = self.max_pixels_per_sample // num_images
            image_processing_config['min_pixels'] = min(image_processing_config['min_pixels'], image_processing_config['max_pixels'])

        for msg in messages:
            for content in msg['content']:
                if content['type'] == 'image':
                    content['image'] = self.preprocess_image(content['image'], image_processing_config)
                    content['image'] = self.image_processor(images = content['image'], **output_kwargs["images_kwargs"], return_tensors="pt")
                    content['num_vision_tokens'] = self.get_num_vision_tokens(content)
                    pixel_values.append(content['image']['pixel_values'])
                    if 'image_grid_thw' in content['image']:
                        image_grid_thw.extend(content['image']['image_grid_thw'])
                elif content['type'] == 'video':
                    content['video'] = self.preprocess_image(content['video'], image_processing_config)
                    if isinstance(self.image_processor, Qwen2VLImageProcessor):
                        content['video'] = self.image_processor(images = None, videos = content['video'], **output_kwargs["images_kwargs"], return_tensors="pt")
                        pixel_values.append(content['video']['pixel_values_videos'])
                    else:
                        content['video'] = self.image_processor(images = content['video'], **output_kwargs["images_kwargs"], return_tensors="pt")
                        pixel_values.append(content['video']['pixel_values'])

                    if 'video_grid_thw' in content['video']:
                        image_grid_thw.extend(content['video']['video_grid_thw'])
                    content['num_vision_tokens'] = self.get_num_vision_tokens(content)
        
        #【文本处理】
        add_generation_prompt = (not is_training and messages[-1]['role'] != 'assistant')
        strip_final_eos = (not is_training and messages[-1]['role'] == 'assistant')
        text_inputs = self.tokenizer.apply_chat_template(
            messages,
            chat_template = self.chat_template,
            tokenize=True,
            tokenizer_kwargs = output_kwargs["text_kwargs"],
            return_assistant_tokens_mask=True, 
            return_dict=True,
            add_generation_prompt=add_generation_prompt,    
            strip_final_eos=strip_final_eos,    
        )
        labels = [-100 if j == 0 else i for i, j in zip(text_inputs['input_ids'], text_inputs['assistant_masks'])]
        labels = labels[:self.max_seq_len]
        input_ids = text_inputs['input_ids'][:self.max_seq_len]

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id in text_inputs['input_ids'][self.max_seq_len:]:
            raise ValueError(f'Too long sequence! {len(text_inputs["input_ids"])}')
        
        outputs = {
            'input_ids': input_ids,
            'labels': labels,
            'num_images': num_images,
        }
        if len(pixel_values) > 0:
            outputs['pixel_values'] = torch.cat(pixel_values, dim=0)
        if len(image_grid_thw) > 0:
            outputs['image_grid_thw'] = torch.stack(image_grid_thw)
        return outputs
    

    def preprocess_image(self, pil_img: Union[Image.Image, List[Image.Image]], image_processing_config):
        if image_processing_config is None:
            return pil_img
        images = pil_img
        if isinstance(pil_img, Image.Image):
            images = [images]
        if image_processing_config['do_crop']:
            images = [self.centralcrop(img, rate=[4, 3]) for img in images]
        if image_processing_config['do_padding']:
            images = [self.expand2square(
                img,
                # tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
                tuple(int(x * 255) for x in [0, 0, 0])
            ) for img in images]
        if image_processing_config['do_resize']:
            images = [self.resize2square(img) for img in images]
        if image_processing_config.get('max_pixels'):
            images = [self.resize2pixels(
                img, 
                int(image_processing_config['max_pixels']), 
                int(image_processing_config['min_pixels'])
            ) for img in images]
        if isinstance(pil_img, Image.Image):
            images = images[0]
        return images

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def resize2square(self, pil_img: Image.Image):
        width, height = pil_img.size
        pil_img = pil_img.resize((max(width, height), max(width, height)))
        return pil_img
    
    def centralcrop(self, pil_img: Image.Image, rate=[4, 3]):
        width, height = pil_img.size
        size = (width, height)
        min_len = min(size)
        longer_side = 0 if width >= height else 1
        center = (width/2, height/2)
        box = [0, 0, size[0], size[1]]

        # if longer_side == 0:
        #     box[0] = max(0, center[0] - 1/2*min_len/rate[1]*rate[0])
        #     box[2] = min(width, center[0] + 1/2*min_len/rate[1]*rate[0])
        # else:
        #     box[1] = max(0, center[1] - 1/2*min_len/rate[1]*rate[0])
        #     box[3] = min(height, center[1] + 1/2*min_len/rate[1]*rate[0])
        box[longer_side] = max(0, center[longer_side] - 1/2*min_len/rate[1]*rate[0])
        box[2 + longer_side] = min(size[longer_side], center[longer_side] + 1/2*min_len/rate[1]*rate[0])

        # box = (width/2-min_len/2, height/2-min_len/2, width/2+min_len/2, height/2+min_len/2)
        pil_img = pil_img.crop(box)
        return pil_img
    
    def resize2pixels(self, pil_img: Image.Image, max_pixels=None, min_pixels=None):
        width, height = pil_img.size
        new_height, new_width = smart_resize(height, width, factor=1, max_pixels=max_pixels, min_pixels=min_pixels)
        pil_img = pil_img.resize((new_width, new_height))
        return pil_img

    def get_num_vision_tokens(self, content):
        if isinstance(self.image_processor, Qwen2VLImageProcessor):
            merge_length = self.image_processor.merge_size**2
            if content['type'] == 'image':
                num_image_tokens = content['image']['image_grid_thw'].prod() // merge_length
            else:
                num_image_tokens = content['video']['video_grid_thw'].prod() // merge_length
            return num_image_tokens
        else:
            # 其他模型：image tokens (-> 2x2 compressed) -> add image_newline and image_new
            k = 'image'if content['type'] == 'image' else 'video'
            pixel_values = content[k]['pixel_values'][0]
            n_frames = len(content[k]['pixel_values'])
                
            height, width = get_image_size(to_numpy_array(pixel_values))
            num_image_tokens = (height // (self.patch_size * self.merge_size)) * (width // (self.patch_size * self.merge_size) + 1) + 1
            return num_image_tokens * n_frames
      
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
