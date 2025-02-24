from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
import math

import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto import AutoModel, AutoModelForCausalLM, CONFIG_MAPPING
from transformers.generation import GenerationMixin

from transformers import LlamaForCausalLM, Qwen2ForCausalLM
# from models.modeling_qwen2 import Qwen2ForCausalLM
from models.modeling_qwen2_vl_fast import Qwen2VLForCausalLM
from models.utils import _pad_input, _unpad_input

logger = logging.get_logger(__name__)


class LlavaConfig(PretrainedConfig):

    model_type = "llava"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_newline_idx=32002,
        image_new_idx=32003,
        projection_head="MLP",
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.image_newline_idx = image_newline_idx
        self.image_new_idx = image_new_idx
        self.projection_head = projection_head

        self.vision_config = vision_config

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            if 'auto_map' in vision_config:
                repo_id, class_ref = vision_config['auto_map']['AutoConfig'].split("--")
                config_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
                self.vision_config = config_class(**vision_config)
            elif vision_config["model_type"] in CONFIG_MAPPING:
                self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)                
            else:
                raise ValueError(f'vision_config["model_type"] = {vision_config["model_type"]} not supported!')
        
        self.text_config = text_config

        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            if 'auto_map' in text_config:
                repo_id, class_ref = text_config['auto_map']['AutoConfig'].split("--")
                config_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
                self.text_config = config_class(**text_config)
            elif text_config["model_type"] in CONFIG_MAPPING:
                self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            else:
                raise ValueError(f'text_config["model_type"] = {text_config["model_type"]} not supported!')
            

        super().__init__(**kwargs)



@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class LlavaCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    position_ids: Optional[torch.LongTensor] = None
    
def add_split_tokens(image_features, image_newline_embed, image_new_embed):
    num_images, num_image_patches, embed_dim = image_features.shape
    num_height_patches, num_width_patches = int(math.sqrt(num_image_patches)), int(math.sqrt(num_image_patches))

    # add image_newline
    image_features = image_features.view(num_images, num_height_patches, num_width_patches, embed_dim)
    image_features = torch.cat([
        image_features,
        image_newline_embed.expand((num_images, num_height_patches, 1, embed_dim))
    ], dim=2)
    num_image_patches += num_height_patches
    image_features = image_features.view(num_images, num_image_patches, embed_dim)

    # add image_new
    image_features = torch.cat([
        image_features,
        image_new_embed.expand((num_images, 1, embed_dim))
    ], dim = 1)

    return image_features


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

        image_newline_idx = torch.tensor([config.image_newline_idx], dtype=torch.long)
        image_new_idx = torch.tensor([config.image_new_idx], dtype=torch.long)
        self.register_buffer('image_newline_idx', image_newline_idx, persistent=False)
        self.register_buffer('image_new_idx', image_new_idx, persistent=False)
        

    def forward(self, image_features, input_embeddings):

        selected_image_feature = image_features[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        hidden_states = self.linear_1(selected_image_feature)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        image_newline_embed = input_embeddings(self.image_newline_idx).squeeze()
        image_new_embed = input_embeddings(self.image_new_idx).squeeze()
        hidden_states = add_split_tokens(hidden_states, image_newline_embed, image_new_embed)
        return hidden_states

class PixelShuffleMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config

        self.downsample_ratio = 0.5
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        image_newline_idx = torch.tensor([config.image_newline_idx], dtype=torch.long)
        image_new_idx = torch.tensor([config.image_new_idx], dtype=torch.long)
        self.register_buffer('image_newline_idx', image_newline_idx, persistent=False)
        self.register_buffer('image_new_idx', image_new_idx, persistent=False)
    
    def forward(self, image_features, input_embeddings):
        selected_image_feature = image_features[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )
        
        image_features = self.pixel_shuffle(selected_image_feature)
        hidden_states = self.mlp(image_features)
        
        image_newline_embed = input_embeddings(self.image_newline_idx).squeeze()
        image_new_embed = input_embeddings(self.image_new_idx).squeeze()
        hidden_states = add_split_tokens(hidden_states, image_newline_embed, image_new_embed)

        return hidden_states

    def pixel_shuffle(self, x, scale_factor=0.5):
        if scale_factor == 1:
            return x
        n, wh, c = x.shape
        h, w = int(math.sqrt(wh)), int(math.sqrt(wh))
        x = x.view(n, h, w, c)

        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], -1, x.shape[-1])
        return x
        

LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

class TarsierPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "llm"
    supports_gradient_checkpointing = True # TODO: support latest gc
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True # TODO: support different cache
    _supports_static_cache = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    @property
    def _no_split_modules(self):
        return self.language_model._no_split_modules + self.vision_tower._no_split_modules 


class TarsierForConditionalGeneration(TarsierPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config, trust_remote_code=True)
        if config.text_config.model_type == 'qwen2':
            self.language_model = Qwen2ForCausalLM(config.text_config)
        elif config.text_config.model_type == 'qwen2_vl':
            self.language_model = Qwen2VLForCausalLM(config.text_config)
        elif config.text_config.model_type == 'llama':
            self.language_model = LlamaForCausalLM(config.text_config)
        else:
            raise ValueError(f'{config.text_config.model_type} not supported!')

        if config.projection_head == 'Pixel_Shuffle':
            self.multi_modal_projector = PixelShuffleMultiModalProjector(config)
        elif config.projection_head == 'MLP':
            self.multi_modal_projector = LlavaMultiModalProjector(config)
        elif config.projection_head == 'auto_map':
            repo_id, class_ref = config.auto_map['ProjectionLayer'].split("--")
            model_class = get_class_from_dynamic_module(class_ref, repo_id)
            self.multi_modal_projector = model_class(config)
        elif config.projection_head is None:
            self.multi_modal_projector = lambda x, *args, **kwargs: x

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        num_images: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_rmpad: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
       
        
        if input_ids is None:
            raise ValueError("You must specify input_ids")
        
        bsz, max_seq_len = input_ids.shape[0], input_ids.shape[1]

        if max_seq_len > 1:
            special_image_mask = input_ids == self.config.image_token_index
            print(f'[{input_ids.device}] num_images: {num_images.tolist()} num_image_tokens: {special_image_mask.sum(-1).tolist()}', flush=True)

        if position_ids is None:
            if 'Qwen2VLForCausalLM' in self.language_model.__class__.__name__:
                position_ids = self.language_model.get_rope_index(input_ids, image_grid_thw, attention_mask) # [bsz, seqlen, 3]
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1 #  # [bsz, seqlen]
                position_ids.masked_fill_(attention_mask == 0, 1)
        
        
        if use_rmpad:
            input_ids, input_ids_indices, cu_seqlens, _ = _unpad_input(input_ids, attention_mask) # [bsz, seqlen] -> [1, seqlen]
            position_ids, _, _, _ = _unpad_input(position_ids, attention_mask)
            input_ids, position_ids = input_ids.unsqueeze(0), position_ids.unsqueeze(0)
        else:
            input_ids_indices, cu_seqlens = None, None

        inputs_embeds = self.get_input_embeddings()(input_ids) # [1, seqlen, dim]
        
        image_features = None
        if pixel_values is not None: # training / first step in generation
            if 'Qwen2VLForCausalLM' in self.language_model.__class__.__name__:
                pixel_values = pixel_values.type(self.vision_tower.get_dtype())
                image_features = self.vision_tower(pixel_values, image_grid_thw)
            else:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                image_features = self.multi_modal_projector(
                    image_outputs.hidden_states,
                    self.get_input_embeddings(),
                )

            special_image_mask = (input_ids == self.config.image_token_index).to(inputs_embeds.device)
            if special_image_mask.sum() > 0:
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_features
                )
            else:
                inputs_embeds = image_features.sum(dim=(0,1)) * 0. + inputs_embeds

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_rmpad=use_rmpad,
            cu_seqlens=cu_seqlens,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if use_rmpad:
                labels = labels.view(-1)[input_ids_indices.long()]
                shift_labels = torch.cat((labels[1:], labels.new_ones((1))*-100))
                shift_labels.requires_grad = False
                lbl_seq_lens = (cu_seqlens[1:]-1).long()
                shift_labels[lbl_seq_lens] = -100
                loss = loss_fct(logits.squeeze(0), shift_labels)
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
        elif use_rmpad: # 训练的时候，就不 unpad logits 了，节省显存。
            logits = _pad_input(logits.squeeze(0), input_ids_indices, bsz, max_seq_len)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            position_ids=position_ids,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            input_ids = input_ids[:, past_length:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if kwargs.get('num_images') is not None:
            model_inputs['num_images'] = kwargs['num_images']

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
        else:
            model_inputs['position_ids'] = position_ids[:, -1, ...].unsqueeze(1).to(device=input_ids.device) + 1
        return model_inputs


    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "position_ids", None) is not None:
            model_kwargs["position_ids"] = outputs.position_ids

        return model_kwargs
