import torch
import os

import ast
import json
import os.path as osp
import re
import shutil
import sys
import tempfile
from copy import deepcopy
from importlib import import_module

import yaml

from ...video_utils import get_video_details, load_frames_from_video
import numpy as np

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .models.umt import UMT
from .shared_utils import setup_model

from easydict import EasyDict

from .models.criterions import get_sim

BASE_KEY = "_base_"
BASE_CONFIG = {}


class Config(object):
    """config"""

    @classmethod
    def pretty_text(cls, cfg: dict, indent=2) -> str:
        """format dict to a string

        Args:
            cfg (EasyDict): the params.

        Returns: The string to display.

        """
        msg = "{\n"
        for i, (k, v) in enumerate(cfg.items()):
            if isinstance(v, dict):
                v = cls.pretty_text(v, indent + 4)
            spaces = " " * indent
            msg += spaces + "{}: {}".format(k, v)
            if i == len(cfg) - 1:
                msg += " }"
            else:
                msg += "\n"
        return msg

    @classmethod
    def dump(cls, cfg, savepath=None):
        """dump cfg to `json` file.

        Args:
            cfg (dict): The dict to dump.
            savepath (str): The filepath to save the dumped dict.

        Returns: TODO

        """
        if savepath is None:
            savepath = osp.join(cfg.WORKSPACE, "config.json")
        json.dump(cfg, open(savepath, "w"), indent=2)

    @classmethod
    def get_config(
        cls, model_name: str, model_path: str, pretrained_path: str
    ) -> EasyDict:
        cfg = EasyDict(BASE_CONFIG)
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
        target_file = os.path.join(current_dir, f"{model_name}.py")  # Construct the new file path
        cfg_from_file = cls.from_file(target_file)
        cfg = merge_a_into_b(cfg_from_file, cfg)
        cfg = cls.merge_list(cfg, ["model.vision_encoder.pretrained", pretrained_path])
        cfg = cls.merge_list(cfg, ["pretrained_path", model_path])
        cfg = eval_dict_leaf(cfg)

        # update some keys to make them show at the last
        for k in BASE_CONFIG:
            cfg[k] = cfg.pop(k)
        return cfg

    @classmethod
    def from_file(cls, filepath: str) -> EasyDict:
        """Build config from file. Supported filetypes: `.py`,`.yaml`,`.json`.

        Args:
            filepath (str): The config file path.

        Returns: TODO

        """
        filepath = osp.abspath(osp.expanduser(filepath))
        if not osp.isfile(filepath):
            raise IOError(f"File does not exist: {filepath}")
        if filepath.endswith(".py"):
            with tempfile.TemporaryDirectory() as temp_config_dir:

                shutil.copytree(
                    osp.dirname(filepath), osp.join(temp_config_dir, "tmp_config")
                )
                sys.path.insert(0, temp_config_dir)
                mod = import_module(
                    "tmp_config." + osp.splitext(osp.basename(filepath))[0]
                )
                # mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                }
                for k in list(sys.modules.keys()):
                    if "tmp_config" in k:
                        del sys.modules[k]
        elif filepath.endswith((".yml", ".yaml")):
            cfg_dict = yaml.load(open(filepath, "r"), Loader=yaml.Loader)
        elif filepath.endswith(".json"):
            cfg_dict = json.load(open(filepath, "r"))
        else:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        cfg_text = filepath + "\n"
        with open(filepath, "r") as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:  # load configs in `BASE_KEY`
            cfg_dir = osp.dirname(filepath)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            for f in base_filename:
                _cfg_dict = Config.from_file(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            cfg_dict = merge_a_into_b(cfg_dict, base_cfg_dict)

        return EasyDict(cfg_dict)

    @classmethod
    def merge_list(cls, cfg, opts: list):
        """merge commandline opts.

        Args:
            cfg: (dict): The config to be merged.
            opts (list): The list to merge. Format: [key1, name1, key2, name2,...].
                The keys can be nested. For example, ["a.b", v] will be considered
                as `dict(a=dict(b=v))`.

        Returns: dict.

        """
        assert len(opts) % 2 == 0, f"length of opts must be even. Got: {opts}"
        for i in range(0, len(opts), 2):
            full_k, v = opts[i], opts[i + 1]
            keys = full_k.split(".")
            sub_d = cfg
            for i, k in enumerate(keys):
                if not hasattr(sub_d, k):
                    raise ValueError(
                        f"The key {k} not exist in the config. Full key:{full_k}"
                    )
                if i != len(keys) - 1:
                    sub_d = sub_d[k]
                else:
                    sub_d[k] = v
        return cfg


def merge_a_into_b(a, b, inplace=False):
    """The values in a will override values in b.

    Args:
        a (dict): source dict.
        b (dict): target dict.

    Returns: dict. recursively merge dict a into dict b.

    """
    if not inplace:
        b = deepcopy(b)
    for key in a:
        if key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                b[key] = merge_a_into_b(a[key], b[key], inplace=True)
            else:
                b[key] = a[key]
        else:
            b[key] = a[key]
    return b


def eval_dict_leaf(d, orig_dict=None):
    """eval values of dict leaf.

    Args:
        d (dict): The dict to eval.

    Returns: dict.

    """
    if orig_dict is None:
        orig_dict = d
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v, orig_dict)
        else:
            eval_dict_leaf(v, orig_dict)
    return d


def eval_string(string, d):
    """automatically evaluate string to corresponding types.

    For example:
        not a string  -> return the original input
        '0'  -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0,1,2]
        'eval(1+2)' -> 3
        'eval(range(5))' -> [0,1,2,3,4]
        '${a}' -> d.a



    Args:
        string (str): The value to evaluate.
        d (dict): The

    Returns: the corresponding type

    """
    if not isinstance(string, str):
        return string
    # if len(string) > 1 and string[0] == "[" and string[-1] == "]":
    #     return eval(string)
    if string[0:5] == "eval(":
        return eval(string[5:-1])

    s0 = string
    s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
    if s1 != s0:
        while s1 != s0:
            s0 = s1
            s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
        return eval(s1)

    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    text_feats = []
    text_atts = []

    text = texts
    text_input = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_txt_l,
        return_tensors="pt",
    ).to(device)

    text_feat = model.encode_text(text_input)[0]
    text_feats.append(text_feat)
    text_atts.append(text_input.attention_mask)

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_feats, text_atts


def extract_vision_feats(image_paths, transforms, model, device, num_frames=4):
    image_feats_all = []
    pooled_image_feats_all = []
    image = []
    for data_path in image_paths:
        total_frames, original_fps, video_duration = get_video_details(data_path)
        uniform_sampling = min(num_frames, total_frames)
        all_indices = np.linspace(0, total_frames - 1, uniform_sampling, dtype=int)
        frames = load_frames_from_video(data_path, all_indices, "decord", True)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        frames = transforms(frames)
        image.append(frames)
    image = torch.stack(image, dim=0)
    image = image.to(device, non_blocking=True)
    image_feat, pooled_image_feat = model.encode_vision(image, test=True)
    image_feat = image_feat.unsqueeze(1)  # (bsz, 1, #frm*L, d)
    image_feats_all.append(image_feat)
    pooled_image_feats_all.append(pooled_image_feat)
    image_feats_all = torch.cat(image_feats_all, dim=0)

    pooled_image_feats_all = torch.cat(pooled_image_feats_all, dim=0)
    return image_feats_all, pooled_image_feats_all


def create_transforms(image_res=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_res, image_res),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform


@torch.no_grad()
def evaluation(
    texts,
    image_paths,
    transforms,
    model,
    tokenizer,
    device,
    num_frames=4,
    max_txt_l=32,
):
    text_feats, text_atts = extract_text_feats(
        texts, max_txt_l, tokenizer, model, device
    )  # (bsz, Lt, d), (bsz, Lt)

    image_feats, pooled_image_feats = extract_vision_feats(
        image_paths, transforms, model, device, num_frames=num_frames
    )  # (bsz, 1, #frm*Li, d) or (bsz, #frm, Li, d), (bsz, #frm, d)
    _pooled_image_feats = pooled_image_feats.to(device, non_blocking=True)
    i2t_scores, _ = get_sim(
        model.vision_proj(_pooled_image_feats), model.text_proj(text_feats[:, 0])
    )

    text_encoder = model.get_text_encoder()
    encoder_output = image_feats[:, 0].to(device, non_blocking=True)
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)

    output = text_encoder(
        encoder_embeds=text_feats,
        attention_mask=text_atts,
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=encoder_att,
        return_dict=True,
        mode="fusion",
    )
    itm_outputs = output.last_hidden_state[:, 0]
    itm_embeds = torch.cat([itm_outputs], dim=0)
    itm_scores = model.itm_head(itm_embeds)[:, 1]

    return (
        itm_scores,
        i2t_scores.diagonal(),
    )


def download_umt(model_name, pretrained_ckpt_name, cache_dir, device="cuda"):
    repo_id = f"zhiqiulin/{model_name}"
    filename = f"{model_name}.pth"
    model_path = os.path.join(cache_dir, model_name) + ".pth"
    if not os.path.exists(model_path):
        hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        os.system(f"wget -O {model_path} {hf_url}")

    pretrained_repo_id = f"zhiqiulin/{pretrained_ckpt_name}"
    pretrained_filename = f"{pretrained_ckpt_name}.pth"
    pretrained_ckpt_path = os.path.join(cache_dir, f"{pretrained_ckpt_name}.pth")
    if not os.path.exists(pretrained_ckpt_path):
        hf_url = f"https://huggingface.co/{pretrained_repo_id}/resolve/main/{pretrained_filename}"
        os.system(f"wget -O {pretrained_ckpt_path} {hf_url}")

    config = Config.get_config(
        model_name=model_name,
        model_path=model_path,
        pretrained_path=pretrained_ckpt_path,
    )
    assert config.evaluate

    (
        model,
        tokenizer,
    ) = setup_model(
        config,
        model_cls=UMT,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=False,
        device=device
    )
    model = model.eval()
    tokenizer = tokenizer
    config = config
    return (
        model_path,
        pretrained_ckpt_path,
        model,
        tokenizer,
        config,
        create_transforms(224),
    )
