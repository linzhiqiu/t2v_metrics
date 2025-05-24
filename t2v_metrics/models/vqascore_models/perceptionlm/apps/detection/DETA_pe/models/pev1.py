import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import broadcast_tensors, einsum, nn
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint

from .utils_d2 import (
    add_decomposed_rel_pos,
    PatchEmbed,
    window_partition,
    window_unpartition,
)


def get_abs_pos(abs_pos, has_cls_token, hw, tile=False):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        if tile == True:
            new_abs_pos = abs_pos.reshape(1, size, size, -1).tile(
                [1, h // size + 1, w // size + 1, 1]
            )[:, :h, :w, :]

            return new_abs_pos
        else:
            new_abs_pos = F.interpolate(
                abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# broadcat, as tortoise-tts was using it
def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


# rotary embedding helper functions
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = (
            torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len + 1
        )  # + 1 is hacking vev0 pt code

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        # freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
        freqs = broadcat(
            (freqs[None, :, :], freqs[:, None, :]), dim=-1
        )  # follow vev0 pt code

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print("======== shape of rope freq", self.freqs_cos.shape, "========")

    def forward(self, tt):
        return tt * self.freqs_cos + rotate_half(tt) * self.freqs_sin


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        ret = F.layer_norm(
            x.type(torch.float32),
            self.normalized_shape,
            self.weight.type(torch.float32),
            self.bias.type(torch.float32),
            self.eps,
        )
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Attention(nn.Module):
    r"""
    Implements attention based on Rope
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[bool] = None,
        vdim: Optional[bool] = None,
        rope=None,
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.rope = rope

        self.scale = self.head_dim ** (-0.5)

    def forward(self, query, attn_mask: Optional[torch.Tensor] = None):
        batch, seq, embed_dim = query.shape

        proj = torch._C._nn.linear(query, self.in_proj_weight, self.in_proj_bias)
        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q_, k_, v_ = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q_ = rearrange(q_, "b s (h d) -> b h s d", h=self.num_heads)
        k_ = rearrange(k_, "b s (h d) -> b h s d", h=self.num_heads)
        v_ = rearrange(v_, "b s (h d) -> b h s d", h=self.num_heads)

        ## rope
        q_ = self.rope(q_).type_as(v_)
        k_ = self.rope(k_).type_as(v_)

        attn = (q_ * self.scale) @ k_.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_ = attn @ v_

        x_ = rearrange(x_, "b h s d -> b s (h d)")

        return torch._C._nn.linear(x_, self.out_proj.weight, self.out_proj.bias)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        drop_path=0.0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        rope=None,
        input_size=None,
        attn_mask=None,
        init_values=0.0,
    ):
        super().__init__()

        self.attn = Attention(embed_dim=d_model, num_heads=n_head, rope=rope)
        self.ls_1 = (
            LayerScale(d_model, init_values=init_values)
            if init_values > 0.0
            else nn.Identity()
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model)),
                ]
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.ls_2 = (
            LayerScale(d_model, init_values=init_values)
            if init_values > 0.0
            else nn.Identity()
        )
        self.window_size = window_size

    def attention_nhwc(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        B, H, W, _ = x.shape
        x = x.reshape(B, H * W, -1)
        x = self.attn(x, attn_mask=self.attn_mask)
        x = x.reshape(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.ln_1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attention_nhwc(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.ls_1(x))
        x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        drop_path_rate=0.0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        img_size=1024,
        patch_size=16,
        rope_win=None,
        rope_glb=None,
        use_act_checkpoint=False,
        act_checkpoint_ratio=1.0,
        attn_mask=None,
        init_values=0.0,
        return_layer=[-1],
    ):
        super().__init__()
        self.use_act_checkpoint = use_act_checkpoint
        self.act_checkpoint_ratio = act_checkpoint_ratio

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.resblocks = nn.ModuleList()
        for i in range(depth):
            block = ResidualAttentionBlock(
                embed_dim,
                num_heads,
                attn_mask=attn_mask,
                drop_path=dpr[i],
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                rope=rope_win if i in window_block_indexes else rope_glb,
                input_size=(img_size // patch_size, img_size // patch_size),
                init_values=init_values,
            )
            self.resblocks.append(block)

        self.return_layer = return_layer

    def forward(self, x: torch.Tensor):
        x_list = []
        for idx, blk in enumerate(self.resblocks):
            if (
                self.use_act_checkpoint
                and (idx / len(self.resblocks)) <= self.act_checkpoint_ratio
            ):
                x = checkpoint(blk, x)
            else:
                x = blk(x)

            if idx in self.return_layer or idx == len(self.resblocks) - 1:
                x_list.append(x)

        return x, x_list


class PEv1_simpleFPN(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        act_checkpoint_ratio=1.0,
        pretrain_img_size=336,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        tile_posemb=False,
        init_values=0.0,
        tta_rope=False,
        return_layer=[-1],
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.conv1 = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.positional_embedding = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim)
            )
            print("positional_embedding:", self.positional_embedding.shape)
            print("positional_embedding:", self.positional_embedding.shape)
            print("positional_embedding:", self.positional_embedding.shape)

        else:
            self.positional_embedding = None

        self.tile_posemb = tile_posemb

        self.ln_pre = LayerNorm(embed_dim)

        half_head_dim = embed_dim // num_heads // 2
        hw_seq_len = img_size // patch_size

        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )
        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=hw_seq_len if intp_freq else None,
        )

        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path_rate=drop_path_rate,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
            rope_win=self.rope_win,
            rope_glb=self.rope_glb,
            img_size=img_size,
            patch_size=patch_size,
            use_act_checkpoint=use_act_checkpoint,
            act_checkpoint_ratio=act_checkpoint_ratio,
            init_values=init_values,
            return_layer=return_layer,
        )

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.positional_embedding is not None:
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.return_layer = return_layer
        # In our method, we don't use backbone feature with stride 4
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Identity()
        self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(self._init_weights)

        strides = [patch_size // 2, patch_size, patch_size * 2]
        self._out_features = ["p{}".format(int(math.log2(s))) for s in strides]
        self._out_feature_strides = {
            "p3": 8,
            "p4": 16,
            "p5": 32,
        }
        self._out_feature_channels = {
            "p3": embed_dim // 2,
            "p4": embed_dim,
            "p5": embed_dim,
        }
        self._size_divisibility = strides[-1]
        self._square_pad = img_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)

        if self.positional_embedding is not None:
            x = x + get_abs_pos(
                self.positional_embedding,
                self.pretrain_use_cls_token,
                (x.shape[1], x.shape[2]),
                self.tile_posemb,
            )
        x = self.ln_pre(x)

        x, x_list = self.transformer(x)

        xp = x.permute(0, 3, 1, 2)  # (b, h, w, c) --> (b, c, h, w)

        features = []
        ops = [self.fpn1, self.fpn2, self.fpn3]
        for i in range(len(ops)):
            features.append(ops[i](xp))
        rets = {"p{}".format(u + 3): v for (u, v) in enumerate(features)}

        return rets


def get_pev1_and_fpn_backbone(args):
    if args.lsj_img_size_max > 0:
        img_size = args.lsj_img_size_max
    else:
        img_size = args.lsj_img_size
    use_act_checkpoint = args.backbone_use_act_checkpoint
    act_checkpoint_ratio = args.backbone_act_checkpoint_ratio
    init_values = args.backbone_init_values
    tile_posemb = args.backbone_tile_posemb
    tta_rope = args.backbone_tta_rope
    multi_layer = args.backbone_multi_layer
    backbone_dp = args.backbone_dp

    if args.backbone_size == "G":
        embed_dim, depth, num_heads, mlp_ratio, dp = 1536, 50, 16, 8960 / 1536, 0.5
        pretrain_img_size, patch_size, window_size = 224, 16, 14
        window_block_indexes = (
            list(range(0, 12))
            + list(range(13, 24))
            + list(range(25, 36))
            + list(range(37, 49))
        )
        pretrain_use_cls_token = False
        if multi_layer:
            return_layer = [12, 24, 36, 49]
        else:
            return_layer = [-1]

    elif args.backbone_size == "Gwin384":
        embed_dim, depth, num_heads, mlp_ratio, dp = 1536, 50, 16, 8960 / 1536, 0.5
        pretrain_img_size, patch_size, window_size = 384, 16, 24
        window_block_indexes = (
            list(range(0, 12))
            + list(range(13, 24))
            + list(range(25, 36))
            + list(range(37, 49))
        )
        pretrain_use_cls_token = False
        if multi_layer:
            return_layer = [12, 24, 36, 49]
        else:
            return_layer = [-1]

    elif args.backbone_size == "Gwin512":
        embed_dim, depth, num_heads, mlp_ratio, dp = 1536, 50, 16, 8960 / 1536, 0.5
        pretrain_img_size, patch_size, window_size = 512, 16, 32
        window_block_indexes = (
            list(range(0, 12))
            + list(range(13, 24))
            + list(range(25, 36))
            + list(range(37, 49))
        )
        pretrain_use_cls_token = False
        if multi_layer:
            return_layer = [12, 24, 36, 49]
        else:
            return_layer = [-1]
    else:
        raise ValueError("Unsupported backbone size")

    if backbone_dp >= 0:
        dp = backbone_dp

    assert (
        depth == args.backbone_layers
    ), f"backbone depth {depth} and layers {args.backbone_layers}(from config) must be the same"

    model = PEv1_simpleFPN(
        use_act_checkpoint=use_act_checkpoint,
        act_checkpoint_ratio=act_checkpoint_ratio,
        pretrain_img_size=pretrain_img_size,
        pretrain_use_cls_token=pretrain_use_cls_token,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=window_size,
        pt_hw_seq_len=16, # Maybe a bug ?
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        tile_posemb=tile_posemb,
        init_values=init_values,
        tta_rope=tta_rope,
        return_layer=return_layer,
    )

    pretrained_backbone_path = args.backbone_path
    if pretrained_backbone_path:
        state_dict = torch.load(pretrained_backbone_path, map_location="cpu")
        load_info = model.load_state_dict(state_dict["model"], strict=False)
        print("Missing keys", load_info.missing_keys)
        print("Unexpected keys", load_info.unexpected_keys)
    else:
        print("Skip pretrained backbone loading")
    return model
