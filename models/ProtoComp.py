import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from models.Transformer_utils import *
from .point_e.models.checkpoint import checkpoint
from .point_e.models.pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .point_e.models.util import timestep_embedding
from pointnet2_ops import pointnet2_utils


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(
            device=device, dtype=dtype, heads=heads, n_ctx=n_ctx
        )
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        output_channels,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )
        self.control_resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )
        self.control_zeroFC = nn.ModuleList(
            [
                zero_module(nn.Linear(width, width, device=device, dtype=dtype))
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        ln_pre,
        scale,
        x_partial: torch.Tensor,
        pe: torch.Tensor,
        x: torch.Tensor,
        control_x: torch.Tensor,
    ):
        out = []
        h = torch.cat([(x_partial + pe), control_x], dim=1)
        h = ln_pre(h)
        for block, zeroFC in zip(self.control_resblocks, self.control_zeroFC):
            h = block(h)
            partial_text_control = h[:, : x_partial.size()[1] + 1].clone()
            partial_text_control = torch.max(partial_text_control, dim=1, keepdim=True)[
                0
            ]
            control = torch.cat(
                [partial_text_control, h[:, x_partial.size()[1] + 1 :]], dim=1
            )
            out.append(zeroFC(control))

        x = ln_pre(x)
        for num, block in enumerate(self.resblocks):
            if num >= (self.layers) // 2:
                x = block(x) + scale * out[num]
            else:
                x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
        partial_scale: float = 1.0,
        low_scale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.low_scale_factor = low_scale_factor
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.partial_scale = partial_scale
        self.time_embed = MLP(
            device=device,
            dtype=dtype,
            width=width,
            init_scale=init_scale * math.sqrt(1.0 / width),
        )
        self.control_time_embed = MLP(
            device=device,
            dtype=dtype,
            width=width,
            init_scale=init_scale * math.sqrt(1.0 / width),
        )

        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            output_channels=self.output_channels,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self,
        l_partial: torch.Tensor,
        pe: torch.Tensor,
        x_partial: torch.Tensor,
        x: torch.Tensor,
        cond_as_token: List[Tuple[torch.Tensor, bool]],
        control_cond_as_token: List[Tuple[torch.Tensor, bool]],
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))
        control_h = self.input_proj(x.permute(0, 2, 1))
        if l_partial is not None:
            l_partial = torch.max(l_partial, dim=1, keepdim=True)[0]
            l_partial = l_partial.expand(
                control_h.size()[0], control_h.size()[1], control_h.size()[2]
            )
            control_h += l_partial * self.low_scale_factor

        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]

        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        for emb, as_token in control_cond_as_token:
            if not as_token:
                control_h = control_h + emb[:, None]
        control_extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in control_cond_as_token
            if as_token
        ]
        if len(control_extra_tokens):
            control_h = torch.cat(control_extra_tokens + [control_h], dim=1)
        h = self.backbone(self.ln_pre, self.partial_scale, x_partial, pe, h, control_h)
        h = self.ln_post(h)

        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)

        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        center_num: list = [512, 128],
        k_knn: int = 16,
        partial_c: int = 512,
        use_low: bool = False,
        **kwargs,
    ):
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx + int(token_cond), **kwargs
        )
        self.n_ctx = n_ctx
        self.token_cond = token_cond

        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device, cache_dir=cache_dir
        )

        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.control_clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )

        self.cond_drop_prob = cond_drop_prob
        self.control_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), zero_module(nn.Linear(128, partial_c))
        )
        self.center_num = center_num
        self.control_grouper = DGCNN_Grouper(k=k_knn)
        self.control_f_input_proj = nn.Sequential(
            nn.Linear(self.control_grouper.num_features, partial_c),
            nn.GELU(),
            zero_module(nn.Linear(partial_c, partial_c)),
        )
        if use_low:
            self.control_fc = zero_module(nn.Linear(3, partial_c))
        else:
            self.control_fc = None

    def cached_model_kwargs(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    def forward(
        self,
        partial: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ):

        if self.control_fc is not None:
            l_partial = self.control_fc(partial.clone())
        else:
            l_partial = None
        coor, f = self.control_grouper(partial, self.center_num)

        pe = self.control_pos_embed(coor)
        x_partial = self.control_f_input_proj(f)

        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        control_t_embed = self.control_time_embed(
            timestep_embedding(t, self.backbone.width)
        )
        clip_out = self.clip(
            batch_size=len(x), images=images, texts=texts, embeddings=embeddings
        )

        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)

        control_clip_embed = self.control_clip_embed(clip_out)
        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        control_cond = [
            (control_clip_embed, self.token_cond),
            (control_t_embed, self.time_token_cond),
        ]
        h = self._forward_with_cond(l_partial, pe, x_partial, x, cond, control_cond)
        return h


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            (
                torch.tensor(channel_scales, dtype=dtype, device=device)
                if channel_scales is not None
                else None
            ),
        )
        self.register_buffer(
            "channel_biases",
            (
                torch.tensor(channel_biases, dtype=dtype, device=device)
                if channel_biases is not None
                else None
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):

        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        """
        K has to be 16
        """
        print("using group version 2")
        self.k = k

        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():

            idx = knn_point(
                k,
                coor_k.transpose(-1, -2).contiguous(),
                coor_q.transpose(-1, -2).contiguous(),
            )
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = (
                torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1)
                * num_points_k
            )
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = (
            feature.view(batch_size, k, num_points_q, num_dims)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        """
        INPUT:
            x : bs N 3
            num : list e.g.[1024, 512]
        ----------------------
        OUTPUT:

            coor bs N 3
            f    bs N C(128)
        """
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f
