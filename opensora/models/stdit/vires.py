import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import LMSDiscreteScheduler
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from opensora.models.stdit.blocks import STDiT3Block, STDiT3Block_with_SketchAttention
from opensora.models.stdit.resblock import CausalConv3d, CausalResnetBlock3D
from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    CaptionEmbedder,
    PatchEmbed3D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


class VIRES_Config(PretrainedConfig):
    model_type = "VIRES"

    def __init__(
            self,
            input_size=(None, None, None),
            input_sq_size=512,
            in_channels=5,
            hint_channels=3,
            patch_size=(1, 2, 2),
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            pred_sigma=True,
            drop_path=0.0,
            caption_channels=4096,
            model_max_length=300,
            qk_norm=True,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
            only_train_temporal=False,
            freeze_y_embedder=False,
            skip_y_embedder=False,
            micro_frame_size=17,
            time_downsample_factor=4,
            **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.hint_channels = hint_channels
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        self.micro_frame_size = micro_frame_size
        self.time_downsample_factor = time_downsample_factor
        super().__init__(**kwargs)


class VIRES(PreTrainedModel):
    config_class = VIRES_Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.hint_channels = config.hint_channels
        self.num_heads = config.num_heads
        self.micro_frame_size = config.micro_frame_size
        self.time_downsample_factor = config.time_downsample_factor

        self.scheduler_pww = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                  num_train_timesteps=1000, )

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]

        # hint
        # hard code
        self.hint_embedder = PatchEmbed3D(config.patch_size, config.hidden_size, config.hidden_size)
        self.input_hint_block = nn.Sequential(
            CausalConv3d(self.hint_channels, 72, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.GroupNorm(2, 72),
            nn.SiLU(),
            CausalConv3d(72, 72, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.GroupNorm(2, 72),
            nn.SiLU(),
            CausalConv3d(72, 144, kernel_size=(3, 3, 3), strides=(1, 2, 2)),
            nn.GroupNorm(2, 144),
            nn.SiLU(),
            CausalConv3d(144, 144, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.GroupNorm(2, 144),
            nn.SiLU(),
            CausalConv3d(144, 288, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.GroupNorm(2, 288),
            nn.SiLU(),
            CausalConv3d(288, 288, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.GroupNorm(2, 288),
            nn.SiLU(),
        )
        self.hint_blocks = nn.ModuleList([
            CausalResnetBlock3D(288, 288),
            CausalConv3d(288, 576, kernel_size=(3, 3, 3), strides=(2, 2, 2)),
            CausalResnetBlock3D(576, 576),
            CausalConv3d(576, config.hidden_size, kernel_size=(3, 3, 3), strides=(2, 2, 2)),
            CausalResnetBlock3D(config.hidden_size, config.hidden_size),
        ])
        self.hint_mid_convs = nn.Sequential(
            CausalConv3d(config.hidden_size, config.hidden_size, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.SiLU(),
            nn.GroupNorm(8, config.hidden_size),
            CausalConv3d(config.hidden_size, config.hidden_size, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            nn.SiLU(),
            nn.GroupNorm(8, config.hidden_size),
            CausalConv3d(config.hidden_size, config.hidden_size, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
        )

        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                ) if i == 0 else STDiT3Block_with_SketchAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for i in range(config.depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        for param in self.parameters():
            param.requires_grad = False

        # for name, param in self.named_parameters():
        #     if ('sketch_attn_1' in name or 'scale_pww' in name or 'hint' in name):
        #         param.requires_grad = True
        #     # elif ('temporal_blocks' in name):
        #     #     param.requires_grad = True
        #     elif ('attn' in name and 'cross' not in name and 'proj' in name):
        #         param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(self, x, hint, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)

        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))

        S = H * W
        base_size = round(S ** 0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)

        # === Sequential ControlNet ===
        # import pdb; pdb.set_trace()
        hint = hint.to(dtype)
        hint_list = []
        # import pdb; pdb.set_trace()
        for i in range(0, hint.shape[2], self.micro_frame_size):
            hint_bs = hint[:, :, i: i + self.micro_frame_size]
            time_padding = (
                0
                if (hint_bs.shape[2] % self.time_downsample_factor == 0)
                else self.time_downsample_factor - hint_bs.shape[2] % self.time_downsample_factor
            )
            hint_bs = pad_at_dim(hint_bs, (time_padding, 0), dim=2)
            hint_bs = self.input_hint_block(hint_bs)
            for res in self.hint_blocks:
                hint_bs = res(hint_bs, t)
            hint_bs = self.hint_mid_convs(hint_bs)
            hint_list.append(hint_bs)
        hint = torch.cat(hint_list, dim=2)

        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        # import pdb; pdb.set_trace()
        hint = self.hint_embedder(hint)
        hint = rearrange(hint, "B (T S) C -> B T S C", T=T, S=S)

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            hint = split_forward_gather_backward(hint, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        hint = rearrange(hint, "B T S C -> B (T S) C", T=T, S=S)

        t1 = timestep.to(torch.int32)
        pww_sigma = torch.index_select(self.scheduler_pww.sigmas.to(t1.device), 0, t1)
        # import pdb; pdb.set_trace()
        # === blocks ===
        for idx, (spatial_block, temporal_block) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
            if (idx == 0):
                x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            else:
                x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, hint, y_lens, x_mask, t0_mlp, T, S, pww_sigma,
                                         timestep)
            x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            if (idx == 0):
                # Standardized self-scaling
                mean_latents, std_latents = torch.mean(x, dim=(1, 2), keepdim=True), torch.std(x, dim=(1, 2),
                                                                                               keepdim=True)
                mean_control, std_control = torch.mean(hint, dim=(1, 2), keepdim=True), torch.std(hint, dim=(1, 2),
                                                                                                  keepdim=True)
                hint = (hint - mean_control) * (hint / (std_control + 1e-12)) - mean_control + mean_latents
                x = x + hint

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


@MODELS.register_module("VIRES")
def Create_VIRES(from_pretrained=None, **kwargs):
    # import pdb; pdb.set_trace()
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = VIRES.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VIRES_Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = VIRES(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
