import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


class CausalConv3d(nn.Module):
    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            pad_mode="constant",
            strides=None,  # allow custom stride
            **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x, hint=None):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv(x)
        return x


class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
            self, embedding_dim: int, out_dim: int, num_groups: int, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class CausalResnetBlockCondNorm3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int = 512,
            groups: int = 36,
            groups_out: Optional[int] = None,
            eps: float = 1e-6,
            output_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

        self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class CausalResnetBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int = 1152,
            groups: int = 36,
            groups_out: Optional[int] = None,
            eps: float = 1e-6,
            output_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), strides=(1, 1, 1))

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:

        # import pdb; pdb.set_trace()

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None, None]

        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor