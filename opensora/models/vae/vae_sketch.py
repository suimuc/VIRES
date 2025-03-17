from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint
from diffusers.models.autoencoders.vae import Encoder as Encoder_2D
from .vae_temporal import Encoder as Encoder_3D
from .utils import DiagonalGaussianDistribution
from .vae_temporal import CausalConv3d



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def make_zero_conv_2d(channels):
    return zero_module(nn.Conv2d(channels, channels, 1, padding=0))

def make_zero_conv_3d(channels):
    return zero_module(nn.Conv3d(channels, channels, 1, padding=0))

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


def exists(v):
    return v is not None



@MODELS.register_module()
class VAE_With_Sketch(nn.Module):
    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        micro_batch_size = None,
        micro_frame_size = None,

    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        # self.time_padding = self.time_downsample_factor - 1
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels
        self.micro_batch_size = micro_batch_size
        self.micro_frame_size = micro_frame_size
        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        layers_per_block = 2
        self.encoder_2d = Encoder_2D(
            in_channels=3,
            out_channels=4,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D"],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=layers_per_block,
            act_fn="silu",
            norm_num_groups=32,
            double_z=True,
        )
        self.quant_conv_2d = torch.nn.Conv2d(2 * 4, 2 * 4, 1)
        self.encoder_3d = Encoder_3D(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        
        
        self.zero_conv_2d = nn.ModuleList()
        for channel in [128, 256, 512, 512]:
            for j in range(layers_per_block):
                self.zero_conv_2d.append(make_zero_conv_2d(channel))
        self.zero_conv_2d.append(make_zero_conv_2d(512))
        self.zero_conv_2d.append(make_zero_conv_2d(512))
        
        self.zero_conv_3d = nn.ModuleList()
        for ch_mult in channel_multipliers:
            for j in range(num_res_blocks):
                self.zero_conv_3d.append(make_zero_conv_3d(ch_mult * filters))
        for j in range(num_res_blocks):        
            self.zero_conv_3d.append(make_zero_conv_3d(channel_multipliers[-1] * filters))
        
        self.quant_conv_3d = CausalConv3d(2 * 4, 2 * 4, 1)

    def encode_detail(self, x):
        # x_z = self.spatial_vae.encode(x)
        features_2d = []
        features_3d = []
        B = x.shape[0]
        x_z = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            # x_z = self.spatial_vae.module.encode(x_z).latent_dist.sample().mul_(self.spatial_vae.scaling_factor)
            # h = self.spatial_vae.module.encoder(x_z)
            sample = x_z
            sample = self.encoder_2d.conv_in(sample)
            # print("2d encoder1")
            # down
            # print(sample.shape)
            # for down_block in self.encoder_2d.down_blocks:
                # sample = down_block(sample)
            zero_idx = 0
            for i in range(len(self.encoder_2d.down_blocks)):
                for resnet in self.encoder_2d.down_blocks[i].resnets:
                    sample = resnet(sample, temb=None)
                    features_2d.append(self.zero_conv_2d[zero_idx](sample))
                    zero_idx += 1

                if self.encoder_2d.down_blocks[i].downsamplers is not None:
                    for downsampler in self.encoder_2d.down_blocks[i].downsamplers:
                        sample = downsampler(sample)

                # print(sample.shape)
            # print("2d encoder2")

            # middle
            # sample = self.spatial_vae.module.encoder.mid_block(sample)
            sample = self.encoder_2d.mid_block.resnets[0](sample, None)
            features_2d.append(self.zero_conv_2d[zero_idx](sample))
            zero_idx+=1
            for attn, resnet in zip(self.encoder_2d.mid_block.attentions, self.encoder_2d.mid_block.resnets[1:]):
                if attn is not None:
                    sample = attn(sample)
                sample = resnet(sample, None)
                features_2d.append(self.zero_conv_2d[zero_idx](sample))
                zero_idx+=1

            # post-process
            sample = self.encoder_2d.conv_norm_out(sample)
            sample = self.encoder_2d.conv_act(sample)
            h = self.encoder_2d.conv_out(sample)

            moments = self.quant_conv_2d(h)
            posterior = DiagonalGaussianDistribution(moments)
            x_z = posterior.sample().mul_(0.18215)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x_z.shape[0], bs):
                x_bs = x_z[i: i + bs]
                zero_idx = 0
                # x_bs = self.spatial_vae.module.encode(x_bs).latent_dist.sample().mul_(self.spatial_vae.scaling_factor)
                # h = self.spatial_vae.module.encoder(x_bs)
                sample = x_bs
                sample = self.encoder_2d.conv_in(sample)
                # print("2d encoder1")
                # print(sample.shape)
                # for down_block in self.spatial_vae.module.encoder.down_blocks:
                    # sample = down_block(sample)
                # import pdb; pdb.set_trace()
                features_2d_bs = []
                for i in range(len(self.encoder_2d.down_blocks)):
                    for resnet in self.encoder_2d.down_blocks[i].resnets:
                        sample = resnet(sample, temb=None)
                        features_2d_bs.append(self.zero_conv_2d[zero_idx](sample))
                        zero_idx += 1
                        

                    if self.encoder_2d.down_blocks[i].downsamplers is not None:
                        for downsampler in self.encoder_2d.down_blocks[i].downsamplers:
                            sample = downsampler(sample)
                # down
                
                    # print(sample.shape)
                # print("2d encoder2")

                # middle
                # import pdb; pdb.set_trace()
                # sample = self.spatial_vae.module.encoder.mid_block(sample)
                sample = self.encoder_2d.mid_block.resnets[0](sample, None)
                features_2d_bs.append(self.zero_conv_2d[zero_idx](sample))
                zero_idx += 1
                for attn, resnet in zip(self.encoder_2d.mid_block.attentions, self.encoder_2d.mid_block.resnets[1:]):
                    if attn is not None:
                        sample = attn(sample)
                    sample = resnet(sample, None)
                    features_2d_bs.append(self.zero_conv_2d[zero_idx](sample))
                    zero_idx += 1
                features_2d.append(features_2d_bs)

                # post-process
                sample = self.encoder_2d.conv_norm_out(sample)
                sample = self.encoder_2d.conv_act(sample)
                h = self.encoder_2d.conv_out(sample)
                moments = self.quant_conv_2d(h)
                posterior = DiagonalGaussianDistribution(moments)
                x_bs = posterior.sample().mul_(0.18215)
                x_out.append(x_bs)
            x_z = torch.cat(x_out, dim=0)
        x_z = rearrange(x_z, "(B T) C H W -> B C T H W", B=B)




        if self.micro_frame_size is None:
            # posterior = self.temporal_vae.encode(x_z)
            time_padding = (
                0
                if (x_z.shape[2] % self.time_downsample_factor == 0)
                else self.time_downsample_factor - x_z.shape[2] % self.time_downsample_factor
            )
            x_1 = pad_at_dim(x_z, (time_padding, 0), dim=2)
            # encoded_feature = self.temporal_vae.encoder(x_1)
            x_1 = self.encoder_3d.conv_in(x_1)
            zero_idx = 0
            for i in range(self.encoder_3d.num_blocks):
                for j in range(self.encoder_3d.num_res_blocks):
                    x_1 = self.encoder_3d.block_res_blocks[i][j](x_1)
                    features_3d.append(self.zero_conv_3d[zero_idx](x_1))
                    zero_idx += 1
                if i < self.encoder_3d.num_blocks - 1:
                    x_1 = self.encoder_3d.conv_blocks[i](x_1)
            for i in range(self.encoder_3d.num_res_blocks):
                x_1 = self.encoder_3d.res_blocks[i](x_1)
                features_3d.append(self.zero_conv_3d[zero_idx](x_1))
                zero_idx += 1

            x_1 = self.encoder_3d.norm1(x_1)
            x_1 = self.encoder_3d.activate(x_1)
            encoded_feature = self.encoder_3d.conv2(x_1)
            moments = self.quant_conv_3d(encoded_feature).to(x_1.dtype)
            posterior = DiagonalGaussianDistribution(moments)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i: i + self.micro_frame_size]
                # posterior = self.temporal_vae.encode(x_z_bs)
                time_padding = (
                    0
                    if (x_z_bs.shape[2] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - x_z_bs.shape[2] % self.time_downsample_factor
                )
                x_1 = pad_at_dim(x_z_bs, (time_padding, 0), dim=2)
                # encoded_feature = self.temporal_vae.encoder(x_1)
                x_1 = self.encoder_3d.conv_in(x_1)
                # print("3d encoder1")
                # print(x_1.shape)
                zero_idx = 0
                features_3d_bs = []
                

                for i in range(self.encoder_3d.num_blocks):
                    for j in range(self.encoder_3d.num_res_blocks):
                        x_1 = self.encoder_3d.block_res_blocks[i][j](x_1)
                        features_3d_bs.append(self.zero_conv_3d[zero_idx](x_1))
                        zero_idx += 1
                    if i < self.encoder_3d.num_blocks - 1:
                        x_1 = self.encoder_3d.conv_blocks[i](x_1)
                        
                    # print(x_1.shape)
                # print("3d encoder2")
                for i in range(self.encoder_3d.num_res_blocks):
                    x_1 = self.encoder_3d.res_blocks[i](x_1)
                    features_3d_bs.append(self.zero_conv_3d[zero_idx](x_1))
                    zero_idx += 1
                    
                features_3d.append(features_3d_bs)

                x_1 = self.encoder_3d.norm1(x_1)
                x_1 = self.encoder_3d.activate(x_1)
                encoded_feature = self.encoder_3d.conv2(x_1)
                moments = self.quant_conv_3d(encoded_feature).to(x_1.dtype)
                posterior = DiagonalGaussianDistribution(moments)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        return features_2d, features_3d
        # if self.cal_loss:
        #     return z, posterior, x_z
        # else:
        #     return (z - self.shift) / self.scale


@MODELS.register_module("VAE_Sketch")
def VAE_Sketch(from_pretrained=None, **kwargs):
    model = VAE_With_Sketch(
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
