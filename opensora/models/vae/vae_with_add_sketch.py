import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel
import numpy as np
from opensora.registry import MODELS, build_module
from opensora.utils.ckpt_utils import load_checkpoint
from typing import Optional, Tuple, Union


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        device = self.parameters.device
        sample_device = "cpu" if device.type == "mps" else device
        sample = torch.randn(self.mean.shape, generator=generator, device=sample_device)
        # make sure sample is on the same device as the parameters and has same dtype
        sample = sample.to(device=device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean




class VideoAutoencoder_With_Sketch_PipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoder_With_Sketch_Pipeline"

    def __init__(
            self,
            vae_2d=None,
            vae_temporal=None,
            vae_sketch = None,
            from_pretrained=None,
            freeze_vae_2d=False,
            cal_loss=False,
            micro_frame_size=None,
            shift=0.0,
            scale=1.0,
            **kwargs,
    ):
        self.vae_2d = vae_2d
        self.vae_temporal = vae_temporal
        self.vae_sketch = vae_sketch
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.cal_loss = cal_loss
        self.micro_frame_size = micro_frame_size
        self.shift = shift
        self.scale = scale
        super().__init__(**kwargs)


class VideoAutoencoder_With_Sketch_Pipeline(PreTrainedModel):
    config_class = VideoAutoencoder_With_Sketch_PipelineConfig

    def __init__(self, config: VideoAutoencoder_With_Sketch_PipelineConfig):
        super().__init__(config=config)
        self.spatial_vae = build_module(config.vae_2d, MODELS)
        self.temporal_vae = build_module(config.vae_temporal, MODELS)
        self.sketch_vae = build_module(config.vae_sketch, MODELS)
        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)
        
        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if ('sketch_vae' in name):
                param.requires_grad = True
        

    def encode_detail(self, x):
        # x_z = self.spatial_vae.encode(x)
        B = x.shape[0]
        x_z = rearrange(x, "B C T H W -> (B T) C H W")

        if self.spatial_vae.micro_batch_size is None:
            # x_z = self.spatial_vae.module.encode(x_z).latent_dist.sample().mul_(self.spatial_vae.scaling_factor)
            # h = self.spatial_vae.module.encoder(x_z)
            sample = x_z
            sample = self.spatial_vae.module.encoder.conv_in(sample)
            # print("2d encoder1")
            # down
            # print(sample.shape)
            # for down_block in self.encoder_2d.down_blocks:
                # sample = down_block(sample)
                
            for i in range(len(self.spatial_vae.module.encoder.down_blocks)):
                for resnet in self.spatial_vae.module.encoder.down_blocks[i].resnets:
                    sample = resnet(sample, temb=None)

                if self.spatial_vae.module.encoder.down_blocks[i].downsamplers is not None:
                    for downsampler in self.spatial_vae.module.encoder.down_blocks[i].downsamplers:
                        sample = downsampler(sample)

                # print(sample.shape)
            # print("2d encoder2")

            # middle
            # sample = self.spatial_vae.module.encoder.mid_block(sample)
            sample = self.spatial_vae.module.encoder.mid_block.resnets[0](sample, None)
            for attn, resnet in zip(self.spatial_vae.module.encoder.mid_block.attentions, self.spatial_vae.module.encoder.mid_block.resnets[1:]):
                if attn is not None:
                    sample = attn(sample)
                sample = resnet(sample, None)

            # post-process
            sample = self.spatial_vae.module.encoder.conv_norm_out(sample)
            sample = self.spatial_vae.module.encoder.conv_act(sample)
            h = self.spatial_vae.module.encoder.conv_out(sample)

            moments = self.spatial_vae.module.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            x_z = posterior.sample().mul_(self.spatial_vae.scaling_factor)
            # torch.cuda.empty_cache()
        else:
            # NOTE: cannot be used for training
            bs = self.spatial_vae.micro_batch_size
            x_out = []
            for i in range(0, x_z.shape[0], bs):
                x_bs = x_z[i: i + bs]
                # x_bs = self.spatial_vae.module.encode(x_bs).latent_dist.sample().mul_(self.spatial_vae.scaling_factor)
                # h = self.spatial_vae.module.encoder(x_bs)
                sample = x_bs
                sample = self.spatial_vae.module.encoder.conv_in(sample)
                # print("2d encoder1")
                # print(sample.shape)
                # for down_block in self.spatial_vae.module.encoder.down_blocks:
                    # sample = down_block(sample)
                # import pdb; pdb.set_trace()
                for i in range(len(self.spatial_vae.module.encoder.down_blocks)):
                    for resnet in self.spatial_vae.module.encoder.down_blocks[i].resnets:
                        sample = resnet(sample, temb=None)

                    if self.spatial_vae.module.encoder.down_blocks[i].downsamplers is not None:
                        for downsampler in self.spatial_vae.module.encoder.down_blocks[i].downsamplers:
                            sample = downsampler(sample)
                # down
                
                    # print(sample.shape)
                # print("2d encoder2")

                # middle
                # import pdb; pdb.set_trace()
                # sample = self.spatial_vae.module.encoder.mid_block(sample)
                sample = self.spatial_vae.module.encoder.mid_block.resnets[0](sample, None)
                for attn, resnet in zip(self.spatial_vae.module.encoder.mid_block.attentions, self.spatial_vae.module.encoder.mid_block.resnets[1:]):
                    if attn is not None:
                        sample = attn(sample)
                    sample = resnet(sample, None)

                # post-process
                sample = self.spatial_vae.module.encoder.conv_norm_out(sample)
                sample = self.spatial_vae.module.encoder.conv_act(sample)
                h = self.spatial_vae.module.encoder.conv_out(sample)
                moments = self.spatial_vae.module.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                x_bs = posterior.sample().mul_(self.spatial_vae.scaling_factor)
                x_out.append(x_bs)
                # torch.cuda.empty_cache()
            x_z = torch.cat(x_out, dim=0)
        x_z = rearrange(x_z, "(B T) C H W -> B C T H W", B=B)




        if self.micro_frame_size is None:
            # posterior = self.temporal_vae.encode(x_z)
            time_padding = (
                0
                if (x_z.shape[2] % self.temporal_vae.time_downsample_factor == 0)
                else self.temporal_vae.time_downsample_factor - x_z.shape[2] % self.temporal_vae.time_downsample_factor
            )
            x_1 = pad_at_dim(x_z, (time_padding, 0), dim=2)
            # encoded_feature = self.temporal_vae.encoder(x_1)
            x_1 = self.temporal_vae.encoder.conv_in(x_1)

            for i in range(self.temporal_vae.encoder.num_blocks):
                for j in range(self.temporal_vae.encoder.num_res_blocks):
                    x_1 = self.temporal_vae.encoder.block_res_blocks[i][j](x_1)
                if i < self.temporal_vae.encoder.num_blocks - 1:
                    x_1 = self.temporal_vae.encoder.conv_blocks[i](x_1)
            for i in range(self.temporal_vae.encoder.num_res_blocks):
                x_1 = self.temporal_vae.encoder.res_blocks[i](x_1)

            x_1 = self.temporal_vae.encoder.norm1(x_1)
            x_1 = self.temporal_vae.encoder.activate(x_1)
            encoded_feature = self.temporal_vae.encoder.conv2(x_1)
            moments = self.temporal_vae.quant_conv(encoded_feature).to(x_1.dtype)
            posterior = DiagonalGaussianDistribution(moments)
            z = posterior.sample()
            # torch.cuda.empty_cache()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i: i + self.micro_frame_size]
                # posterior = self.temporal_vae.encode(x_z_bs)
                time_padding = (
                    0
                    if (x_z_bs.shape[2] % self.temporal_vae.time_downsample_factor == 0)
                    else self.temporal_vae.time_downsample_factor - x_z_bs.shape[2] % self.temporal_vae.time_downsample_factor
                )
                x_1 = pad_at_dim(x_z_bs, (time_padding, 0), dim=2)
                # encoded_feature = self.temporal_vae.encoder(x_1)
                x_1 = self.temporal_vae.encoder.conv_in(x_1)
                # print("3d encoder1")
                # print(x_1.shape)

                for i in range(self.temporal_vae.encoder.num_blocks):
                    for j in range(self.temporal_vae.encoder.num_res_blocks):
                        x_1 = self.temporal_vae.encoder.block_res_blocks[i][j](x_1)
                    if i < self.temporal_vae.encoder.num_blocks - 1:
                        x_1 = self.temporal_vae.encoder.conv_blocks[i](x_1)
                        
                    # print(x_1.shape)
                # print("3d encoder2")
                for i in range(self.temporal_vae.encoder.num_res_blocks):
                    x_1 = self.temporal_vae.encoder.res_blocks[i](x_1)

                x_1 = self.temporal_vae.encoder.norm1(x_1)
                x_1 = self.temporal_vae.encoder.activate(x_1)
                encoded_feature = self.temporal_vae.encoder.conv2(x_1)
                moments = self.temporal_vae.quant_conv(encoded_feature).to(x_1.dtype)
                posterior = DiagonalGaussianDistribution(moments)
                z_list.append(posterior.sample())
                # torch.cuda.empty_cache()
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            return (z - self.shift) / self.scale




    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i: i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i: i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def decode_detail(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            # x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            time_padding = (
                0
                if (num_frames % self.temporal_vae.time_downsample_factor == 0)
                else self.temporal_vae.time_downsample_factor - num_frames % self.temporal_vae.time_downsample_factor
            )
            z = self.temporal_vae.post_quant_conv(z)

            # x_z = self.temporal_vae.decoder(z)
            x_z = self.temporal_vae.decoder.conv1(z)
            for i in range(self.temporal_vae.decoder.num_res_blocks):
                x_z = self.temporal_vae.decoder.res_blocks[i](x_z)
            for i in reversed(range(self.temporal_vae.decoder.num_blocks)):
                for j in range(self.temporal_vae.decoder.num_res_blocks):
                    x_z = self.temporal_vae.decoder.block_res_blocks[i][j](x_z)
                if i > 0:
                    t_stride = 2 if self.temporal_vae.decoder.temporal_downsample[i - 1] else 1
                    x_z = self.temporal_vae.decoder.conv_blocks[i - 1](x_z)
                    x_z = rearrange(
                        x_z,
                        "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                        ts=t_stride,
                        hs=self.temporal_vae.decoder.s_stride,
                        ws=self.temporal_vae.decoder.s_stride,
                    )
                # torch.cuda.empty_cache()

            x_z = self.temporal_vae.decoder.norm1(x_z)
            x_z = self.temporal_vae.decoder.activate(x_z)
            x_z = self.temporal_vae.decoder.conv_out(x_z)

            x_z = x_z[:, :, time_padding:]
            # return x

            # x = self.spatial_vae.decode(x_z)
            B = x_z.shape[0]
            x = rearrange(x_z, "B C T H W -> (B T) C H W")
            if self.spatial_vae.micro_batch_size is None:
                # x = self.spatial_vae.module.decode(x / 0.18215).sample
                x = x / 0.18215
                x = self.spatial_vae.module.post_quant_conv(x)
                # x = self.spatial_vae.module.decoder(x)
                x = self.spatial_vae.module.decoder.conv_in(x)
                x = self.spatial_vae.module.decoder.mid_block(x)
                # for up_block in self.spatial_vae.module.decoder.up_blocks:
                    # x = up_block(x)
                for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                    for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                        x = resnet(x, temb=None)

                    if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                        for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                            x = upsampler(x)
                            
                            
                x = self.spatial_vae.module.decoder.conv_norm_out(x)
                x = self.spatial_vae.module.decoder.conv_act(x)
                x = self.spatial_vae.module.decoder.conv_out(x)
                # torch.cuda.empty_cache()

            else:
                # NOTE: cannot be used for training
                bs = self.spatial_vae.micro_batch_size
                x_out = []
                for i in range(0, x.shape[0], bs):
                    x_bs = x[i: i + bs]
                    # x_bs = self.spatial_vae.module.decode(x_bs / 0.18215).sample
                    x_bs = x_bs / 0.18215
                    x_bs = self.spatial_vae.module.post_quant_conv(x_bs)
                    # x_bs = self.spatial_vae.module.decoder(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_in(x_bs)
                    x_bs = self.spatial_vae.module.decoder.mid_block(x_bs)
                    # print(x_bs.shape)
                    # for up_block in self.spatial_vae.module.decoder.up_blocks:
                        # x_bs = up_block(x_bs)
                    for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                        for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                            x_bs = resnet(x_bs, temb=None)

                        if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                            for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                                x_bs = upsampler(x_bs)
                        # print(x_bs.shape)
                    x_bs = self.spatial_vae.module.decoder.conv_norm_out(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_act(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_out(x_bs)
                    x_out.append(x_bs)
                    # torch.cuda.empty_cache()
                x = torch.cat(x_out, dim=0)
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)

        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i: i + self.micro_z_frame_size]

                # x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))

                num_frames_bs = min(self.micro_frame_size, num_frames)

                time_padding = (
                    0
                    if (num_frames_bs % self.temporal_vae.time_downsample_factor == 0)
                    else self.temporal_vae.time_downsample_factor - num_frames_bs % self.temporal_vae.time_downsample_factor
                )
                z_bs = self.temporal_vae.post_quant_conv(z_bs)

                # x_z = self.temporal_vae.decoder(z)
                x_z_bs = self.temporal_vae.decoder.conv1(z_bs)
                for i in range(self.temporal_vae.decoder.num_res_blocks):
                    x_z_bs = self.temporal_vae.decoder.res_blocks[i](x_z_bs)
                # print("3d decoder1")
                # print(x_z_bs.shape)
                for i in reversed(range(self.temporal_vae.decoder.num_blocks)):
                    for j in range(self.temporal_vae.decoder.num_res_blocks):
                        x_z_bs = self.temporal_vae.decoder.block_res_blocks[i][j](x_z_bs)
                    if i > 0:
                        t_stride = 2 if self.temporal_vae.decoder.temporal_downsample[i - 1] else 1
                        x_z_bs = self.temporal_vae.decoder.conv_blocks[i - 1](x_z_bs)
                        x_z_bs = rearrange(
                            x_z_bs,
                            "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                            ts=t_stride,
                            hs=self.temporal_vae.decoder.s_stride,
                            ws=self.temporal_vae.decoder.s_stride,
                        )
                    # print(x_z_bs.shape)
                # print("3d decoder2")

                x_z_bs = self.temporal_vae.decoder.norm1(x_z_bs)
                x_z_bs = self.temporal_vae.decoder.activate(x_z_bs)
                x_z_bs = self.temporal_vae.decoder.conv_out(x_z_bs)
                # print(x_z_bs.shape)

                x_z_bs = x_z_bs[:, :, time_padding:]

                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)

            # x = self.spatial_vae.decode(x_z)

            B = x_z.shape[0]
            x = rearrange(x_z, "B C T H W -> (B T) C H W")
            # import pdb; pdb.set_trace()
            if self.spatial_vae.micro_batch_size is None:
                # x = self.spatial_vae.module.decode(x / 0.18215).sample
                x = x / 0.18215
                x = self.spatial_vae.module.post_quant_conv(x)
                # x = self.spatial_vae.module.decoder(x)
                x = self.spatial_vae.module.decoder.conv_in(x)
                x = self.spatial_vae.module.decoder.mid_block(x)
                # print("2d decoder1")
                # print(x.sahpe)
                # for up_block in self.spatial_vae.module.decoder.up_blocks:
                    # x = up_block(x)
                    # print(x.sahpe)
                for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                    for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                        x = resnet(x, temb=None)

                    if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                        for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                            x = upsampler(x)
                # print("2d decoder2")
                x = self.spatial_vae.module.decoder.conv_norm_out(x)
                x = self.spatial_vae.module.decoder.conv_act(x)
                x = self.spatial_vae.module.decoder.conv_out(x)

            else:
                # NOTE: cannot be used for training
                bs = self.spatial_vae.micro_batch_size
                x_out = []
                for i in range(0, x.shape[0], bs):
                    x_bs = x[i: i + bs]
                    # x_bs = self.spatial_vae.module.decode(x_bs / 0.18215).sample
                    x_bs = x_bs / 0.18215
                    x_bs = self.spatial_vae.module.post_quant_conv(x_bs)
                    # x_bs = self.spatial_vae.module.decoder(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_in(x_bs)
                    x_bs = self.spatial_vae.module.decoder.mid_block(x_bs)
                    # print("2d decoder1")
                    # print(x_bs.shape)
                    # for up_block in self.spatial_vae.module.decoder.up_blocks:
                        # x_bs = up_block(x_bs)
                        # print(x_bs.shape)
                    for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                        for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                            x_bs = resnet(x_bs, temb=None)

                        if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                            for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                                x_bs = upsampler(x_bs)
                    # print("2d decoder 2")
                    x_bs = self.spatial_vae.module.decoder.conv_norm_out(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_act(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_out(x_bs)
                    x_out.append(x_bs)
                x = torch.cat(x_out, dim=0)
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)

        if self.cal_loss:
            return x, x_z
        else:
            return x

        
    def decode_detail_with_sketch(self, z, num_frames=None, features_2d = None, features_3d = None):
        # import pdb; pdb.set_trace()
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            # x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            time_padding = (
                0
                if (num_frames % self.temporal_vae.time_downsample_factor == 0)
                else self.temporal_vae.time_downsample_factor - num_frames % self.temporal_vae.time_downsample_factor
            )
            z = self.temporal_vae.post_quant_conv(z)

            # x_z = self.temporal_vae.decoder(z)
            x_z = self.temporal_vae.decoder.conv1(z)
            for i in range(self.temporal_vae.decoder.num_res_blocks):
                x_z = self.temporal_vae.decoder.res_blocks[i](x_z)
                x_z += features_3d.pop()
            for i in reversed(range(self.temporal_vae.decoder.num_blocks)):
                for j in range(self.temporal_vae.decoder.num_res_blocks):
                    x_z = self.temporal_vae.decoder.block_res_blocks[i][j](x_z)
                    x_z += features_3d.pop()
                if i > 0:
                    t_stride = 2 if self.temporal_vae.decoder.temporal_downsample[i - 1] else 1
                    x_z = self.temporal_vae.decoder.conv_blocks[i - 1](x_z)
                    x_z = rearrange(
                        x_z,
                        "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                        ts=t_stride,
                        hs=self.temporal_vae.decoder.s_stride,
                        ws=self.temporal_vae.decoder.s_stride,
                    )

            x_z = self.temporal_vae.decoder.norm1(x_z)
            x_z = self.temporal_vae.decoder.activate(x_z)
            x_z = self.temporal_vae.decoder.conv_out(x_z)

            x_z = x_z[:, :, time_padding:]
            # torch.cuda.empty_cache()
            # return x

            # x = self.spatial_vae.decode(x_z)
            B = x_z.shape[0]
            x = rearrange(x_z, "B C T H W -> (B T) C H W")
            if self.spatial_vae.micro_batch_size is None:
                # x = self.spatial_vae.module.decode(x / 0.18215).sample
                x = x / 0.18215
                x = self.spatial_vae.module.post_quant_conv(x)
                # x = self.spatial_vae.module.decoder(x)
                x = self.spatial_vae.module.decoder.conv_in(x)
                # x = self.spatial_vae.module.decoder.mid_block(x)
                x = self.spatial_vae.module.decoder.mid_block.resnets[0](x, None)
                x += features_2d.pop()
                for attn, resnet in zip(self.spatial_vae.module.decoder.mid_block.attentions, self.spatial_vae.module.decoder.mid_block.resnets[1:]):
                    if attn is not None:
                        x = attn(x)
                    x = resnet(x, None)
                    x += features_2d.pop()
                # for up_block in self.spatial_vae.module.decoder.up_blocks:
                    # x = up_block(x)
                for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                    cnt_mmm = 0
                    for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                        x = resnet(x, temb=None)
                        if (cnt_mmm < 2):
                            x += features_2d.pop()
                            cnt_mmm += 1

                    if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                        for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                            x = upsampler(x)
                            
                            
                x = self.spatial_vae.module.decoder.conv_norm_out(x)
                x = self.spatial_vae.module.decoder.conv_act(x)
                x = self.spatial_vae.module.decoder.conv_out(x)
                # torch.cuda.empty_cache()

            else:
                # NOTE: cannot be used for training
                bs = self.spatial_vae.micro_batch_size
                x_out = []
                for idx in range(0, x.shape[0], bs):
                    x_bs = x[idx: idx + bs]
                    # x_bs = self.spatial_vae.module.decode(x_bs / 0.18215).sample
                    x_bs = x_bs / 0.18215
                    x_bs = self.spatial_vae.module.post_quant_conv(x_bs)
                    # x_bs = self.spatial_vae.module.decoder(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_in(x_bs)
                    # x_bs = self.spatial_vae.module.decoder.mid_block(x_bs)
                    x_bs = self.spatial_vae.module.decoder.mid_block.resnets[0](x_bs, None)
                    x_bs += features_2d[idx // bs].pop()
                    for attn, resnet in zip(self.spatial_vae.module.decoder.mid_block.attentions, self.spatial_vae.module.decoder.mid_block.resnets[1:]):
                        if attn is not None:
                            x_bs = attn(x_bs)
                        x_bs = resnet(x_bs, None)
                        x_bs += features_2d[idx // bs].pop()
                    # print(x_bs.shape)
                    # for up_block in self.spatial_vae.module.decoder.up_blocks:
                        # x_bs = up_block(x_bs)
                    for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                        cnt_mmm = 0
                        for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                            x_bs = resnet(x_bs, temb=None)
                            if (cnt_mmm < 2):
                                x_bs += features_2d[idx // bs].pop()
                                cnt_mmm += 1

                        if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                            for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                                x_bs = upsampler(x_bs)
                        # print(x_bs.shape)
                    x_bs = self.spatial_vae.module.decoder.conv_norm_out(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_act(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_out(x_bs)
                    x_out.append(x_bs)
                    # torch.cuda.empty_cache()
                x = torch.cat(x_out, dim=0)
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)

        else:
            x_z_list = []
            for idx in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, idx: idx + self.micro_z_frame_size]

                # x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))

                num_frames_bs = min(self.micro_frame_size, num_frames)

                time_padding = (
                    0
                    if (num_frames_bs % self.temporal_vae.time_downsample_factor == 0)
                    else self.temporal_vae.time_downsample_factor - num_frames_bs % self.temporal_vae.time_downsample_factor
                )
                z_bs = self.temporal_vae.post_quant_conv(z_bs)

                # x_z = self.temporal_vae.decoder(z)
                x_z_bs = self.temporal_vae.decoder.conv1(z_bs)
                for i in range(self.temporal_vae.decoder.num_res_blocks):
                    x_z_bs = self.temporal_vae.decoder.res_blocks[i](x_z_bs)
                    x_z_bs += features_3d[idx // self.micro_z_frame_size].pop()
                # print("3d decoder1")
                # print(x_z_bs.shape)
                for i in reversed(range(self.temporal_vae.decoder.num_blocks)):
                    for j in range(self.temporal_vae.decoder.num_res_blocks):
                        x_z_bs = self.temporal_vae.decoder.block_res_blocks[i][j](x_z_bs)
                        x_z_bs += features_3d[idx // self.micro_z_frame_size].pop()
                    if i > 0:
                        t_stride = 2 if self.temporal_vae.decoder.temporal_downsample[i - 1] else 1
                        x_z_bs = self.temporal_vae.decoder.conv_blocks[i - 1](x_z_bs)
                        x_z_bs = rearrange(
                            x_z_bs,
                            "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                            ts=t_stride,
                            hs=self.temporal_vae.decoder.s_stride,
                            ws=self.temporal_vae.decoder.s_stride,
                        )
                    # print(x_z_bs.shape)
                # print("3d decoder2")

                x_z_bs = self.temporal_vae.decoder.norm1(x_z_bs)
                x_z_bs = self.temporal_vae.decoder.activate(x_z_bs)
                x_z_bs = self.temporal_vae.decoder.conv_out(x_z_bs)
                # print(x_z_bs.shape)

                x_z_bs = x_z_bs[:, :, time_padding:]

                x_z_list.append(x_z_bs)
                # torch.cuda.empty_cache()
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)

            # x = self.spatial_vae.decode(x_z)

            B = x_z.shape[0]
            x = rearrange(x_z, "B C T H W -> (B T) C H W")
            # import pdb; pdb.set_trace()
            if self.spatial_vae.micro_batch_size is None:
                # x = self.spatial_vae.module.decode(x / 0.18215).sample
                x = x / 0.18215
                x = self.spatial_vae.module.post_quant_conv(x)
                # x = self.spatial_vae.module.decoder(x)
                x = self.spatial_vae.module.decoder.conv_in(x)
                # x = self.spatial_vae.module.decoder.mid_block(x)
                x = self.spatial_vae.module.decoder.mid_block.resnets[0](x, None)
                x += features_2d.pop()
                for attn, resnet in zip(self.spatial_vae.module.decoder.mid_block.attentions, self.spatial_vae.module.decoder.mid_block.resnets[1:]):
                    if attn is not None:
                        x = attn(x)
                    x = resnet(x, None)
                    x += features_2d.pop()
                # print("2d decoder1")
                # print(x.sahpe)
                # for up_block in self.spatial_vae.module.decoder.up_blocks:
                    # x = up_block(x)
                    # print(x.sahpe)
                for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                    cnt_mmm = 0
                    for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                        x = resnet(x, temb=None)
                        if (cnt_mmm < 2):
                            x += features_2d.pop()
                            cnt_mmm += 1

                    if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                        for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                            x = upsampler(x)
                # print("2d decoder2")
                x = self.spatial_vae.module.decoder.conv_norm_out(x)
                x = self.spatial_vae.module.decoder.conv_act(x)
                x = self.spatial_vae.module.decoder.conv_out(x)
                # torch.cuda.empty_cache()

            else:
                # NOTE: cannot be used for training
                bs = self.spatial_vae.micro_batch_size
                x_out = []
                for idx in range(0, x.shape[0], bs):
                    x_bs = x[idx: idx + bs]
                    # x_bs = self.spatial_vae.module.decode(x_bs / 0.18215).sample
                    x_bs = x_bs / 0.18215
                    x_bs = self.spatial_vae.module.post_quant_conv(x_bs)
                    # x_bs = self.spatial_vae.module.decoder(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_in(x_bs)
                    # x_bs = self.spatial_vae.module.decoder.mid_block(x_bs)
                    x_bs = self.spatial_vae.module.decoder.mid_block.resnets[0](x_bs, None)
                    x_bs += features_2d[idx // bs].pop()
                    for attn, resnet in zip(self.spatial_vae.module.decoder.mid_block.attentions, self.spatial_vae.module.decoder.mid_block.resnets[1:]):
                        if attn is not None:
                            x_bs = attn(x_bs)
                        x_bs = resnet(x_bs, None)
                        x_bs += features_2d[idx // bs].pop()
                    # print("2d decoder1")
                    # print(x_bs.shape)
                    # for up_block in self.spatial_vae.module.decoder.up_blocks:
                        # x_bs = up_block(x_bs)
                        # print(x_bs.shape)
                    # import pdb; pdb.set_trace()
                    for i in range(len(self.spatial_vae.module.decoder.up_blocks)):
                        cnt_mmm = 0
                        for resnet in self.spatial_vae.module.decoder.up_blocks[i].resnets:
                            x_bs = resnet(x_bs, temb=None)
                            if (cnt_mmm < 2):
                                # print(f"x_bs:{x_bs.shape}, hint:{features_2d[idx // bs].pop().shape}")
                                x_bs += features_2d[idx // bs].pop()
                                cnt_mmm += 1

                        if self.spatial_vae.module.decoder.up_blocks[i].upsamplers is not None:
                            for upsampler in self.spatial_vae.module.decoder.up_blocks[i].upsamplers:
                                x_bs = upsampler(x_bs)
                    # print("2d decoder 2")
                    x_bs = self.spatial_vae.module.decoder.conv_norm_out(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_act(x_bs)
                    x_bs = self.spatial_vae.module.decoder.conv_out(x_bs)
                    x_out.append(x_bs)
                    # torch.cuda.empty_cache()
                x = torch.cat(x_out, dim=0)
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        # print(features_2d)
        # print(features_3d)

        if self.cal_loss:
            return x, x_z
        else:
            return x
        
        
    def forward(self, x, hint):
        assert self.cal_loss, "This method is only available when cal_loss is True"
        # import pdb; pdb.set_trace()
        # with torch.no_grad():
        z, posterior, x_z = self.encode_detail(x)
        features_2d, features_3d = self.sketch_vae.encode_detail(hint)
        # with torch.no_grad():
        x_rec, x_z_rec = self.decode_detail_with_sketch(z, num_frames=x_z.shape[2], features_2d = features_2d, features_3d = features_3d)
        return x_rec, x_z_rec, z, posterior, x_z

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
def OpenSoraVAE_With_Sketch_V1_2(
        micro_batch_size=4,
        micro_frame_size=17,
        from_pretrained=None,
        local_files_only=False,
        freeze_vae_2d=False,
        cal_loss=False,
        sdxlvae_path = None,
        force_huggingface=False,
):
    vae_2d = dict(
        type="VideoAutoencoderKL",
        from_pretrained = sdxlvae_path,
        micro_batch_size=micro_batch_size,
        local_files_only=local_files_only,
    )
    vae_temporal = dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    )
    vae_sketch = dict(
        type="VAE_Sketch",
        micro_batch_size = micro_batch_size,
        micro_frame_size = micro_frame_size,
        from_pretrained=None,
    )
    shift = (-0.10, 0.34, 0.27, 0.98)
    scale = (3.85, 2.32, 2.33, 3.06)
    kwargs = dict(
        vae_2d=vae_2d,
        vae_temporal=vae_temporal,
        vae_sketch = vae_sketch,
        freeze_vae_2d=freeze_vae_2d,
        cal_loss=cal_loss,
        micro_frame_size=micro_frame_size,
        shift=shift,
        scale=scale,
    )

    if force_huggingface or (from_pretrained is not None and not os.path.exists(from_pretrained)):
        model = VideoAutoencoder_With_Sketch_Pipeline.from_pretrained(from_pretrained, **kwargs)
    else:
        config = VideoAutoencoder_With_Sketch_PipelineConfig(**kwargs)
        model = VideoAutoencoder_With_Sketch_Pipeline(config)

        if from_pretrained:
            load_checkpoint(model, from_pretrained)
    return model