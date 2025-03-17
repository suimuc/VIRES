import os
from pprint import pformat
import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    prepare_multi_resolution_info,
    read_video,
    read_mask,
    down_sample_mask,
)
from opensora.utils.misc import create_logger, is_distributed, is_main_process, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False

    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
                resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels + 1,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    video_path = cfg.get("input_video", None)
    if (video_path is None):
        print('Please input video_path')
        video_path = int(input())

    sketch_video_path = cfg.get("sketch_video", None)
    if (sketch_video_path is None):
        print('Please input sketch_video_path')
        sketch_video_path = input()

    mask_path = cfg.get("mask_video", None)

    input_prompt = cfg.get("prompt", None)
    if not isinstance(input_prompt, list):
        input_prompt = [input_prompt]

    try:
        video, video_fps = read_video(video_path)
        video_sketch, video_sketch_fps = read_video(sketch_video_path)
        if (mask_path is None):
            video_mask = torch.ones(1, 1, 512, 512)
            video_mask_ori = torch.ones(1, 1, 512, 512)
        else:
            video_mask, video_mask_ori, video_mask_fps = read_mask(mask_path)
    except Exception as e:
        print(f"Error happened: {e}")
        exit(0)

    cfg_guidance_scale = cfg.get("cfg_guidance_scale", None)
    num_sampling_steps = cfg.get("sampling_steps", None)
    if (cfg.start_frame):
        print(f'Please input start_frame, range from:0, {len(video) - 51}')
        start_frame = int(input())
    else:
        if (len(video) < 51):
            print(f"The video length must higher than 51 frames")
            exit(0)
        start_frame = 0

    generate_size = 512

    gt = torch.nn.functional.interpolate(video, size=(generate_size, generate_size), mode='bilinear',
                                         align_corners=False)
    gt = (gt - 127.5) / 127.5

    video_mask = torch.nn.functional.interpolate(video_mask, size=(generate_size, generate_size), mode='bilinear',
                                                 align_corners=False)
    video_mask = video_mask > 0

    video_mask_ori = torch.nn.functional.interpolate(video_mask_ori, size=(generate_size, generate_size),
                                                     mode='bilinear', align_corners=False)
    video_mask_ori = video_mask_ori > 0

    video_sketch = torch.nn.functional.interpolate(video_sketch, size=(generate_size, generate_size), mode='bilinear',
                                                   align_corners=False)

    gt = gt[start_frame:start_frame + 51]
    video_mask = video_mask[start_frame:start_frame + 51]
    video_mask_ori = video_mask_ori[start_frame:start_frame + 51]
    video_sketch = video_sketch[start_frame:start_frame + 51]

    video_sketch = video_sketch * video_mask
    vae_hint = video_sketch.clone()
    vae_hint = (vae_hint - 127.5) / 127.5
    vae_hint = vae_hint.to(device, dtype)
    video_sketch = video_sketch / 255.

    gt = gt.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_mask = video_mask.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_mask_ori = video_mask_ori.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_sketch = video_sketch.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    vae_hint = vae_hint.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)

    video_mask_ds = down_sample_mask(video_mask, micro_frame_size=vae.micro_frame_size, down_time=4, down_height=8,
                                     down_width=8).to(device, dtype)

    model_args = prepare_multi_resolution_info(
        multi_resolution, len(input_prompt), image_size, num_frames, fps, device, dtype
    )

    # add noise only to the mask area
    z_gt = vae.encode(gt)
    z = torch.randn(len(input_prompt), vae.out_channels, *latent_size, device=device, dtype=dtype)
    z = z * video_mask_ds + z_gt * (1. - video_mask_ds)

    # masks: latent frames to be denoised
    masks = torch.ones(len(input_prompt), latent_size[0]).to(device=z.device)
    torch.cuda.empty_cache()

    samples = scheduler.sample(
        model,
        text_encoder,
        z=z,
        prompts=input_prompt,
        device=device,
        additional_args=model_args,
        mask=masks,
        hint_ori=video_sketch,
        video_mask=video_mask_ds,
        num_sampling_steps=num_sampling_steps,
        guidance_scale=cfg_guidance_scale,
    )

    features_2d, features_3d = vae.sketch_vae.encode_detail(vae_hint)
    samples = vae.decode_detail_with_sketch(samples.to(dtype), num_frames=num_frames, features_2d=features_2d,
                                            features_3d=features_3d)
    samples_with_gt = samples * video_mask_ori.to(samples.dtype).to(samples.device) + (
                gt * (1.0 - video_mask_ori).to(gt.dtype).to(gt.device)).to(samples.dtype).to(samples.device)

    if is_main_process():
        save_path = os.path.join(save_dir, "output.mp4")
        save_sample(samples_with_gt[0], fps=save_fps, save_path=save_path)

    logger.info("Inference finished.")


if __name__ == "__main__":
    main()