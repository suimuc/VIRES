import gradio as gr
import os
import torch
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    prepare_multi_resolution_info,
    read_video,
    read_mask,
    down_sample_mask,
)
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.utils.misc import to_torch_dtype
from opensora.datasets import save_sample
import time
import cv2

# Load model globally (only once)
print("Loading models...")
torch.set_grad_enabled(False)
cfg = parse_configs(training=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

# Build text-encoder and VAE
text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

# Prepare video size
image_size = cfg.get("image_size")
num_frames = get_num_frames(cfg.num_frames)
input_size = (num_frames, *image_size)
latent_size = vae.get_latent_size(input_size)

# Build diffusion model
model = build_module(
    cfg.model,
    MODELS,
    input_size=latent_size,
    in_channels=vae.out_channels + 1,
    caption_channels=text_encoder.output_dim,
    model_max_length=text_encoder.model_max_length,
    enable_sequence_parallelism=False,  # assume no distributed env for simplicity
).to(device, dtype).eval()
text_encoder.y_embedder = model.y_embedder

# Build scheduler
scheduler = build_module(cfg.scheduler, SCHEDULERS)


def process_video(input_video, mask_video, sketch_video, prompt, cfg_guidance_scale, num_sampling_steps, start_frame):
    # import pdb; pdb.set_trace()
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    os.makedirs('outputs', exist_ok=True)

    generate_size = 512

    video, _ = read_video(input_video)
    video_sketch, _ = read_video(sketch_video)
    if mask_video is None:
        video_mask = torch.ones(1, 1, generate_size, generate_size)
        video_mask_ori = torch.ones(1, 1, generate_size, generate_size)
    else:
        video_mask, video_mask_ori, _ = read_mask(mask_video)

    gt = torch.nn.functional.interpolate(video, size=(generate_size, generate_size), mode='bilinear',align_corners=False)
    gt = (gt - 127.5) / 127.5

    video_mask = torch.nn.functional.interpolate(video_mask, size=(generate_size, generate_size), mode='bilinear',align_corners=False) > 0
    video_mask_ori = torch.nn.functional.interpolate(video_mask_ori, size=(generate_size, generate_size),mode='bilinear', align_corners=False) > 0
    video_sketch = torch.nn.functional.interpolate(video_sketch, size=(generate_size, generate_size), mode='bilinear',align_corners=False)

    gt = gt[start_frame:start_frame + 51]
    video_mask = video_mask[start_frame:start_frame + 51]
    video_mask_ori = video_mask_ori[start_frame:start_frame + 51]
    video_sketch = video_sketch[start_frame:start_frame + 51]

    video_sketch = video_sketch * video_mask
    vae_hint = (video_sketch.clone() - 127.5) / 127.5
    video_sketch = video_sketch / 255.

    gt = gt.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_mask = video_mask.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_mask_ori = video_mask_ori.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_sketch = video_sketch.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    vae_hint = vae_hint.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype)
    video_mask_ds = down_sample_mask(video_mask, vae.micro_frame_size, down_time=4, down_height=8, down_width=8).to(device, dtype)

    model_args = prepare_multi_resolution_info(
        multi_resolution, len([prompt]), image_size, num_frames, fps, device, dtype
    )

    with torch.no_grad():
        z_gt = vae.encode(gt)
    z = torch.randn(len([prompt]), vae.out_channels, *latent_size, device=device, dtype=dtype)
    z = z * video_mask_ds + z_gt * (1. - video_mask_ds)

    masks = torch.ones(len([prompt]), latent_size[0]).to(z.device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        samples = scheduler.sample(
            model, text_encoder, z=z, prompts=[prompt], device=device,
            additional_args=model_args, mask=masks, hint_ori=video_sketch, video_mask=video_mask_ds,
            num_sampling_steps=num_sampling_steps, guidance_scale=cfg_guidance_scale, gradio_bar=gr.Progress(),
        )

        features_2d, features_3d = vae.sketch_vae.encode_detail(vae_hint)
        samples = vae.decode_detail_with_sketch(samples.to(dtype), num_frames=num_frames, features_2d=features_2d,
                                                features_3d=features_3d)
        samples_with_gt = samples * video_mask_ori + gt * (1.0 - video_mask_ori)

    timestamp = int(time.time())
    save_path = f'outputs/output_{timestamp}.mp4'
    save_sample(samples_with_gt[0], fps=save_fps, save_path=save_path)

    return save_path


# 获取视频的帧数
def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


# 事件：点击生成按钮
def inference(input_video_path, mask_video_path, sketch_video_path, prompt, cfg_scale, num_steps, start_frame):
    # 处理逻辑使用原视频分辨率
    output = process_video(input_video_path, mask_video_path, sketch_video_path, prompt, cfg_scale, num_steps,
                           start_frame)
    return output


with gr.Blocks(
        title="VIRES",
        theme=gr.themes.Soft(),
        css="""
        .video-box { margin-bottom: 15px; max-width: 100%; }
        .video-container { 
            max-height: 350px; 
            max-width: 350px;  # 添加宽度限制，与高度匹配 1:1
            overflow: hidden; 
            display: flex; 
            justify-content: center; 
            align-items: center;  # 居中显示视频
        }
        .video-container video { 
            object-fit: contain;  # 保持视频比例，不裁剪
            height: 100%; 
            width: 100%; 
        }
    """
) as demo:
    gr.Markdown("# OpenSora Video Generation")
    with gr.Row():
        # 左侧：三个视频输入框，适配 1:1 比例
        with gr.Column(scale=2):
            input_video = gr.Video(
                label="Input Video",
                interactive=True,
                elem_classes=["video-box", "video-container"]
            )
            mask_video = gr.Video(
                label="Mask Video",
                interactive=True,
                elem_classes=["video-box", "video-container"]
            )
            sketch_video = gr.Video(
                label="Sketch Video",
                interactive=True,
                elem_classes=["video-box", "video-container"]
            )
        # 右侧：输出视频和其他输入
        with gr.Column(scale=3):
            output_video = gr.Video(label="Generated Video", height=400)
            with gr.Group():
                prompt = gr.Textbox(label="Prompt")
                cfg_scale = gr.Slider(1, 10, value=6.0, step=0.1, label="CFG Guidance Scale")
                num_steps = gr.Slider(10, 50, value=30, step=1, label="Number of Sampling Steps")
                start_frame = gr.Slider(0, 1000, value=0, step=1, label="Start Frame", maximum=1000)
                submit_btn = gr.Button("Generate", variant="primary")


    # 事件：更新滑块最大值
    def update_slider_maximum(video_path):
        frame_count = get_video_frame_count(video_path)  # 假设此函数获取原视频帧数
        return gr.update(maximum=frame_count - 51)


    input_video.change(
        fn=update_slider_maximum,
        inputs=input_video,
        outputs=start_frame
    )

    submit_btn.click(
        fn=inference,
        inputs=[input_video, mask_video, sketch_video, prompt, cfg_scale, num_steps, start_frame],
        outputs=output_video
    )


demo.launch(server_name="0.0.0.0", server_port=8080)