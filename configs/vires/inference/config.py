image_size = (512, 512)
num_frames = 51
fps = 25
save_fps = 25
frame_interval = 1

save_dir = "./outputs/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"

input_video = 'assets/clothes_input.mp4'
sketch_video = 'assets/clothes_sketch.avi'
mask_video = 'assets/clothes_mask.avi'
prompt = "The video features a man and a woman walking side by side on a paved pathway. The man is dressed in a blue jacket, blue pants, and black shoes, with his hands clasped behind his back. The woman is wearing a blue sweater, blue pants, and black shoes, and she has a black shoulder bag. They are walking in an outdoor setting that appears to be a public area, possibly a park or a promenade, as indicated by the presence of greenery, flowers, and other people in the background. The lighting suggests it is daytime, and the shadows cast on the ground indicate that the sun is at a low angle, possibly in the morning or late afternoon."

model = dict(
    type = "VIRES",
    from_pretrained = "checkpoints/VIRES/",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
)
text_encoder = dict(
    type="t5",
    from_pretrained="checkpoints/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="control_mask_rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)
vae = dict(
    type = "OpenSoraVAE_With_Sketch_V1_2",
    from_pretrained="checkpoints/VIRES_VAE",
    sdxlvae_path = "checkpoints/VIRES_VAE/sdxlvae/",
    micro_frame_size=17,
    micro_batch_size=4,
)