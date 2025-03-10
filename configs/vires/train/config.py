# Dataset settings
dataset = dict(
    type = "VariableVideoTextGrayHintMaskDataset",
    data_path = '',
    num_frames = 51,
    frame_interval = 1,
    image_size = (512, 512),
    transform_name = "resize_crop",
)

grad_checkpoint = True

# Acceleration settings
num_workers = 2
num_bucket_build_workers = 4
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="VIRES",
    from_pretrained="checkpoints/VIRES/",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    freeze_y_embedder=True,
)
vae = dict(
    type="OpenSoraVAE_With_Sketch_V1_2",
    from_pretrained="checkpoints/VIRES_VAE",
    sdxlvae_path="checkpoints/VIRES_VAE/sdxlvae/",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="checkpoints/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="control_mask_rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
# 25%
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.002,
    "quarter_random": 0.002,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.002,
    "image_random": 0.0,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 10
ckpt_every = 500

# continue training
# load = '/xxxx/xxx-VIRES/epoch0-global_step10000'


# optimization settings
batch_size = 1
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000
