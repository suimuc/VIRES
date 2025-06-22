# [CVPR2025] VIRES: Video Instance Repainting via Sketch and Text Guided Generation
## [Project page](https://hjzheng.net/projects/VIRES/) | [Paper](https://arxiv.org/abs/2411.16199)
Official implementation of VIRES: Video Instance Repainting with Sketch and Text Guidance, which is accepted by CVPR 2025.
```html
<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
  <tr>
    <td>
      <video align='center' src="https://hjzheng.net/projects/VIRES/demo_video.mp4" muted autoplay loop></video>
    </td>
  </tr>
</table>
```
# Showcase
```html
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
    <td><strong>Input</strong></td>
    <td><strong>Mask</strong></td>
    <td><strong>Output</strong></td>
  </tr>
  <tr>
      <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/car/input.mp4" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/car/mask.mp4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/car/VIRES.mp4" width="100%" controls autoplay loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/dog/input.mp4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/dog/mask.mp4" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://hjzheng.net/projects/VIRES/Teasers/dog/VIRES.mp4" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>
```
# Installation
### Setup Environment
For CUDA 12.1, you can install the dependencies with the following commands. Otherwise, you need to manually install `torch`, `torchvision` and `xformers`.
```shell
# create a virtual env and activate (conda as an example)
conda create -n vires python=3.9
conda activate vires

# download the repo
git clone https://github.com/suimuc/VIRES
cd VIRES

# install torch, torchvision and xformers
pip install -r requirements-cu121.txt

# install others packages
pip install --no-deps -r requirements.txt
pip install -v -e .
# install flash attention
# set enable_flash_attn=True in config to enable flash attention
pip install flash-attn --no-build-isolation
# install apex
# set enable_layernorm_kernel=True and shardformer=True in config to enable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git@24.04.01
```

# Inference
### Model Weights
| Model     | Download Link|
|-----------| ----------------------------------------------- |
| VIRES     | [:link:](https://huggingface.co/suimu/VIRES) |
| VIRES-VAE | [:link:](https://huggingface.co/suimu/VIRES_VAE) |
| T5        | [:link:](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) |

To run VIRES, please follow these steps:

1- Download models using huggingface-cli:
```shell
huggingface-cli download suimu/VIRES --local-dir ./checkpoints/VIRES
huggingface-cli download suimu/VIRES_VAE --local-dir ./checkpoints/VIRES_VAE
huggingface-cli download DeepFloyd/t5-v1_1-xxl --local-dir ./checkpoints/t5-v1_1-xxl
```
2- Prepare a config file under `configs/vires/inference`, and make sure that the model paths 
for `model`, `text_encoder`, and `vae` in the config file match the paths of the models you just downloaded.

3- Run the following command:

The basic command line inference is as follows:
```shell
python scripts/inference.py configs/vires/inference/config.py \
--save-dir ./outputs/ --input_video "assets/clothes_input.mp4" \
--sketch_video "assets/clothes_sketch.avi" --mask_video "assets/clothes_mask.avi" \
--prompt "The video features a man and a woman walking side by side on a paved pathway. The man is dressed in a blue jacket, with his hands clasped behind his back." \
--cfg_guidance_scale 7 \
--sampling_steps 30
```
To enable sequence parallelism, you need to use torchrun to run the inference script. 
The following command will run the inference with 2 GPUs:
```shell
torchrun --nproc_per_node 2 scripts/inference.py configs/vires/inference/config.py \
--save-dir ./outputs/ --input_video "assets/clothes_input.mp4" \
--sketch_video "assets/clothes_sketch.avi" --mask_video "assets/clothes_mask.avi" \
--prompt "The video features a man and a woman walking side by side on a paved pathway. The man is dressed in a blue jacket, with his hands clasped behind his back." \
--cfg_guidance_scale 7 \
--sampling_steps 30
```

4- The results will be generated under the `save-dir` directory. Note that by default, 
the configuration will only edit the first 51 frames of the given video (frames 0 to 50). 
If you want to edit any arbitrary 51 frames in the video, use the `--start_frame` option. 
After running the command using `--start_frame`, the terminal will prompt you to enter the starting frame number, 
and the script will then process frames from `start_frame` to `start_frame + 50`.
### WebUI Demo
To run our grad.io based web demo, prepare all the model weights, 
ensure the config file has the correct model paths, and then run the following command:
```shell
python scripts/app.py configs/vires/inference/config.py
```

# Dataset
Download the [VIRESET](https://huggingface.co/datasets/suimu/VIRESET) from huggingface. After get the csv file contain 
absolute path of video clips and corresponding json files, 
provide the path of csv file to the `VariableVideoTextGrayHintMaskDataset` in `opensora/datasets/datasets.py`, 
Then, the dataset will return a video tensor of the specified image_size (normalized between -1 and 1) 
and a mask tensor (with values of 0 or 1).


# Training
To train VIRES, please follow these steps:

1- Download VIRESET refer to [Dataset](#dataset)

2- Prepare a config file under `configs/vires/train`, specify the CSV file path in the `data_path` field 
of the dataset dictionary and make sure the model paths  for `model`, `text_encoder`, and `vae` in the config file 
match the paths of the models you just downloaded.

3- Download the HED model for sketch generation.
```shell
wget -O "checkpoints/ControlNetHED.pth" "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
```

4- Run the following command:
```shell
torchrun --standalone --nproc_per_node 8 scripts/train.py configs/vires/train/config.py --outputs your_experiment_dir
```
5- During training, 
all data,including model weights, optimizer states, and loss values logged in TensorBoard format,
will be saved in `your_experiment_dir`, and you can configure the parameters `epochs`, `log_every`, 
and `ckpt_every` in the configuration file to specify the number of training epochs, 
the interval for logging loss, and the interval for saving checkpoints, respectively.

6- If training is interrupted, you can configure the `load` parameter in the configuration file 
to resume training from a specific step. Alternatively, you can use `--load` in the command line.

**NOTE**: Due to the high number of input and output channels in the 3D convolutions of the **Sequential ControlNet**, 
GPU memory usage is significantly increased. As a result, the training process for VIRES 
was conducted on a setup with 8 GPUs, each equipped with 96 GH100 cards. 

It is recommended for users to reduce the input and output channels in the **Sequential ControlNet**’s 3D convolutions, 
which can be found in the `opensora/models/stdit/vires.py` file, `line168-183`. 
When making these changes, you only need to ensure that the last out_channels of `self.hint_mid_convs` matches the `config.hidden_size`.

Rest assured, reducing the channels will not significantly degrade the model’s performance, but it will greatly reduce memory usage.

# Citation
```bibtex
@article{vires,
    title={VIRES: Video Instance Repainting via Sketch and Text Guided Generation},
    author={Weng, Shuchen and Zheng, Haojie and Zhang, Peixuan and Hong, Yuchen and Jiang, Han and Li, Si and Shi, Boxin},
    journal={arXiv preprint arXiv:2411.16199},
    year={2024}
}
```
