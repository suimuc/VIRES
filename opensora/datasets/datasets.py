import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFile
from torchvision.transforms import Compose, RandomHorizontalFlip
import random
from opensora.datasets.utils import video_transforms
from opensora.registry import DATASETS
import json
import base64
import cv2
import pycocotools.mask as mask_util
from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop, \
    read_video, read_gray_with_gamma_correction, temporal_random_crop_1, crop_to_square, get_transforms_video_2

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


@DATASETS.register_module()
class VariableVideoTextGrayHintMaskDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path=None,
            num_frames=51,
            frame_interval=1,
            image_size=(512, 512),
            transform_name=None,

    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))

    def getitem(self, index):

        sample = self.data.iloc[index]
        path = sample["path"]
        text = sample["text"]

        height = self.image_size[0]
        width = self.image_size[1]
        num_frames = self.num_frames
        ar = height / width

        vframes, video_fps = read_video(path)

        random_value = random.uniform(1, 5)
        if (random.random() < 0.5):
            gamma = random_value
        else:
            gamma = 1.0 / random_value
        edges, _ = read_gray_with_gamma_correction(path, gamma)

        video, condition = temporal_random_crop_1(vframes, edges, num_frames, self.frame_interval)

        del vframes

        video_fps = video_fps // self.frame_interval

        # load mask
        json_path = sample["json_path"]
        masklet_id = sample["masklet_id"]
        mask_start_frame = sample["mask_start_frame"]
        mask_end_frame = sample["mask_end_frame"]
        min_y = sample['min_y']
        max_y = sample['max_y']
        min_x = sample['min_x']
        max_x = sample['max_x']
        annot = json.load(open(json_path))
        masks = []

        for i in range(mask_start_frame, mask_end_frame + 1):
            tmp = annot["masklet_continues"][i][masklet_id]
            tmp['counts'] = base64.b64decode(tmp['counts'].encode('utf-8'))
            mask_ori = mask_util.decode(tmp) > 0
            mask_ori = mask_ori.astype('uint8')
            h, w = mask_ori.shape
            re = min(h // 128, w // 128)
            mask = cv2.resize(mask_ori, (w // re, h // re), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # Gauss
            mask = cv2.GaussianBlur(mask, (5, 5), 4)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            masks.append(np.expand_dims(np.expand_dims(mask, axis=0), axis=0))

        masks = [torch.from_numpy(mask) for mask in masks]
        masks = torch.cat(masks, dim=0)
        masks = masks.to(torch.float)

        crop_min_x, crop_max_x, crop_min_y, crop_max_y = crop_to_square(video.shape[3], video.shape[2], min_x, max_x,
                                                                        min_y, max_y)
        video = video[:, :, crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        masks = masks[:, :, crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        condition = condition[:, :, crop_min_y:crop_max_y, crop_min_x:crop_max_x]

        video = torch.nn.functional.interpolate(video, size=self.image_size, mode='bilinear',
                                                align_corners=False)  # T C H W
        masks = torch.nn.functional.interpolate(masks, size=self.image_size, mode='bilinear',
                                                align_corners=False)  # T C H W
        condition = torch.nn.functional.interpolate(condition, size=self.image_size, mode='bilinear',
                                                    align_corners=False)  # T C H W

        # transform
        video = video / 255. * 2 - 1
        masks = masks > 0
        masks = masks.to(torch.float)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        masks = masks.permute(1, 0, 2, 3)

        ret = {
            "video": video,
            "video_mask": masks,
            "text": text,
            "edge": condition,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "bound_box": [crop_min_x, crop_max_x, crop_min_y, crop_max_y],
            "fps": video_fps,
        }
        return ret

    def __getitem__(self, index):
        for _ in range(30):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
            self,
            data_path=None,
            num_frames=16,
            frame_interval=1,
            image_size=(256, 256),
            transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            # vframes, vinfo = read_video(path, backend="av")
            vframes, video_fps = extract_frames(path)
            # video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
            self,
            data_path=None,
            num_frames=None,
            frame_interval=1,
            image_size=(None, None),
            transform_name=None,
            dummy_text_feature=False,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # print(index)
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            # vframes, vinfo = read_video(path, backend="av")
            vframes, video_fps, edges = extract_frames_edges(path)
            # video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video, condition = temporal_random_crop_1(vframes, edges, num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
            transform = get_transforms_condition(self.transform_name, (height, width))
            condition = transform(condition)
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        condition = condition.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "edge": condition,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret