import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import paddle
from paddle.io import Dataset
from paddle.vision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales # 好像没用
        self.interp = Image.ANTIALIAS  # 等比压缩

        self.frame_idxs = frame_idxs # [0, -1, 1, "s"]

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.brightness = 0.2 # 亮度
        self.contrast = 0.2 # 对比度
        self.saturation = 0.2 # 饱和度
        self.hue = 0.1 # 色调

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation='nearest') # changed

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and np.random.random() > 0.5
        do_flip = self.is_train and np.random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0] # 文件夹

        if len(line) == 3: # 如果是单目序列
            frame_index = int(line[1])
        else: # 立体图像
            frame_index = 0

        if len(line) == 3:
            side = line[2] # 是左图像还是右图像
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid 调整内参以匹配金字塔中的每个比例'
        # 因为模型有4个尺度，所以对应4个相机内参
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K) # 求伪逆，用于最小二乘

            inputs[("K", scale)] = paddle.to_tensor(K)
            inputs[("inv_K", scale)] = paddle.to_tensor(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter()._get_param(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x) # def f(x): return x

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0) # 增加一个维度
            inputs["depth_gt"] = paddle.to_tensor(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32) # 生成对角阵，主对角线上元素为1，其余位置均为0
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = paddle.to_tensor(stereo_T)  # 外参

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
