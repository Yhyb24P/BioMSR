# 针对 LR-HR 图像对的同步几何变换
# 实现随机翻转和旋转，利用显微图像的旋转不变性扩充数据

import random
import torch
import torchvision.transforms.functional as TF

class PairedRandomTransforms:
    """
    对 LR 和 HR 图像对同时应用相同的随机几何变换。
    包括: 随机水平翻转, 随机垂直翻转, 随机旋转 90/180/270 度
    """
    def __init__(self, flip_prob=0.5, rot_prob=0.5):
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob

    def __call__(self, lr, hr):
        """
        Args:
            lr (Tensor): [C, H, W]
            hr (Tensor): [C, H*scale, W*scale]
        """
        # 1. 随机水平翻转
        if random.random() < self.flip_prob:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        # 2. 随机垂直翻转
        if random.random() < self.flip_prob:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        # 3. 随机旋转 (0, 90, 180, 270)
        if random.random() < self.rot_prob:
            # 随机选择旋转角度
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            if angle > 0:
                lr = TF.rotate(lr, angle)
                hr = TF.rotate(hr, angle)

        return lr, hr

# 简单的 Compose 包装
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr):
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr