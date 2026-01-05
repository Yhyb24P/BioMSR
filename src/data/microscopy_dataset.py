# 显微镜图像数据集加载器
# 专为 BioSR/W2S 等数据集设计，处理 16-bit TIFF 图像
# 包含自动切块 (Patching) 和 归一化逻辑

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff  # 相比 PIL，tifffile 对科学图像支持更好
from pathlib import Path

class MicroscopyDataset(Dataset):
    """
    显微图像数据集加载器。
    
    Args:
        lr_dir (str): 低分辨率(LR)图像文件夹路径
        hr_dir (str): 高分辨率(HR)图像文件夹路径 (训练模式必需)
        is_train (bool): 是否为训练模式。训练模式下会应用数据增强和切块。
        patch_size (int): 训练时的切块大小 (HR尺度)。LR切块会自动除以 upscale_factor。
        upscale_factor (int): 超分倍率 (通常为 2 或 4)。
        transform (callable, optional): 几何变换函数。
        noise_aug (callable, optional): 物理噪声增强函数 (仅作用于 LR)。
        data_range (float): 归一化范围，通常为 65535.0 (16-bit) 或 255.0 (8-bit)。
    """
    def __init__(self, lr_dir, hr_dir=None, is_train=True, 
                 patch_size=128, upscale_factor=2, 
                 transform=None, noise_aug=None, data_range=65535.0):
        super(MicroscopyDataset, self).__init__()
        
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir) if hr_dir else None
        self.is_train = is_train
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.noise_aug = noise_aug
        self.data_range = float(data_range)

        # 扫描文件
        self.lr_files = sorted(list(self.lr_dir.glob('*.tif*')))
        
        if self.is_train and self.hr_dir:
            self.hr_files = sorted(list(self.hr_dir.glob('*.tif*')))
            # 简单校验文件名匹配
            assert len(self.lr_files) == len(self.hr_files), \
                f"LR和HR文件数量不匹配: {len(self.lr_files)} vs {len(self.hr_files)}"
        else:
            self.hr_files = []

    def __len__(self):
        return len(self.lr_files)

    def _read_img(self, path):
        """ 读取 TIFF 并归一化到 [0, 1] """
        img = tiff.imread(str(path))
        img = img.astype(np.float32) / self.data_range
        
        # 增加通道维度: (H, W) -> (C, H, W)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        # 如果是 (H, W, C)，转为 (C, H, W)
        elif img.ndim == 3 and img.shape[2] <= 4:
            img = np.transpose(img, (2, 0, 1))
            
        return torch.from_numpy(img)

    def _get_patch(self, lr, hr):
        """ 随机切块 (Random Crop) """
        lr_h, lr_w = lr.shape[1], lr.shape[2]
        hr_patch_size = self.patch_size
        lr_patch_size = hr_patch_size // self.upscale_factor

        # 随机选取左上角坐标 (LR grid)
        tx = random.randrange(0, lr_w - lr_patch_size + 1)
        ty = random.randrange(0, lr_h - lr_patch_size + 1)

        # 对应的 HR 坐标
        tx_hr, ty_hr = tx * self.upscale_factor, ty * self.upscale_factor

        # Crop
        lr_patch = lr[:, ty:ty+lr_patch_size, tx:tx+lr_patch_size]
        hr_patch = hr[:, ty_hr:ty_hr+hr_patch_size, tx_hr:tx_hr+hr_patch_size]

        return lr_patch, hr_patch

    def __getitem__(self, idx):
        # 1. 读取数据
        lr_path = self.lr_files[idx]
        lr_img = self._read_img(lr_path)

        if self.is_train and self.hr_files:
            hr_path = self.hr_files[idx]
            hr_img = self._read_img(hr_path)
            
            # 2. 随机切块 (仅在训练时且图像大于Patch时)
            if self.patch_size < hr_img.shape[1] and self.patch_size < hr_img.shape[2]:
                lr_img, hr_img = self._get_patch(lr_img, hr_img)
            
            # 3. 几何变换 (翻转/旋转)
            if self.transform:
                lr_img, hr_img = self.transform(lr_img, hr_img)
                
            # 4. 物理噪声注入 (Data Augmentation)
            # 注意：噪声只加在 LR 输入上，HR 保持纯净作为 GT
            if self.noise_aug:
                lr_img = self.noise_aug(lr_img)
                
            return {'lr': lr_img, 'hr': hr_img, 'path': str(lr_path.name)}
            
        else:
            # 推理模式
            return {'lr': lr_img, 'path': str(lr_path.name)}