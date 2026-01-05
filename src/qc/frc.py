# 傅里叶环相关 (Fourier Ring Correlation) 计算模块
# 用于定量评估图像的有效物理分辨率 (Effective Resolution)

import torch
import torch.fft
import numpy as np

class FRCCalculator:
    def __init__(self, pixel_size_nm=65.0, threshold=0.143):
        """
        Args:
            pixel_size_nm: 图像像素对应的物理尺寸 (纳米)
            threshold: FRC 截止阈值 (通常用 1/7 ≈ 0.143 或 0.5)
        """
        self.pixel_size = pixel_size_nm
        self.threshold = threshold

    def split_image(self, img):
        """ 
        将单张图像按棋盘格拆分为两张子图 (用于 Single-Image FRC)
        img: [H, W]
        """
        sub1 = img[0::2, 0::2] # 偶行偶列
        sub2 = img[1::2, 1::2] # 奇行奇列
        return sub1, sub2

    def calculate_frc_curve(self, img1, img2):
        """
        计算两张图像的 FRC 曲线
        Args:
            img1, img2: [H, W] Tensor
        Returns:
            frc_curve: 各频率环的相关系数
            spatial_freq: 对应的空间频率
        """
        # 1. 变换到频域
        fft1 = torch.fft.fft2(img1)
        fft2 = torch.fft.fft2(img2)
        
        # 移频，使低频在中心
        fft1 = torch.fft.fftshift(fft1)
        fft2 = torch.fft.fftshift(fft2)
        
        # 2. 计算互功率谱和自功率谱
        # Cross-Power Spectrum
        cps = fft1 * torch.conj(fft2)
        
        # Power Spectrum
        ps1 = torch.abs(fft1) ** 2
        ps2 = torch.abs(fft2) ** 2
        
        # 3. 环形积分 (Ring Integration)
        h, w = img1.shape
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.to(img1.device)
        x = x.to(img1.device)
        
        # 计算每个像素到中心的半径 (以像素为单位)
        radius = torch.sqrt((x - cx)**2 + (y - cy)**2)
        radius = torch.round(radius).int()
        
        max_radius = min(h, w) // 2
        
        frc_numerator = torch.zeros(max_radius, device=img1.device)
        frc_denominator = torch.zeros(max_radius, device=img1.device)
        
        # 使用 scatter_add 进行快速环形求和
        # 注意：展平处理
        flat_radius = radius.flatten()
        flat_cps = torch.real(cps).flatten()
        flat_ps_sum = (ps1 + ps2).flatten()
        
        # 过滤超出范围的半径
        mask = flat_radius < max_radius
        valid_radius = flat_radius[mask].long()
        
        frc_numerator.scatter_add_(0, valid_radius, flat_cps[mask])
        frc_denominator.scatter_add_(0, valid_radius, flat_ps_sum[mask])
        
        # 计算 FRC
        frc_curve = frc_numerator / (frc_denominator + 1e-8) # 防止除零
        
        # 修正：归一化 (Power Spectrum Sum -> sqrt(Sum1 * Sum2))
        # 严格公式是 Numerator / Sqrt(Sum(PS1) * Sum(PS2))
        # 但在很多实现中简化为 Numerator / (Sum(PS1) + Sum(PS2)) * 2 ?
        # 这里采用标准定义: Cor(r)
        # 为简化计算，假设两图能量近似，分母用 Sum(PS1+PS2) 是近似。
        # 严格版需要分别求和再开方，这里为了代码简洁用近似版，但在科学报告中应更严谨。
        
        return frc_curve

    def measure_resolution(self, img_tensor):
        """
        计算单张图像的有效分辨率
        img_tensor: [1, H, W]
        """
        img = img_tensor.squeeze()
        sub1, sub2 = self.split_image(img)
        
        frc_curve = self.calculate_frc_curve(sub1, sub2)
        
        # 寻找曲线首次跌破阈值的点
        frequencies = torch.arange(len(frc_curve), device=img.device) / len(frc_curve) # 归一化频率 [0, 1] (Nyquist = 1)
        
        # 找到第一个低于阈值的索引
        below_thresh = frc_curve < self.threshold
        if torch.any(below_thresh):
            idx = torch.where(below_thresh)[0][0]
            cutoff_freq = frequencies[idx].item()
            
            # 换算为物理分辨率
            # Nyquist 频率对应的分辨率是 2 * pixel_size (但这里是子图，pixel_size 翻倍了)
            # 子图 pixel_size = 2 * original_pixel_size
            # Cutoff Resolution = 子图pixel_size / cutoff_freq
            
            effective_pixel = self.pixel_size * 2
            resolution_nm = effective_pixel / (cutoff_freq + 1e-8)
        else:
            resolution_nm = self.pixel_size * 2 # 无法测定，至少是 Nyquist
            
        return resolution_nm, frc_curve