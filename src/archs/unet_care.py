# 经典 U-Net 架构，适配 CARE (Content-Aware Image Restoration) 框架
# 主要用于 SR 前的降噪 (Denoising) 或 简单 SR 任务
# 采用 Residual Learning 策略：模型学习的是 Residual (Noise) 而非直接输出图像

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ 基础卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet_CARE(nn.Module):
    """
    Bio-medical U-Net for Denoising/Restoration
    Args:
        in_channels (int): 输入通道数 (默认为1，灰度图)
        out_channels (int): 输出通道数
        n_filters (int): 起始卷积核数量
        residual (bool): 是否采用全局残差学习 (Input + Output)
    """
    def __init__(self, in_channels=1, out_channels=1, n_filters=32, residual=True):
        super(UNet_CARE, self).__init__()
        self.residual = residual
        
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(in_channels, n_filters)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = ConvBlock(n_filters * 4, n_filters * 8)
        
        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters)
        
        self.final = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        
        # Bridge
        b = self.bridge(p3)
        
        # Decoder
        d3 = self.up3(b)
        # Skip connection need to handle size mismatch if input size is not power of 2
        # Here we assume padding is handled or input is standard size
        if d3.size() != x3.size():
            d3 = F.interpolate(d3, size=x3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.size() != x2.size():
            d2 = F.interpolate(d2, size=x2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.size() != x1.size():
            d1 = F.interpolate(d1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        
        if self.residual:
            # Global Residual: Learning the noise/restoration map
            # Output = Input + Restoration
            out = out + x
            
        return out