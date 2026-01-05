# PatchGAN Discriminator (for optional GAN-based training)
# 警告：仅在需要极高视觉锐度且已做好充分幻觉验证（SQUIRREL check）的情况下使用
# 对于纯定量分析任务（如光子计数），建议禁用判别器，仅使用 SwinIR + L1/SSIM Loss

import torch
import torch.nn as nn

class Discriminator_PatchGAN(nn.Module):
    """
    PatchGAN 判别器。
    输出是一个特征图，图中每个像素代表原图中一块区域（Patch）的真假概率。
    """
    def __init__(self, in_channels=1, n_filters=64):
        super(Discriminator_PatchGAN, self).__init__()

        self.model = nn.Sequential(
            # Input: (B, C, H, W)
            
            # Layer 1: [B, 64, H/2, W/2]
            nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: [B, 128, H/4, W/4]
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: [B, 256, H/8, W/8]
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: [B, 512, H/16, W/16]
            nn.Conv2d(n_filters * 4, n_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer: [B, 1, H/16, W/16] (Stride=1, no compression)
            nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=1)
            # Sigmoid is usually handled in the Loss function (BCEWithLogitsLoss) for stability
        )

    def forward(self, x):
        return self.model(x)

# 辅助函数：初始化权重
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)