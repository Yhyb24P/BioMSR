# 显微图像超分辨率复合损失函数 (Hybrid Loss) - 细胞级优化版
# 调整方向：降低频域权重，引入梯度损失以强化边缘锐度（Edge Sharpening）
# 目标：服务于下游的细胞分割与计数任务，减少细胞内部伪影

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from math import exp

class SSIMLoss(nn.Module):
    """
    结构相似性损失 (Structural Similarity Index Measure Loss)
    针对显微图像的单通道 (Gray) 特性进行了优化。
    """
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class FFTLoss(nn.Module):
    """ 频域损失：约束幅度和相位的一致性 """
    def __init__(self, loss_weight=0.01, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        loss_mag = self.criterion(torch.abs(pred_fft), torch.abs(target_fft))
        loss_phase = self.criterion(torch.angle(pred_fft), torch.angle(target_fft))
        return self.loss_weight * (loss_mag + loss_phase)

class GradientLoss(nn.Module):
    """ 
    梯度损失 (Edge Sharpening Loss) 
    利用 Sobel 算子计算图像梯度，强制模型恢复清晰的细胞边缘
    """
    def __init__(self, loss_weight=0.1):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        # 定义 Sobel 卷积核
        self.register_buffer('kernel_x', torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3))

    def forward(self, pred, target):
        # 计算 X 和 Y 方向的梯度
        grad_pred_x = F.conv2d(pred, self.kernel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.kernel_y, padding=1)
        grad_target_x = F.conv2d(target, self.kernel_x, padding=1)
        grad_target_y = F.conv2d(target, self.kernel_y, padding=1)
        
        # 计算 L1 梯度差异
        loss_grad = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)
        return self.loss_weight * loss_grad

class HybridLoss(nn.Module):
    """
    BioSR 细胞级专用复合损失函数
    New Strategy: L1 + SSIM + Reduced FFT + Gradient
    """
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_fft=0.01, w_grad=0.1):
        super(HybridLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        self.w_grad = w_grad
        
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=11)
        self.fft = FFTLoss(loss_weight=1.0)
        self.grad = GradientLoss(loss_weight=1.0)

    def forward(self, pred, target):
        loss = 0
        loss_dict = {}
        
        # 1. Pixel Loss (保真度)
        if self.w_l1 > 0:
            l_l1 = self.l1(pred, target)
            loss += self.w_l1 * l_l1
            loss_dict['l1'] = l_l1.item()
            
        # 2. Structural Loss (结构一致性)
        if self.w_ssim > 0:
            l_ssim = self.ssim(pred, target)
            loss += self.w_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()
            
        # 3. Frequency Loss (大幅降低权重，仅保留基础约束)
        if self.w_fft > 0:
            l_fft = self.fft(pred, target)
            loss += self.w_fft * l_fft
            loss_dict['fft'] = l_fft.item()

        # 4. Gradient Loss (新增：强化细胞边缘锐度)
        if self.w_grad > 0:
            l_grad = self.grad(pred, target)
            loss += self.w_grad * l_grad
            loss_dict['grad'] = l_grad.item()
            
        return loss, loss_dict