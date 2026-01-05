import torch
import torch.nn as nn
import torch.fft

class FFTLoss(nn.Module):
    """ 频域损失：约束幅度和相位的一致性 """
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        # 实数FFT
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        
        # 幅度损失 (Magnitude Loss) - 恢复锐度
        loss_mag = self.criterion(torch.abs(pred_fft), torch.abs(target_fft))
        
        # 相位损失 (Phase Loss) - 恢复结构位置
        loss_phase = self.criterion(torch.angle(pred_fft), torch.angle(target_fft))
        
        return self.loss_weight * (loss_mag + loss_phase)

class HybridLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_fft=0.1):
        super(HybridLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_fft = w_fft
        
        self.l1 = nn.L1Loss()
        self.fft = FFTLoss(loss_weight=1.0)
        # SSIM 需要引入 torchvision 或 kornia 库，此处简化为占位
        # self.ssim = KorniaSSIM() 

    def forward(self, pred, target):
        loss = 0
        loss_dict = {}
        
        # Pixel Loss
        if self.w_l1 > 0:
            l_l1 = self.l1(pred, target)
            loss += self.w_l1 * l_l1
            loss_dict['l1'] = l_l1.item()
            
        # Frequency Loss
        if self.w_fft > 0:
            l_fft = self.fft(pred, target)
            loss += self.w_fft * l_fft
            loss_dict['fft'] = l_fft.item()
            
        return loss, loss_dict
