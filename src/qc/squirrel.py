# NanoJ-SQUIRREL 质量控制模块 (Python PyTorch 版)
# 核心思想：SR图像退化后应与原始LR图像在统计上一致
# 用途：生成误差图 (Error Map) 和 全局指标 (RSE, RSP)

import torch
import torch.nn.functional as F
import numpy as np

class SquirrelQC:
    """
    SQUIRREL: Super-resolution Quantitative Image Rating and Reporting of Error Locations
    """
    def __init__(self, upscale_factor=2, device='cuda'):
        self.scale = upscale_factor
        self.device = device

    def calculate_metrics(self, sr_tensor, lr_tensor):
        """
        计算 SQUIRREL 指标
        Args:
            sr_tensor: [B, 1, H_hr, W_hr] - 超分辨结果 (已归一化到 0-1)
            lr_tensor: [B, 1, H_lr, W_lr] - 原始低分辨输入 (已归一化到 0-1)
        Returns:
            rse (float): 相对平方误差 (越小越好)
            rsp (float): 线性相关系数 (越大越好, close to 1)
            error_map (Tensor): 误差图 [B, 1, H_lr, W_lr]
        """
        # 1. 模拟退化过程 (Simulate Degradation)
        # 注意：理想情况下应使用显微镜的 PSF 卷积，这里使用 Bicubic 近似
        # 确保尺寸严格匹配
        H_lr, W_lr = lr_tensor.shape[2:]
        sr_simulated = F.interpolate(
            sr_tensor, 
            size=(H_lr, W_lr), 
            mode='bicubic', 
            align_corners=False
        )
        
        # 2. 亮度与对比度校准 (Intensity Rescaling)
        # 求解线性方程: LR ≈ alpha * SR_sim + beta
        # 对每个 Batch 独立计算
        batch_size = sr_tensor.shape[0]
        rse_list = []
        rsp_list = []
        error_maps = []

        for i in range(batch_size):
            S = sr_simulated[i].view(-1)
            R = lr_tensor[i].view(-1)
            
            # 计算线性回归参数
            # alpha = Cov(S, R) / Var(S)
            mean_s = torch.mean(S)
            mean_r = torch.mean(R)
            
            numerator = torch.sum((S - mean_s) * (R - mean_r))
            denominator = torch.sum((S - mean_s) ** 2) + 1e-8 # 防止除零
            
            alpha = numerator / denominator
            beta = mean_r - alpha * mean_s
            
            # 匹配后的模拟图
            S_matched = alpha * sr_simulated[i] + beta
            
            # 3. 计算误差图 (Error Map)
            # SQUIRREL 定义 Error Map 为绝对差值
            diff = torch.abs(S_matched - lr_tensor[i])
            error_maps.append(diff.unsqueeze(0))
            
            # 4. 计算 RSE (Root Scaled Error)
            # RSE = ||LR - S_matched|| / ||LR - mean(LR)||
            rmse = torch.sqrt(torch.sum(diff ** 2))
            norm_factor = torch.sqrt(torch.sum((R - mean_r)**2)) + 1e-8
            rse = rmse / norm_factor
            rse_list.append(rse.item())
            
            # 5. 计算 RSP (Pearson Correlation)
            # 就是前面的 numerator / (std_s * std_r)
            std_s = torch.sqrt(denominator)
            std_r = norm_factor
            rsp = numerator / (std_s * std_r + 1e-8)
            rsp_list.append(rsp.item())

        return np.mean(rse_list), np.mean(rsp_list), torch.stack(error_maps)

    def get_heatmap(self, error_map):
        """ 辅助函数：将误差图转为热力图 (用于 TensorBoard 显示) """
        # 简单的归一化用于显示
        return error_map / (error_map.max() + 1e-8)