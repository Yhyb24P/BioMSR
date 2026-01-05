# 幻觉指数 (Hallucination Index) 计算模块
# 基于统计分布差异 (Hellinger Distance) 评估生成伪影的风险

import torch
import torch.nn.functional as F
import numpy as np

class HallucinationIndex:
    def __init__(self, bins=50, min_val=0.0, max_val=1.0):
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val

    def _compute_histogram(self, tensor):
        """ 计算张量的归一化直方图 (概率密度函数 PDF) """
        # 将张量展平
        data = tensor.float().view(-1)
        # 截断
        data = torch.clamp(data, self.min_val, self.max_val)
        
        # 计算直方图
        hist = torch.histc(data, bins=self.bins, min=self.min_val, max=self.max_val)
        
        # 归一化 (Sum = 1)
        prob = hist / (torch.sum(hist) + 1e-8)
        return prob

    def _hellinger_distance(self, p, q):
        """ 计算两个概率分布之间的 Hellinger 距离 """
        # H(P, Q) = (1/sqrt(2)) * sqrt( sum( (sqrt(pi) - sqrt(qi))^2 ) )
        return (1.0 / np.sqrt(2.0)) * torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))

    def _get_structure_map(self, img):
        """ 
        提取结构特征图 (如 Sobel 梯度幅值) 
        相比原始像素，梯度分布更能反映纹理真实性
        """
        # Sobel Kernel
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float=True, device=img.device).view(1,1,3,3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float=True, device=img.device).view(1,1,3,3)
        
        gx = F.conv2d(img, kx, padding=1)
        gy = F.conv2d(img, ky, padding=1)
        
        grad_mag = torch.sqrt(gx**2 + gy**2)
        return grad_mag

    def calculate_hi(self, sr_img, ref_img):
        """
        计算单张图像的幻觉指数
        Args:
            sr_img: [1, 1, H, W] 超分辨结果
            ref_img: [1, 1, H, W] 参考真实图像 (GT 或 同类高质量图像)
        Returns:
            hi_score: 幻觉指数 (越低越好，0表示分布完全一致)
        """
        # 1. 提取结构特征
        # 仅比较纹理分布，忽略亮度差异，先归一化
        sr_norm = (sr_img - sr_img.min()) / (sr_img.max() - sr_img.min() + 1e-8)
        ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min() + 1e-8)
        
        feat_sr = self._get_structure_map(sr_norm)
        feat_ref = self._get_structure_map(ref_norm)
        
        # 2. 计算特征分布 (Gradient Magnitude Histogram)
        # 梯度的范围通常在 [0, 2] 之间 (归一化图卷积后)
        # 临时调整直方图范围
        original_max = self.max_val
        self.max_val = 2.0 
        
        pdf_sr = self._compute_histogram(feat_sr)
        pdf_ref = self._compute_histogram(feat_ref)
        
        self.max_val = original_max # 恢复
        
        # 3. 计算 Hellinger 距离
        hi = self._hellinger_distance(pdf_sr, pdf_ref)
        
        return hi.item()