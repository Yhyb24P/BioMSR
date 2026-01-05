# 显微图像专用处理工具包
# 包含: 直方图匹配 (域适应), 鲁棒归一化, 位深转换

import numpy as np
import torch

def percentile_normalization(img, p_min=0.1, p_max=99.9, clip=True):
    """
    鲁棒百分位归一化 (Robust Percentile Normalization)
    抗热像素干扰，显微图像预处理的标准操作。
    
    Args:
        img: Numpy array or Torch tensor
        p_min, p_max: 归一化的下限和上限百分位
        clip: 是否将超出范围的值截断到 [0, 1]
    """
    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()
        is_tensor = True
    else:
        img_np = img
        is_tensor = False
        
    # 计算百分位
    vmin = np.percentile(img_np, p_min)
    vmax = np.percentile(img_np, p_max)
    
    # 防止除零 (纯黑图像)
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
        
    img_norm = (img_np - vmin) / (vmax - vmin)
    
    if clip:
        img_norm = np.clip(img_norm, 0.0, 1.0)
        
    if is_tensor:
        return torch.from_numpy(img_norm).to(img.device)
    return img_norm

def histogram_matching(source, template):
    """
    直方图匹配 (Histogram Matching) - 用于简单域适应
    将 source 图像的像素值分布调整为与 template 图像一致。
    
    Args:
        source: 输入图像 (Target Domain, e.g., Lab Data)
        template: 模板图像 (Source Domain, e.g., BioSR Data)
    Returns:
        matched: 匹配后的图像
    """
    # 确保输入为 Numpy
    if torch.is_tensor(source):
        src = source.detach().cpu().numpy()
    else:
        src = source
    if torch.is_tensor(template):
        tmpl = template.detach().cpu().numpy()
    else:
        tmpl = template

    oldshape = src.shape
    src = src.ravel()
    tmpl = tmpl.ravel()

    # 计算源和模板的累积分布函数 (CDF)
    s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(tmpl, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # 插值映射
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    matched = interp_t_values[bin_idx].reshape(oldshape)
    
    if torch.is_tensor(source):
        return torch.from_numpy(matched).float().to(source.device)
    return matched

def tensor2uint16(tensor):
    """ 将 [0, 1] 的 Tensor 转换为 uint16 格式 (0-65535) 用于保存 TIFF """
    img = tensor.detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 65535).astype(np.uint16)
    
    # 移除 Batch 和 Channel 维度如果它们是 1
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
        
    return img