# 结果可视化与绘图工具
# 支持: 多图并列展示, 误差热力图绘制, ROI 局部放大

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# 引入归一化工具
from .img_process import percentile_normalization

def show_sr_results(lr, sr, hr=None, error_map=None, save_path=None, title_suffix=""):
    """
    绘制标准的 SR 对比图: LR | SR | HR (Optional) | Error Map (Optional)
    """
    # 转换为 Numpy 并归一化以便显示
    img_list = []
    title_list = []
    
    # LR
    lr_np = percentile_normalization(lr)
    if lr_np.ndim == 3: lr_np = lr_np.squeeze()
    img_list.append(lr_np)
    title_list.append(f"Low Res Input\n(Raw)")

    # SR
    sr_np = percentile_normalization(sr)
    if sr_np.ndim == 3: sr_np = sr_np.squeeze()
    img_list.append(sr_np)
    title_list.append(f"Super Res\n(SwinIR)")

    # HR (如果有)
    if hr is not None:
        hr_np = percentile_normalization(hr)
        if hr_np.ndim == 3: hr_np = hr_np.squeeze()
        img_list.append(hr_np)
        title_list.append(f"Ground Truth\n(Reference)")

    # Error Map (如果有)
    if error_map is not None:
        if torch.is_tensor(error_map):
            err_np = error_map.detach().cpu().numpy().squeeze()
        else:
            err_np = error_map.squeeze()
        img_list.append(err_np)
        title_list.append(f"SQUIRREL Error\n(Magma)")

    # 绘图
    num_imgs = len(img_list)
    fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
    if num_imgs == 1: axes = [axes]

    for i, ax in enumerate(axes):
        img = img_list[i]
        
        # 针对 Error Map 使用特定 colormap
        if "Error" in title_list[i]:
            im = ax.imshow(img, cmap='magma', vmin=0, vmax=np.percentile(img, 99))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img, cmap='gray')
            
        ax.set_title(title_list[i])
        ax.axis('off')
        
        # 添加局部放大 (针对 SR 和 HR)
        if "Super Res" in title_list[i] or "Ground Truth" in title_list[i]:
            add_zoom_inset(ax, img, zoom=2.5)

    if title_suffix:
        plt.suptitle(f"SR Result Analysis - {title_suffix}", fontsize=14)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def add_zoom_inset(ax, img, zoom=2.5, loc=2):
    """
    在图像上添加局部放大框 (Zoom Inset)
    自动选择图像中心区域进行放大
    """
    h, w = img.shape
    # 定义感兴趣区域 (ROI) - 默认取中心
    cy, cx = h // 2, w // 2
    roi_h, roi_w = h // 8, w // 8
    
    y1, y2 = cy - roi_h, cy + roi_h
    x1, x2 = cx - roi_w, cx + roi_w
    
    # 创建嵌入轴
    axins = zoomed_inset_axes(ax, zoom, loc=loc) # loc=2 (upper left)
    
    # 根据类型选择 cmap
    cmap = 'magma' if 'Error' in ax.get_title() else 'gray'
    axins.imshow(img, cmap=cmap, origin="upper")
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1) # 注意坐标系方向
    
    # 移除刻度
    axins.set_xticks([])
    axins.set_yticks([])
    
    # 画连接线
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="yellow", lw=1)