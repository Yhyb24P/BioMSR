# 端到端联合训练器 (End-to-End Trainer)
# 串联 SR 模型与 下游任务模型 (如 Segmentation)
# 目标: 优化 SR 模型，使其输出不仅视觉清晰，更能提升下游分割精度

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.archs.swinir import SwinIR_Bio
from src.archs.unet_care import UNet_CARE  # 假设用一个 U-Net 作为简化的分割网络演示
from src.losses.hybrid_loss import HybridLoss

class End2EndTrainer:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # 1. 初始化 SR 模型 (Generator)
        self.sr_model = SwinIR_Bio(upscale=config['upscale']).to(device)
        
        # 2. 初始化 分割模型 (Task Model)
        # 假设分割网络输出 1 个通道 (Probability Map) 或 StarDist 的 (N+1) 通道
        # 这里以简单的二值分割为例
        self.seg_model = UNet_CARE(in_channels=1, out_channels=1, residual=False).to(device)
        
        # 如果有预训练的分割模型，务必加载！
        if config.get('seg_pretrained'):
            self.seg_model.load_state_dict(torch.load(config['seg_pretrained']))
            print("Loaded pretrained segmentation model.")
            
        # 3. 优化器
        # 我们主要想训练 SR 模型，分割模型可以微调 (Fine-tune) 或 冻结 (Freeze)
        params_to_opt = list(self.sr_model.parameters())
        if config.get('finetune_seg', False):
            params_to_opt += list(self.seg_model.parameters())
        else:
            self.seg_model.eval() # 冻结模式
            for param in self.seg_model.parameters():
                param.requires_grad = False
                
        self.optimizer = optim.AdamW(params_to_opt, lr=config['lr'])
        self.scaler = GradScaler()
        
        # 4. 损失函数
        self.sr_criterion = HybridLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss() # 假设是二分类分割

    def train_epoch(self, train_loader, epoch):
        self.sr_model.train()
        # 如果微调分割，则 seg_model.train()，否则 .eval()
        if self.config.get('finetune_seg', False):
            self.seg_model.train()
        else:
            self.seg_model.eval()
            
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"E2E Epoch {epoch}")
        
        for batch in pbar:
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)     # SR 的真值
            mask = batch['mask'].to(self.device) # 分割的真值 (GT Mask)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # 1. SR 前向传播
                sr_img = self.sr_model(lr)
                
                # 2. 计算 SR 损失
                loss_sr, _ = self.sr_criterion(sr_img, hr)
                
                # 3. 分割前向传播 (输入是生成的 SR 图!)
                # 注意：如果 seg_model 冻结，SR 图作为输入仍会保留计算图以回传梯度
                seg_logits = self.seg_model(sr_img)
                
                # 4. 计算 分割损失 (Task Loss)
                loss_seg = self.seg_criterion(seg_logits, mask)
                
                # 5. 联合损失
                # lambda_task 是关键超参，平衡视觉质量与任务精度
                lambda_task = self.config.get('lambda_task', 0.1)
                loss_total = loss_sr + lambda_task * loss_seg
            
            # 反向传播
            self.scaler.scale(loss_total).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.sr_model.parameters(), 0.5)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss_total.item()
            pbar.set_postfix({'L_sr': f"{loss_sr.item():.3f}", 'L_seg': f"{loss_seg.item():.3f}"})
            
        return total_loss / len(train_loader)