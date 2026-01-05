# 标准超分辨率模型训练器 (SR Trainer)
# 封装了 SwinIR 的训练、验证、模型保存与加载逻辑
# 集成了混合精度训练 (AMP) 以支持大图输入

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import math

# 导入项目模块 (假设路径已在 PYTHONPATH 中)
from src.archs.swinir import SwinIR_Bio
from src.losses.hybrid_loss import HybridLoss

class SRTrainer:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # 1. 初始化模型
        self.model = SwinIR_Bio(
            img_size=config['patch_size'],
            patch_size=config['patch_size'],
            in_chans=1,
            embed_dim=config['embed_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            upscale=config['upscale']
        ).to(device)
        
        # 2. 初始化损失函数
        self.criterion = HybridLoss(
            w_l1=config['w_l1'], 
            w_ssim=config['w_ssim'], 
            w_fft=config['w_fft']
        ).to(device)
        
        # 3. 优化器与调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-7
        )
        
        # 4. 混合精度 Scaler
        self.scaler = GradScaler()
        
        # 记录最佳指标
        self.best_psnr = 0.0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch in pbar:
            # 数据迁移
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                sr = self.model(lr)
                loss, loss_dict = self.criterion(sr, hr)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪 (防止 Transformer 梯度爆炸)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'l1': f"{loss_dict['l1']:.4f}"})
            
        self.scheduler.step()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        for batch in val_loader:
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            
            with autocast():
                sr = self.model(lr)
            
            # 计算指标 (简单版，实际可调用 src.qc.squirrel)
            mse = nn.MSELoss()(sr, hr).item()
            psnr = 10 * math.log10(1 / mse) if mse > 0 else 100
            total_psnr += psnr
            
        avg_psnr = total_psnr / len(val_loader)
        
        # 保存最佳模型
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint('best_model.pth')
            
        return avg_psnr

    def save_checkpoint(self, filename):
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr
        }, save_path)
        print(f"Saved model to {save_path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {path}")