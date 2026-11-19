"""
訓練器
======

對應原 notebook：Cell 1 的 Trainer
拆解筆記：4.10

訓練一步：
  ① 混合精度前向計算 → ② 算損失 → ③ 反向傳播 → ④ 梯度裁剪 → ⑤ 更新參數 → ⑥ 調整學習率

[新增] 的部分（不改變訓練邏輯）：
  - 設定從 yaml 讀取，每一輪的數據交給 Experiment 記錄與匯出
  - 模型檔存在該次實驗的資料夾，並存入「真正的」完整設定（原作者存的是空字典，問題 P9）
  - 沒有 GPU 時自動關閉混合精度，讓 CPU 也能試跑
  - 新版 PyTorch 的寫法 torch.amp.autocast / torch.amp.GradScaler（功能和原本相同）
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.losses import DSDNLALoss


class Trainer:

    def __init__(self, model, cfg, device, train_loader, experiment=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.t = cfg['train']
        self.device = device
        self.exp = experiment

        # [原作者] AdamW，betas 和 eps 是原始 Transformer 論文的設定
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.t['learning_rate'],
            weight_decay=self.t['weight_decay'],
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # [原作者] OneCycleLR：前 5% 步數升高，之後依餘弦曲線下降
        # 問題 P7：原作者假設每輪 1,000 步，實際步數是 len(train_loader)
        if self.t.get('use_assumed_lr_steps', True):
            total_steps = self.t['num_epochs'] * self.t.get('lr_steps_per_epoch_assumed', 1000)
        else:
            total_steps = self.t['num_epochs'] * len(train_loader)
        self.total_steps = total_steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.t['learning_rate'],
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
        )

        loss_cfg = cfg['loss']
        self.criterion = DSDNLALoss(
            text_weight=loss_cfg['text_weight'],
            diffusion_weight=loss_cfg['diffusion_weight'],
            alignment_weight=loss_cfg['alignment_weight'],
        )

        # [新增] 沒有 GPU 時關閉混合精度
        self.use_amp = bool(self.t['use_amp']) and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            neural = batch['neural'].to(self.device)
            tokens = batch['tokens'].to(self.device) if batch['tokens'] is not None else None

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits, model_losses = self.model(
                    neural,
                    speech_latent=None,          # P2：從來沒有提供語音資料
                    target_tokens=tokens,
                    training=True,
                )
                losses = self.criterion(logits, tokens, model_losses)
                total_loss = losses['total']

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.t['gradient_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_losses.append(total_loss.item())
            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
            self.global_step += 1

        return float(np.mean(epoch_losses))

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        for batch in tqdm(val_loader, desc='Validation'):
            neural = batch['neural'].to(self.device)
            tokens = batch['tokens'].to(self.device) if batch['tokens'] is not None else None
            logits, model_losses = self.model(neural, None, tokens, training=False)
            losses = self.criterion(logits, tokens, model_losses)
            val_losses.append(losses['total'].item())
        return float(np.mean(val_losses))

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg,              # [新增] 存入真正的完整設定（修正 P9 的紀錄問題）
        }, path)
        print(f'Saved new best model to: {path}')

    def train(self, train_loader, val_loader):
        """
        訓練到 num_epochs 輪，或驗證損失連續 patience 輪沒進步就提早停止。
        問題 P8：只用驗證損失挑模型，沒有計算 WER
        """
        ckpt_path = os.path.join(self.exp.dir if self.exp else '.', 'best_model.pt')
        stopped_early = False
        epoch = -1

        for epoch in range(self.t['num_epochs']):
            start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            improved = val_loss < self.best_val_loss

            if improved:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, ckpt_path)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.exp:
                self.exp.log_metrics({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr_end_of_epoch': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step,
                    'best_so_far': improved,
                    'epoch_sec': round(time.time() - start, 1),
                })

            if self.patience_counter >= self.t['patience']:
                print(f"Early stopping after {self.t['patience']} epochs without improvement.")
                stopped_early = True
                break

        return {
            'best_val_loss': self.best_val_loss,
            'epochs_run': epoch + 1,
            'stopped_early': stopped_early,
            'scheduler_total_steps': self.total_steps,
            'steps_run': self.global_step,
            'checkpoint': ckpt_path,
        }
