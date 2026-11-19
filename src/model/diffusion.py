"""
擴散先驗模型（沒有運作的模組）
============================

對應原 notebook：Cell 0 的 DiffusionPrior
拆解筆記：4.7

作者原本想用它從語音資料學習「語音的分布」，再拿來對齊腦訊號。
但訓練時從來沒有提供語音資料（speech_latent=None），所以這段完全不會執行（問題 P2）。

v0 仍然保留，因為作者的權重檔裡有這個模組的參數（21,276,672 個），
少了它就無法用嚴格模式載入權重。
這裡的去噪網路使用 PyTorch 內建的 TransformerEncoderLayer，不受 transformer_impl 影響。
"""

import torch
import torch.nn as nn

from src.model.positional import SinusoidalPosEmb


class DiffusionPrior(nn.Module):

    def __init__(self, d_model=512, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps

        # 雜訊排程：每一步加多少雜訊
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # 時間步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # 6 層去噪 Transformer
        self.denoiser = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(6)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def q_sample(self, x_0, t, noise=None):
        """加噪：x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · 雜訊"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    def forward(self, x_t, t):
        """去噪：預測加進去的雜訊"""
        t_emb = self.time_embed(t).unsqueeze(1)    # (B, 1, d_model)
        x = x_t + t_emb
        for layer in self.denoiser:
            x = layer(x)
        return self.out_proj(x)

    @torch.no_grad()
    def sample(self, z_neural, n_steps=None):
        """生成：從純雜訊一步步去噪"""
        if n_steps is None:
            n_steps = self.n_steps
        B, L, D = z_neural.shape
        x = torch.randn_like(z_neural)
        for t in reversed(range(n_steps)):
            t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
            noise_pred = self.forward(x, t_batch)
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            x = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
            if t > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        return x
