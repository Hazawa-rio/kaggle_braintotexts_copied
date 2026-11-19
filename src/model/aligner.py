"""
跨模態對齊器（沒有運作的模組）
============================

對應原 notebook：Cell 0 的 CrossModalAligner
拆解筆記：4.7

作者原本想用對比學習（InfoNCE）把腦訊號的表示拉近語音的表示。
但沒有提供語音資料、損失權重也是 0，所以實際上沒有作用（問題 P2）。

v0 保留的原因同 diffusion.py：作者的權重檔裡有這個模組的參數（2,103,296 個）。
這裡的注意力使用 PyTorch 內建的 nn.MultiheadAttention，不受 transformer_impl 影響。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAligner(nn.Module):

    def __init__(self, d_model=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        self.proj_neural = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.proj_speech = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.alignment_net = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=0.1, batch_first=True
        )

    def info_nce_loss(self, z_neural, z_speech):
        B, L_n, D = z_neural.shape
        _, L_s, _ = z_speech.shape

        z_n = F.normalize(self.proj_neural(z_neural), dim=-1)
        z_s = F.normalize(self.proj_speech(z_speech), dim=-1)

        aligned_z_s, attn_weights = self.alignment_net(z_n, z_s, z_s)
        pos_sim = (z_n * aligned_z_s).sum(dim=-1)   # 原作者有算，但沒有用到

        z_n_flat = z_n.reshape(B * L_n, D)
        z_s_flat = z_s.reshape(B * L_s, D)
        logits = torch.matmul(z_n_flat, z_s_flat.T) / self.temperature
        # 原作者的標籤寫法假設兩邊位置一一對應，長度不同時並不合理（見筆記 4.7）
        labels = torch.arange(B * L_n, device=logits.device) % (B * L_s)
        loss = F.cross_entropy(logits, labels)
        return loss, attn_weights

    def forward(self, z_neural, z_speech):
        loss, attn_weights = self.info_nce_loss(z_neural, z_speech)
        aligned_speech, _ = self.alignment_net(z_neural, z_speech, z_speech)
        return {'loss': loss, 'aligned_speech': aligned_speech, 'attn_weights': attn_weights}
