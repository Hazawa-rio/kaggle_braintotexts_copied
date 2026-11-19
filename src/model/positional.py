"""
位置編碼
========

對應原 notebook：Cell 0 的 PositionalEncoding、SinusoidalPosEmb（照原樣搬過來）
拆解筆記：4.4

  PositionalEncoding：告訴 Transformer「每個位置是第幾個」。
                      注意力機制本身不分順序，所以要另外加上位置資訊。
  SinusoidalPosEmb： 擴散模型用的「時間步」編碼（v0 保留，實際上沒有執行）
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦位置編碼（Attention Is All You Need, 2017）

    輸入：x (B, L, d_model)
    輸出：(B, L, d_model)  形狀不變，每個位置加上一組獨一無二的 sin / cos 數值
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                                   # (5000, 512)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 0, 1, ..., 4999
        # 每一對維度的「轉速」不同：前面的轉得快、後面的轉得慢
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # 偶數維用 sin
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇數維用 cos
        pe = pe.unsqueeze(0)                            # (1, 5000, 512)，方便和批次相加
        # register_buffer：固定值、不是可學習的參數，但會存進模型檔
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]    # 只取前 L 個位置
        return self.dropout(x)


class SinusoidalPosEmb(nn.Module):
    """擴散模型的時間步編碼：把整數 t 轉成 dim 維的向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
