"""
神經編碼器
==========

對應原 notebook：Cell 0 的 NeuralEncoder
拆解筆記：4.5

把腦訊號轉成「理解過整句上下文」的表示，交給解碼器。

  輸入：x (B, T, 512)        每 20 毫秒一組 512 個特徵
  輸出：z_neural (B, L, 512)  L ≈ T / 2

[新增] 參數 impl：
  'pytorch'：Transformer 部分用 PyTorch 內建的 nn.TransformerEncoder（原 notebook 的寫法）
  'mine'   ：改用 src/model/my_transformer.py 裡你自己寫的版本
  兩者的參數名稱完全相同，所以作者的權重兩邊都能載入。
"""

import torch.nn as nn

from src.model.my_transformer import MyTransformerEncoder, MyTransformerEncoderLayer
from src.model.positional import PositionalEncoding


class NeuralEncoder(nn.Module):

    def __init__(self, n_channels=512, d_model=512, n_layers=8, n_heads=8, dropout=0.1,
                 impl='pytorch'):
        super().__init__()

        # [原作者] 1D 卷積：一次看 5 個時間點（100 毫秒），每次移 2 格 → 長度減半
        #          輸出長度 L = floor((T + 2×2 − 5) / 2) + 1 = ceil(T / 2)
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # [原作者] 位置編碼
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=5000)

        # [原作者] 8 層 Transformer 編碼器，Pre-LN（norm_first=True）
        layer_args = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,   # 512 × 4 = 2048
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        if impl == 'pytorch':
            encoder_layer = nn.TransformerEncoderLayer(**layer_args)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        elif impl == 'mine':                                        # [新增]
            encoder_layer = MyTransformerEncoderLayer(**layer_args)
            self.transformer = MyTransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            raise ValueError(f"impl 只能是 'pytorch' 或 'mine'，收到 {impl!r}")

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape                 # (B, T, 512)
        x = x.transpose(1, 2)             # → (B, 512, T)    Conv1d 要求通道在中間
        x = self.input_proj(x)            # → (B, 512, T/2)  時間減半
        x = x.transpose(1, 2)             # → (B, T/2, 512)
        x = self.pos_enc(x)               # 形狀不變，加入位置資訊

        # P5：mask 參數存在，但 DSDNLA 呼叫時沒有傳入，補齊的 0 也會被注意力看到
        z_neural = self.transformer(x, src_key_padding_mask=mask)
        z_neural = self.layer_norm(z_neural)
        return z_neural                   # (B, L, 512)
