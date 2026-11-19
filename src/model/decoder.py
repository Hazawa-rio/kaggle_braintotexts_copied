"""
文字解碼器
==========

對應原 notebook：Cell 0 的 TextDecoder
拆解筆記：4.6

根據編碼器的輸出，一個字母一個字母寫出句子。

兩種模式：
  訓練：教師強制（把正確答案整句餵進去，每個位置預測下一個字母）
  推論：逐字產生（寫一個、加進輸入、再預測下一個）

[新增] 參數 impl：'pytorch'（原寫法）或 'mine'（你自己寫的 Transformer）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.my_transformer import MyTransformerDecoder, MyTransformerDecoderLayer
from src.model.positional import PositionalEncoding


class TextDecoder(nn.Module):

    def __init__(self, d_model=512, vocab_size=256, n_layers=6, n_heads=8, dropout=0.1,
                 impl='pytorch'):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # [原作者] 每個字母對應一個 512 維向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # [原作者] 6 層 Transformer 解碼器，Pre-LN
        layer_args = dict(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        if impl == 'pytorch':
            decoder_layer = nn.TransformerDecoderLayer(**layer_args)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        elif impl == 'mine':                                        # [新增]
            decoder_layer = MyTransformerDecoderLayer(**layer_args)
            self.decoder = MyTransformerDecoder(decoder_layer, num_layers=n_layers)
        else:
            raise ValueError(f"impl 只能是 'pytorch' 或 'mine'，收到 {impl!r}")

        # [原作者] 512 維 → 每個字母一個分數
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, z_neural, target_tokens=None, max_len=200):
        """
        z_neural：(B, L, 512) 編碼器的輸出
        target_tokens：(B, S) 正確答案（訓練時）；None 時改走逐字產生
        回傳：(B, S, vocab_size)
        """
        if target_tokens is not None:
            # 教師強制
            tgt_emb = self.token_embedding(target_tokens)     # (B, S) → (B, S, 512)
            tgt_emb = self.pos_enc(tgt_emb)

            # 因果遮罩：每個位置只能看自己和前面（浮點版本，上三角是 -inf）
            T = target_tokens.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(z_neural.device)

            # P5：沒有傳 memory_key_padding_mask，交叉注意力也會看到腦訊號補的 0
            out = self.decoder(tgt_emb, z_neural, tgt_mask=causal_mask)
            logits = self.output_proj(out)                    # (B, S, vocab_size)
            return logits
        else:
            return self.generate(z_neural, max_len)

    @torch.no_grad()
    def generate(self, z_neural, max_len=200, temperature=1.0):
        """
        逐字產生（只有不用集束搜尋時才會走到這裡）

        P3：特殊符號編號錯誤（v0 保留）
            這裡假設 0 = BOS、1 = EOS，但 CharTokenizer 是 0 = PAD、1 = BOS、2 = EOS
            → 用 PAD 開頭，看到 BOS 才停
        P4：用抽樣（multinomial）而不是取最大值，每次結果不同
        """
        B = z_neural.size(0)
        device = z_neural.device

        generated = torch.zeros(B, 1, dtype=torch.long, device=device)   # P3：應為 BOS = 1

        for _ in range(max_len):
            tgt_emb = self.token_embedding(generated)
            tgt_emb = self.pos_enc(tgt_emb)

            T = generated.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.decoder(tgt_emb, z_neural, tgt_mask=causal_mask)

            logits = self.output_proj(out[:, -1, :])                     # 只看最後一個位置
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)        # P4

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 1).all():                                  # P3：應為 EOS = 2
                break

        return generated
