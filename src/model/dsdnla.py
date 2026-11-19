"""
DSDNLA：把四個模組組在一起的外殼
================================

對應原 notebook：Cell 0 的 DSDNLA
拆解筆記：4.8

  ① NeuralEncoder      ✅ 有運作
  ② DiffusionPrior     ❌ 沒有運作（v0 保留，為了載入權重）
  ③ CrossModalAligner  ❌ 沒有運作（同上）
  ④ TextDecoder        ✅ 有運作

參數量（字母表 36）：75,173,412
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.aligner import CrossModalAligner
from src.model.decoder import TextDecoder
from src.model.diffusion import DiffusionPrior
from src.model.encoder import NeuralEncoder


class DSDNLA(nn.Module):

    def __init__(self, n_channels=512, d_model=512, vocab_size=256, n_encoder_layers=8,
                 n_decoder_layers=6, n_heads=8, dropout=0.1, transformer_impl='pytorch'):
        super().__init__()
        self.neural_encoder = NeuralEncoder(
            n_channels=n_channels, d_model=d_model, n_layers=n_encoder_layers,
            n_heads=n_heads, dropout=dropout, impl=transformer_impl)
        self.diffusion_prior = DiffusionPrior(d_model=d_model)
        self.aligner = CrossModalAligner(d_model=d_model)
        self.text_decoder = TextDecoder(
            d_model=d_model, vocab_size=vocab_size, n_layers=n_decoder_layers,
            n_heads=n_heads, dropout=dropout, impl=transformer_impl)

    def forward(self, neural_features, speech_latent=None, target_tokens=None, training=True):
        """
        注意：參數 training 和 model.train() / model.eval() 是兩回事
          training：只控制要不要計算擴散和對齊的損失
          model.eval()：控制 Dropout、BatchNorm 的行為
        """
        z_neural = self.neural_encoder(neural_features)    # P5：沒有傳 mask

        losses = {}
        if training and speech_latent is not None:          # P2：speech_latent 永遠是 None
            B = z_neural.size(0)
            t = torch.randint(0, self.diffusion_prior.n_steps, (B,), device=z_neural.device)
            z_noisy, noise_true = self.diffusion_prior.q_sample(speech_latent, t)
            noise_pred = self.diffusion_prior(z_noisy, t)
            losses['diffusion'] = F.mse_loss(noise_pred, noise_true)
            alignment_out = self.aligner(z_neural, speech_latent)
            losses['alignment'] = alignment_out['loss']
            z_for_decode = z_neural
        else:
            z_for_decode = z_neural

        logits = self.text_decoder(z_for_decode, target_tokens)
        return logits, losses

    @torch.no_grad()
    def inference(self, neural_features, max_len=200):
        """逐字產生（走的是有 P3 問題的 generate()；最終提交用的是集束搜尋，不走這裡）"""
        z_neural = self.neural_encoder(neural_features)
        return self.text_decoder.generate(z_neural, max_len=max_len)


def build_model(cfg, vocab_size):
    """[新增] 依設定檔建立模型。"""
    m = cfg['model']
    return DSDNLA(
        n_channels=m['n_channels'],
        d_model=m['d_model'],
        vocab_size=vocab_size,
        n_encoder_layers=m['n_encoder_layers'],
        n_decoder_layers=m['n_decoder_layers'],
        n_heads=m['n_heads'],
        dropout=m['dropout'],
        transformer_impl=m.get('transformer_impl', 'pytorch'),
    )


def count_parameters(model):
    """[新增] 各模組的參數量，回傳 [{'module':..., 'params':...}, ...]"""
    rows = []
    for name, child in model.named_children():
        rows.append({'module': name, 'params': sum(p.numel() for p in child.parameters())})
    rows.append({'module': 'total', 'params': sum(p.numel() for p in model.parameters())})
    return rows
