"""
損失函數
========

對應原 notebook：Cell 1 的 DSDNLALoss
拆解筆記：4.9

  總損失 = 1.0 × 文字損失 + 0.5 × 擴散損失 + 0.3 × 對齊損失
  但擴散和對齊從來沒有被計算（問題 P2），所以實際上 總損失 = 文字損失
"""

import torch.nn as nn


class DSDNLALoss(nn.Module):

    def __init__(self, text_weight=1.0, diffusion_weight=0.5, alignment_weight=0.3):
        super().__init__()
        self.text_weight = text_weight
        self.diffusion_weight = diffusion_weight
        self.alignment_weight = alignment_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)   # 補齊的 PAD（0）不算損失

    def forward(self, logits, target_tokens, model_losses):
        """
        logits：(B, S, V)  解碼器輸出
        target_tokens：(B, S)

        「往右移一格」：
          解碼器輸入：     <BOS>  h   i   !  <EOS>
          輸出位置：        p0   p1  p2  p3   p4
          要對照的答案：     h    i   !  <EOS>      ← target[:, 1:]
          p4 沒有答案，所以用 logits[:, :-1] 丟掉
        """
        total_loss = 0
        losses = {}

        if target_tokens is not None:
            B, T, V = logits.shape
            logits_flat = logits[:, :-1].reshape(-1, V)
            target_flat = target_tokens[:, 1:].reshape(-1)
            text_loss = self.ce_loss(logits_flat, target_flat)
            losses['text'] = text_loss
            total_loss += self.text_weight * text_loss

        if 'diffusion' in model_losses:
            losses['diffusion'] = model_losses['diffusion']
            total_loss += self.diffusion_weight * model_losses['diffusion']

        if 'alignment' in model_losses:
            losses['alignment'] = model_losses['alignment']
            total_loss += self.alignment_weight * model_losses['alignment']

        losses['total'] = total_loss
        return losses
