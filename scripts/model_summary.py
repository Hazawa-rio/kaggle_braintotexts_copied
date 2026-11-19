"""
實驗三：模型摘要
================

依照設定檔建立模型，計算各模組的參數量，並用隨機資料跑一次前向計算，確認形狀。
不需要資料、不需要 GPU。

用法：
  python scripts/model_summary.py
  python scripts/model_summary.py --set model.n_encoder_layers=4       # 看模型會變多小
  python scripts/model_summary.py --set model.transformer_impl=mine    # 用你自己寫的 Transformer

v0 的正確答案：
  neural_encoder   26,532,352
  diffusion_prior  21,276,672
  aligner           2,103,296
  text_decoder     25,261,092
  total            75,173,412
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.config import load_config
from src.experiment import Experiment
from src.model.dsdnla import build_model, count_parameters
from src.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description='模型參數量與形狀檢查')
    parser.add_argument('--config', default='configs/v0.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    if 'experiment_name' not in changes:
        cfg['experiment_name'] = 'model_summary' + ''.join(
            f"_{k.split('.')[-1]}-{v}" for k, (_, v) in changes.items())

    tokenizer = CharTokenizer()
    with Experiment(cfg, changes) as exp:
        model = build_model(cfg, tokenizer.vocab_size)
        rows = count_parameters(model)
        exp.save_artifact('parameters.csv', rows)
        for r in rows:
            print(f"  {r['module']:<16} {r['params']:>12,}")

        # 用隨機資料跑一次：一句 3 秒的話（150 個時間點）、20 個 token
        model.eval()
        B, T, S = 2, 150, 20
        x = torch.randn(B, T, cfg['model']['n_channels'])
        tokens = torch.randint(3, tokenizer.vocab_size, (B, S))
        with torch.no_grad():
            z = model.neural_encoder(x)
            logits, _ = model(x, target_tokens=tokens, training=False)
        print(f'  編碼器：{tuple(x.shape)} → {tuple(z.shape)}')
        print(f'  整個模型：{tuple(x.shape)} + {tuple(tokens.shape)} → {tuple(logits.shape)}')

        result = {r['module']: r['params'] for r in rows}
        result['encoder_out_shape'] = str(tuple(z.shape))
        result['logits_shape'] = str(tuple(logits.shape))
        exp.finish(result)


if __name__ == '__main__':
    main()
