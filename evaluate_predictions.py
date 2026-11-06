"""
實驗一：評估預測結果
====================

讀入一份「正確句 vs 預測句」的 CSV，計算 WER 和三種錯誤，並匯出所有數據。
不需要資料集、不需要 GPU，現在就能跑。

之後拿到案例 B 在驗證集上的輸出時，也是用這支程式評分。

輸入 CSV 的格式（欄位名稱要一樣，id 可省略）：
  id,reference,prediction
  0,I want some water,i want sum water
  1,How are you?,how you

用法（在 repo 根目錄）：
  python scripts/evaluate_predictions.py --input examples/sample_predictions.csv

調整參數（例如關掉正規化，看分數差多少）：
  python scripts/evaluate_predictions.py --input examples/sample_predictions.csv \
      --set eval.normalize=false --set experiment_name=eval_no_normalize
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import load_config
from src.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description='評估預測結果並匯出數據')
    parser.add_argument('--config', default='configs/v0.yaml', help='設定檔路徑')
    parser.add_argument('--input', required=True, help='含 reference、prediction 欄位的 CSV')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值',
                        help='覆蓋設定，可重複使用多次')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    # 沒有特別指定名稱時，自動用「輸入檔名 + 改了哪些參數」命名
    if 'experiment_name' not in changes:
        name = 'eval_' + os.path.splitext(os.path.basename(args.input))[0]
        name += ''.join(f"_{k.split('.')[-1]}-{v}" for k, (_, v) in changes.items())
        cfg['experiment_name'] = name

    with open(args.input, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    refs = [r['reference'] for r in rows]
    hyps = [r['prediction'] for r in rows]
    ids = [r.get('id', i) for i, r in enumerate(rows)]

    with Experiment(cfg, changes) as exp:
        result = exp.save_predictions(refs, hyps, ids)
        exp.finish({
            'wer': result['wer'],
            'S': result['S'], 'D': result['D'], 'I': result['I'], 'N': result['N'],
            'n_sentences': result['n_sentences'],
        })

    print(f"\nWER = {result['wer']:.2%}"
          f"（替換 {result['S']}、刪除 {result['D']}、插入 {result['I']}，總字數 {result['N']}）")
    print(f'每句的詳細結果：{os.path.join(exp.dir, "predictions.csv")}')


if __name__ == '__main__':
    main()
