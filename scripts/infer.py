"""
實驗六：推論與評分
==================

用訓練好的模型產生句子。
  --split val ：驗證集有正確答案，會自動計算 WER 並拆解三種錯誤
  --split test：測試集沒有答案，會產生 Kaggle 的提交檔 submission.csv

用法：
  # 先用驗證集前 10 句試跑
  python scripts/infer.py --split val --limit 10

  # 用作者的模型，評估整個驗證集（v0 的第一個真實成績）
  python scripts/infer.py --split val

  # 調整推論設定，例如關掉拼字檢查、只保留 5 個候選
  python scripts/infer.py --split val --set inference.use_spellcheck=false --set inference.beam_width=5

  # 用自己訓練的模型
  python scripts/infer.py --split val --checkpoint runs/某次實驗/best_model.pt

  # 產生提交檔
  python scripts/infer.py --split test

輸出（runs/日期-時間_實驗名稱/）：
  predictions.csv   每一句的正確答案、預測、錯誤拆解（val）
  details.csv       每一句的候選句、後處理前的句子（方便分析 P12、P13）
  submission.csv    Kaggle 提交檔（test）
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import load_config
from src.experiment import Experiment
from src.infer import run_inference


def main():
    parser = argparse.ArgumentParser(description='推論與評分')
    parser.add_argument('--config', default='configs/v0.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--checkpoint', default=None, help='預設用設定檔的 paths.checkpoint')
    parser.add_argument('--limit', type=int, default=None, help='只跑前幾句（試跑用）')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    checkpoint = args.checkpoint or cfg['paths']['checkpoint']
    if not os.path.exists(checkpoint):
        sys.exit(f'找不到模型檔：{checkpoint}')
    if 'experiment_name' not in changes:
        cfg['experiment_name'] = f'infer_{args.split}' + ''.join(
            f"_{k.split('.')[-1]}-{v}" for k, (_, v) in changes.items())

    with Experiment(cfg, changes) as exp:
        exp.meta['checkpoint'] = checkpoint
        results = run_inference(cfg, checkpoint, split=args.split, limit=args.limit)
        exp.save_artifact('details.csv', results)

        summary = {'n_sentences': len(results), 'checkpoint': checkpoint, 'split': args.split}
        if args.split == 'val':
            r = exp.save_predictions([x['reference'] for x in results],
                                     [x['prediction'] for x in results],
                                     [x['id'] for x in results])
            summary.update({'wer': r['wer'], 'S': r['S'], 'D': r['D'], 'I': r['I'], 'N': r['N']})
            print(f"\nWER = {r['wer']:.2%}（替換 {r['S']}、刪除 {r['D']}、插入 {r['I']}，總字數 {r['N']}）")
        else:
            # 提交檔用一般的 utf-8（不加 BOM），避免 Kaggle 讀錯第一個欄位名稱
            path = os.path.join(exp.dir, 'submission.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'text'])
                writer.writerows([x['id'], x['prediction']] for x in results)
            print(f'\n提交檔：{path}')

        print('\n前 10 筆預測：')
        for x in results[:10]:
            print(f"  {x['id']:>4}  {x['prediction']}")
        exp.finish(summary)


if __name__ == '__main__':
    main()
