"""
實驗二：學習率排程（問題 P7）
============================

算出原 notebook 的 OneCycleLR 在整個訓練過程中，學習率實際是怎麼變化的，
並匯出每一步的數值。不需要資料集、不需要 GPU，也不需要安裝 PyTorch。

背景
----
原作者這樣設定排程的總步數：
  total_steps = num_epochs * 1000         ← 假設每輪 1,000 步
但實際每輪的步數是：
  ceil(訓練句數 ÷ batch_size) = ceil(8072 ÷ 16) = 505 步
所以學習率曲線只會走到一半左右。這支程式用數字把這件事呈現出來。

用法（在 repo 根目錄）：
  # v0：原作者的寫法
  python scripts/lr_schedule.py

  # v1：改用實際步數
  python scripts/lr_schedule.py --set train.use_assumed_lr_steps=false \
      --set experiment_name=lr_actual_steps

  # 再試試看：只訓練 10 輪會怎樣
  python scripts/lr_schedule.py --set train.num_epochs=10 --set experiment_name=lr_10epochs

輸出：
  metrics.csv       每一輪開始時的學習率
  lr_per_step.csv   每一步的學習率（可以用 Excel 畫成折線圖）
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import load_config
from src.experiment import Experiment


def one_cycle_lr(step, total_steps, max_lr, pct_start=0.05,
                 div_factor=25.0, final_div_factor=1e4):
    """
    計算 PyTorch OneCycleLR 在第 step 步的學習率（餘弦版本、兩段式）。

    照 PyTorch 原始碼的公式手寫，參數預設值也和 PyTorch 相同；
    原 notebook 只指定了 max_lr、total_steps、pct_start=0.05、anneal_strategy='cos'。

      起始學習率 = max_lr ÷ 25
      最低學習率 = 起始學習率 ÷ 10,000
      第一段（前 5% 步數）：從起始學習率用餘弦曲線升到 max_lr
      第二段（其餘步數）  ：從 max_lr 用餘弦曲線降到最低學習率
    """
    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div_factor
    warmup_end = float(pct_start * total_steps) - 1
    final_end = total_steps - 1

    if step <= warmup_end:
        start, end, pct = initial_lr, max_lr, step / warmup_end
    else:
        start, end = max_lr, min_lr
        pct = (step - warmup_end) / (final_end - warmup_end)
    # 餘弦內插：pct 從 0 到 1 時，數值從 start 平滑變到 end
    return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)


def main():
    parser = argparse.ArgumentParser(description='計算並匯出學習率排程')
    parser.add_argument('--config', default='configs/v0.yaml')
    parser.add_argument('--n-train', type=int, default=8072,
                        help='訓練句數（原 notebook 印出的數字是 8072）')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    if 'experiment_name' not in changes and changes:
        cfg['experiment_name'] = 'lr_' + '_'.join(
            f"{k.split('.')[-1]}-{v}" for k, (_, v) in changes.items())
    elif 'experiment_name' not in changes:
        cfg['experiment_name'] = 'lr_v0_original'

    t = cfg['train']
    steps_per_epoch = math.ceil(args.n_train / t['batch_size'])
    steps_run = steps_per_epoch * t['num_epochs']          # 跑滿所有輪數時實際走的步數
    if t['use_assumed_lr_steps']:
        total_steps = t['num_epochs'] * t['lr_steps_per_epoch_assumed']   # 原作者的寫法
    else:
        total_steps = steps_run                                          # 修正後的寫法
    if steps_run > total_steps:
        raise ValueError('實際步數超過排程總步數，PyTorch 會報錯')

    with Experiment(cfg, changes) as exp:
        per_step = []
        for s in range(steps_run):
            lr = one_cycle_lr(s, total_steps, t['learning_rate'])
            per_step.append({'step': s, 'epoch': s // steps_per_epoch, 'lr': lr})
            if s % steps_per_epoch == 0:
                exp.log_metrics({'epoch': s // steps_per_epoch, 'step': s, 'lr': lr})
        exp.save_artifact('lr_per_step.csv', per_step)

        lrs = [r['lr'] for r in per_step]
        peak_step = lrs.index(max(lrs))
        exp.finish({
            'steps_per_epoch': steps_per_epoch,
            'scheduler_total_steps': total_steps,
            'steps_actually_run': steps_run,
            'schedule_progress_pct': round(100 * steps_run / total_steps, 1),
            'warmup_epochs': round(peak_step / steps_per_epoch, 2),
            'peak_lr': max(lrs),
            'last_lr': lrs[-1],
            'last_lr_vs_peak_pct': round(100 * lrs[-1] / max(lrs), 1),
        })

    print(f'\n每輪 {steps_per_epoch} 步，共跑 {steps_run} 步；排程總步數設定為 {total_steps}')
    print(f'排程走完 {100 * steps_run / total_steps:.1f}%，'
          f'結束時學習率還有最高值的 {100 * lrs[-1] / max(lrs):.1f}%')


if __name__ == '__main__':
    main()
