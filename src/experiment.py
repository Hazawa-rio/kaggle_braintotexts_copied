"""
實驗紀錄與數據匯出
==================

[新增] 原 notebook 沒有這個功能。每次執行實驗時，自動做這幾件事：

  1. 建立一個專屬資料夾：runs/20261110-153012_v0_baseline/
  2. 存下這次「實際使用的完整設定」          → config.yaml
  3. 記錄執行環境（時間、Git 版本、改了哪些參數） → meta.json
  4. 記錄過程中的數據（每一輪的損失、WER）    → metrics.csv
  5. 記錄每一句的預測結果和錯誤拆解          → predictions.csv
  6. 記錄最終成績                          → summary.json
  7. 在總表新增一列，方便比較所有實驗        → experiments/all_runs.csv

所有 CSV 都用 utf-8-sig 編碼，用 Excel 直接打開中文也不會亂碼。

用法
----
  from src.config import load_config
  from src.experiment import Experiment

  cfg, changes = load_config('configs/v0.yaml', overrides=[...])
  with Experiment(cfg, changes) as exp:
      for epoch in range(...):
          ...
          exp.log_metrics({'epoch': epoch, 'train_loss': ..., 'val_loss': ...})
      result = exp.save_predictions(refs, hyps)
      exp.finish({'val_wer': result['wer']})

  用 with 包起來的好處：中途出錯時，也會把「失敗」記進總表，不會漏掉紀錄。
"""

import csv
import json
import os
import platform
import random
import subprocess
from datetime import datetime

from src.config import flatten, save_config
from src.metrics import count_errors


def set_seed(seed):
    """
    固定所有隨機來源，讓實驗可以重現（問題 P17）。
    沒有安裝 numpy 或 torch 時會自動跳過。
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_git_commit():
    """取得目前程式碼的 Git 版本（短雜湊），方便日後找回當時的程式碼。"""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or 'unknown'
        # 有改動還沒提交的話，加上標記提醒自己
        # （實驗總表每次都會變動，不算在內）
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.',
                                ':(exclude)experiments/all_runs.csv'],
                               capture_output=True, text=True, timeout=5)
        return commit + ('-dirty' if dirty.stdout.strip() else '')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _write_csv(path, rows):
    """把字典清單寫成 CSV。欄位取所有列的聯集，保留第一次出現的順序。"""
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class Experiment:
    """一次實驗的紀錄器。"""

    def __init__(self, cfg, changes=None):
        self.cfg = cfg
        self.changes = changes or {}
        self.start_time = datetime.now()
        self.metrics_rows = []
        self.final = {}
        self.status = 'running'

        # 1. 建立專屬資料夾：時間 + 實驗名稱
        stamp = self.start_time.strftime('%Y%m%d-%H%M%S')
        self.run_id = f"{stamp}_{cfg['experiment_name']}"
        self.dir = os.path.join(cfg['paths']['runs_dir'], self.run_id)
        n = 2
        while os.path.exists(self.dir):          # 同一秒內跑兩次時，加上編號避免覆蓋
            self.dir = os.path.join(cfg['paths']['runs_dir'], f'{self.run_id}_{n}')
            n += 1
        self.run_id = os.path.basename(self.dir)
        os.makedirs(self.dir)

        # 2. 存下實際使用的完整設定
        save_config(cfg, os.path.join(self.dir, 'config.yaml'))

        # 3. 記錄執行環境
        self.meta = {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(timespec='seconds'),
            'git_commit': get_git_commit(),
            'python': platform.python_version(),
            'machine': platform.platform(),
            'changes': {k: {'from': old, 'to': new}
                        for k, (old, new) in self.changes.items()},
        }
        self._dump_json('meta.json', self.meta)

        set_seed(cfg.get('seed', 42))
        print(f'[實驗] 開始：{self.run_id}')
        print(f'[實驗] 輸出資料夾：{self.dir}')
        for k, (old, new) in self.changes.items():
            print(f'[實驗] 參數變更：{k} = {old} → {new}')

    # ------------------------------------------------------------------
    # 記錄
    # ------------------------------------------------------------------

    def log_metrics(self, row):
        """
        記錄一筆過程數據（通常是每一輪一筆），並立刻寫入 metrics.csv。
        每次都重寫整個檔案，所以就算中途當機，已記錄的數據也不會遺失。

        例如：exp.log_metrics({'epoch': 3, 'train_loss': 1.23, 'val_loss': 1.45})
        """
        self.metrics_rows.append(dict(row))
        _write_csv(os.path.join(self.dir, 'metrics.csv'), self.metrics_rows)
        print('[實驗] ' + '  '.join(f'{k}={_fmt(v)}' for k, v in row.items()))

    def save_predictions(self, refs, hyps, ids=None):
        """
        計算每一句的錯誤，存成 predictions.csv，並回傳整體結果。

        輸入：
          refs：正確句清單
          hyps：預測句清單
          ids： 每句的編號（不給就用 0, 1, 2, ...）

        輸出：字典 {'wer', 'S', 'D', 'I', 'N', 'n_sentences'}
        """
        if len(refs) != len(hyps):
            raise ValueError(f'句數不同：正確句 {len(refs)}，預測句 {len(hyps)}')
        ids = ids if ids is not None else list(range(len(refs)))
        normalize = self.cfg.get('eval', {}).get('normalize', True)

        rows, total = [], {'S': 0, 'D': 0, 'I': 0, 'N': 0}
        for i, ref, hyp in zip(ids, refs, hyps):
            r = count_errors(ref, hyp, normalize=normalize)
            for k in total:
                total[k] += r[k]
            rows.append({
                'id': i, 'reference': ref, 'prediction': hyp,
                'S': r['S'], 'D': r['D'], 'I': r['I'], 'N': r['N'],
                'sentence_wer': round(r['errors'] / r['N'], 4) if r['N'] else '',
            })
        _write_csv(os.path.join(self.dir, 'predictions.csv'), rows)

        errors = total['S'] + total['D'] + total['I']
        total['wer'] = errors / total['N'] if total['N'] else float('inf')
        total['n_sentences'] = len(rows)
        return total

    def save_artifact(self, name, rows):
        """存任何其他表格，例如 save_artifact('candidates.csv', [...])。"""
        _write_csv(os.path.join(self.dir, name), rows)

    # ------------------------------------------------------------------
    # 結束
    # ------------------------------------------------------------------

    def finish(self, final_metrics, status='finished'):
        """
        記錄最終成績，並在總表 experiments/all_runs.csv 新增一列。

        例如：exp.finish({'val_wer': 0.312, 'S': 120, 'D': 45, 'I': 30})
        """
        self.final = dict(final_metrics)
        self.status = status
        end = datetime.now()
        summary = {
            **self.meta,
            'end_time': end.isoformat(timespec='seconds'),
            'duration_sec': round((end - self.start_time).total_seconds(), 1),
            'status': status,
            'final_metrics': self.final,
        }
        self._dump_json('summary.json', summary)
        self._append_summary_row(summary)
        print(f'[實驗] 結束（{status}）：' +
              '  '.join(f'{k}={_fmt(v)}' for k, v in self.final.items()))

    def _append_summary_row(self, summary):
        """總表的一列 = 基本資訊 + 最終成績 + 攤平的完整設定。"""
        changes_text = '; '.join(f'{k}={new}' for k, (_, new) in self.changes.items())
        row = {
            'run_id': self.run_id,
            'experiment_name': self.cfg['experiment_name'],
            'status': summary['status'],
            'start_time': summary['start_time'],
            'duration_sec': summary['duration_sec'],
            'git_commit': summary['git_commit'],
            'changes': changes_text or '（無，使用設定檔原值）',
        }
        row.update({f'result.{k}': _fmt(v) for k, v in self.final.items()})
        row.update({f'cfg.{k}': v for k, v in flatten(self.cfg).items()})

        path = self.cfg['paths']['summary_csv']
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        rows = _read_csv(path)
        rows.append(row)
        _write_csv(path, rows)
        print(f'[實驗] 已加入總表：{path}')

    def _dump_json(self, name, obj):
        with open(os.path.join(self.dir, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------
    # 讓 with 語法可以用：出錯時也會記錄
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.status == 'running':
            if exc_type is None:
                self.finish(self.final, status='finished')
            else:
                self.finish({**self.final, 'error': f'{exc_type.__name__}: {exc}'},
                            status='failed')
        return False   # 不吞掉錯誤，讓錯誤訊息照常顯示


def _fmt(v):
    """浮點數統一保留 6 位有效數字，其他原樣。"""
    return float(f'{v:.6g}') if isinstance(v, float) else v
