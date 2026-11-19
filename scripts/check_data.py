"""
實驗四：資料檢查
================

下載資料後第一個要跑的程式。確認資料放對位置，並統計基本數字：
  - 各切分的檔案數與句數（原 notebook：訓練 8,072、驗證 1,426、測試 1,450）
  - 腦訊號長度（時間點數）的分布
  - 512 個特徵的平均值和標準差（判斷資料有沒有正規化過，待確認事項 Q2）
  - 句子的字元長度分布，以及有多少句超過集束搜尋的長度上限（問題 P10）

用法：
  python scripts/check_data.py
  python scripts/check_data.py --set paths.data_dir=其他路徑
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import h5py
import numpy as np

from src.config import load_config
from src.dataset import find_split_files, read_sentence_label
from src.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description='檢查資料')
    parser.add_argument('--config', default='configs/v0.yaml')
    parser.add_argument('--feature-sample', type=int, default=500,
                        help='抽幾句來統計特徵的平均值和標準差')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    cfg['experiment_name'] = 'check_data'
    data_dir = cfg['paths']['data_dir']
    if not os.path.isdir(data_dir):
        sys.exit(f'找不到資料夾：{data_dir}\n請先下載資料並解壓縮，或用 --set paths.data_dir=... 指定位置')

    beam_limit = cfg['inference']['beam_max_len'] - 1   # 扣掉 <BOS>，最多可以寫幾個字元

    with Experiment(cfg, changes) as exp:
        sessions = sorted(d for d in os.listdir(data_dir) if d.startswith('t15.'))
        print(f'日期資料夾：{len(sessions)} 個')

        result = {'n_sessions': len(sessions)}
        per_split = []
        feature_buf = []

        for split in ['train', 'val', 'test']:
            files = find_split_files(data_dir, split)
            n_trials, lengths, char_lens = 0, [], []
            for path in files:
                with h5py.File(path, 'r') as f:
                    for key in f.keys():
                        trial = f[key]
                        feats = trial['input_features']
                        n_trials += 1
                        lengths.append(feats.shape[0])
                        if split == 'train' and len(feature_buf) < args.feature_sample:
                            feature_buf.append(feats[:])
                        text = read_sentence_label(trial)
                        if text is not None:
                            char_lens.append(len(text))

            row = {
                'split': split, 'files': len(files), 'trials': n_trials,
                'T_min': min(lengths) if lengths else '', 'T_median': int(np.median(lengths)) if lengths else '',
                'T_max': max(lengths) if lengths else '',
                'seconds_median': round(np.median(lengths) * 0.02, 2) if lengths else '',
            }
            if char_lens:
                row.update({
                    'chars_median': int(np.median(char_lens)), 'chars_max': max(char_lens),
                    f'chars_over_{beam_limit}': sum(c > beam_limit for c in char_lens),
                })
            per_split.append(row)
            print(row)
            result[f'{split}_trials'] = n_trials

        exp.save_artifact('splits.csv', per_split)

        if feature_buf:
            allf = np.concatenate(feature_buf, axis=0)        # (總時間點數, 512)
            mean, std = allf.mean(axis=0), allf.std(axis=0)
            exp.save_artifact('feature_stats.csv', [
                {'feature': i, 'mean': float(mean[i]), 'std': float(std[i])} for i in range(allf.shape[1])
            ])
            result['feature_mean_avg'] = float(mean.mean())
            result['feature_std_avg'] = float(std.mean())
            print(f'特徵平均值的平均：{mean.mean():.4f}，標準差的平均：{std.mean():.4f}')
            print('（平均接近 0、標準差接近 1，代表資料已經正規化過）')

        exp.finish(result)


if __name__ == '__main__':
    main()
