"""
實驗五：訓練
============

用法：
  # 小規模試跑：只用 100 句訓練 2 輪（確認流程沒問題、記憶體夠）
  python scripts/train.py --set train.max_trials=100 --set train.num_epochs=2 \
      --set experiment_name=smoke_test

  # 過擬合測試：10 句反覆訓練，損失應該降到接近 0
  python scripts/train.py --set train.max_trials=10 --set train.num_epochs=200 \
      --set train.patience=1000 --set augment.enabled=false --set experiment_name=overfit_10

  # 完整的 v0 訓練（需要 GPU）
  python scripts/train.py

  # 用你自己寫的 Transformer 訓練
  python scripts/train.py --set model.transformer_impl=mine

輸出（runs/日期-時間_實驗名稱/）：
  metrics.csv      每一輪的訓練損失、驗證損失、學習率
  best_model.pt    驗證損失最低時的模型
  parameters.csv   各模組參數量
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import BrainToTextDataset, collate_fn, find_split_files
from src.experiment import Experiment
from src.model.dsdnla import build_model, count_parameters
from src.tokenizer import CharTokenizer
from src.train import Trainer


def main():
    parser = argparse.ArgumentParser(description='訓練模型')
    parser.add_argument('--config', default='configs/v0.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='區塊.參數=數值')
    args = parser.parse_args()

    cfg, changes = load_config(args.config, args.set)
    t = cfg['train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Experiment(cfg, changes) as exp:
        print(f'Device: {device}')
        tokenizer = CharTokenizer()

        data_dir = cfg['paths']['data_dir']
        train_files = find_split_files(data_dir, 'train')
        val_files = find_split_files(data_dir, 'val')
        if not train_files:
            raise FileNotFoundError(f'在 {data_dir} 找不到訓練資料，請先跑 scripts/check_data.py')
        print(f'Found {len(train_files)} train files, {len(val_files)} val files')

        max_trials = t.get('max_trials')
        train_dataset = BrainToTextDataset(train_files, tokenizer, mode='train',
                                           augment=cfg['augment']['enabled'],
                                           augment_cfg=cfg['augment'], max_trials=max_trials)
        val_dataset = BrainToTextDataset(val_files, tokenizer, mode='val', augment=False,
                                         max_trials=max_trials)

        train_loader = DataLoader(train_dataset, batch_size=t['batch_size'], shuffle=True,
                                  collate_fn=collate_fn, num_workers=t['num_workers'],
                                  pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=t['batch_size'], shuffle=False,
                                collate_fn=collate_fn, num_workers=t['num_workers'],
                                pin_memory=(device.type == 'cuda'))

        model = build_model(cfg, tokenizer.vocab_size)
        params = count_parameters(model)
        exp.save_artifact('parameters.csv', params)
        print(f"Model parameters: {params[-1]['params']:,}")

        trainer = Trainer(model, cfg, device, train_loader, experiment=exp)
        print(f'每輪 {len(train_loader)} 步；學習率排程總步數 {trainer.total_steps}')
        result = trainer.train(train_loader, val_loader)
        result.update({
            'n_params': params[-1]['params'],
            'train_trials': len(train_dataset),
            'val_trials': len(val_dataset),
            'steps_per_epoch': len(train_loader),
            'device': device.type,
        })
        exp.finish(result)


if __name__ == '__main__':
    main()
