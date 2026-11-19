"""
讀資料與資料增強
================

對應原 notebook：Cell 1 的 BrainToTextDataset、collate_fn
拆解筆記：4.2、4.3

每一筆資料讀出：
  neural：(T, 512) 腦訊號
  tokens：(S,) 句子編碼後的數字；測試集沒有答案時是 None
  text：  原始句子；測試集是 None

[新增] 的部分（不改變原本的行為）：
  - 資料增強的機率和強度改成從設定檔讀取（預設值和原 notebook 相同）
  - max_trials：只讀前幾筆，用來小規模試跑
  - find_split_files：找出各切分的檔案
"""

import glob
import os

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm


def find_split_files(data_dir, split):
    """[新增] 找出某個切分（train / val / test）的所有 HDF5 檔，依日期排序。"""
    return sorted(glob.glob(os.path.join(data_dir, 't15.*', f'data_{split}.hdf5')))


def read_sentence_label(trial):
    """[新增] 讀取句子文字；有些 h5py 版本會回傳 bytes，統一轉成字串。"""
    if 'sentence_label' not in trial.attrs:
        return None
    text = trial.attrs['sentence_label']
    return text.decode('utf-8') if isinstance(text, bytes) else str(text)


class BrainToTextDataset(torch.utils.data.Dataset):

    def __init__(self, hdf5_paths, tokenizer, mode='train', augment=True,
                 augment_cfg=None, max_trials=None):
        self.hdf5_paths = hdf5_paths
        self.tokenizer = tokenizer
        self.mode = mode
        self.augment = augment and (mode == 'train')
        self.aug = augment_cfg or {}

        # [原作者] 一次打開所有檔案並保存起來
        # 問題 P16：搭配 num_workers > 0 時，偶爾會出現讀檔錯誤；遇到時把 num_workers 設成 0
        self.trial_keys = []
        self.open_files = {}
        print(f'Loading {mode} data...')
        for i, h5_path in enumerate(tqdm(hdf5_paths)):
            f = h5py.File(h5_path, 'r')
            self.open_files[i] = f
            for key in f.keys():
                self.trial_keys.append((i, key))

        if max_trials is not None:                     # [新增] 小規模試跑用
            self.trial_keys = self.trial_keys[:max_trials]
        print(f'Loaded {len(self.trial_keys)} trials')

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        file_idx, key = self.trial_keys[idx]
        trial = self.open_files[file_idx][key]

        neural = torch.tensor(trial['input_features'][:], dtype=torch.float32)   # (T, 512)

        # [原作者] 讀句子文字，不讀音素標籤，也不讀日期（問題 P6）
        text = read_sentence_label(trial)
        tokens = self.tokenizer.encode(text) if text is not None else None

        if self.augment:
            neural = self.apply_augment(neural)

        return {'neural': neural, 'tokens': tokens, 'text': text}

    def apply_augment(self, x):
        """
        [原作者] 三種資料增強（數值改成從設定檔讀，預設和原本相同）
          ① 時間遮罩：30% 機率，把連續 5–14 個時間點設成 0
          ② 電極關閉：20% 機率，隨機關掉約 10% 的通道（沒有放大補償）
          ③ 高斯雜訊：25% 機率，加上標準差 0.05 的雜訊（問題 P15：官方是 1.0）
        """
        a = self.aug
        if torch.rand(1) < a.get('time_mask_prob', 0.3):
            mask_len = torch.randint(a.get('time_mask_min', 5), a.get('time_mask_max', 15), (1,)).item()
            start = torch.randint(0, max(1, x.size(0) - mask_len), (1,)).item()
            x[start:start + mask_len] = 0

        if torch.rand(1) < a.get('channel_drop_prob', 0.2):
            drop_mask = torch.bernoulli(torch.ones(x.size(1)) * a.get('channel_keep_ratio', 0.9))
            x = x * drop_mask

        if torch.rand(1) < a.get('noise_prob', 0.25):
            x = x + torch.randn_like(x) * a.get('noise_std', 0.05)

        return x


def collate_fn(batch):
    """
    [原作者] 把長短不一的句子補齊成同樣長度
      neural：補 0.0 → (B, T_max, 512)
      tokens：補 0（PAD）→ (B, S_max)

    問題 P5：只補齊、沒有回傳真實長度，所以模型不知道哪些位置是補上去的
    """
    neurals = [item['neural'] for item in batch]
    tokens_list = [item['tokens'] for item in batch if item['tokens'] is not None]
    texts = [item['text'] for item in batch if item['text'] is not None]

    neural_padded = nn.utils.rnn.pad_sequence(neurals, batch_first=True, padding_value=0.0)
    tokens_padded = (nn.utils.rnn.pad_sequence(tokens_list, batch_first=True, padding_value=0)
                     if tokens_list else None)

    return {'neural': neural_padded, 'tokens': tokens_padded, 'texts': texts}
