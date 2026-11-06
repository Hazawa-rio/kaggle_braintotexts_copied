"""
字母級編碼器：句子 ⇄ 數字
==============================

對應原 notebook：Cell 1 的 CharTokenizer
（Cell 2 有一份幾乎相同的重複定義，這裡合併成一份）

用途
----
- 訓練時：把正確句子轉成數字，交給解碼器學習（encode）
- 推論時：把解碼器產生的數字轉回句子（decode）

字母表（共 36 個）
-----------------
  編號    符號        說明
  0       <PAD>       補齊用：同一批句子長短不同，短的後面補 0
  1       <BOS>       句子開始（Begin Of Sequence）
  2       <EOS>       句子結束（End Of Sequence）
  3–28    a–z         26 個小寫字母
  29      ' '         空白（字與字之間）
  30–35   ' . , ! ? -  6 種標點

範例
----
  "Hi!"  →  轉小寫 "hi!"  →  [1, 10, 11, 33, 2]
                              BOS  h   i   !  EOS

標記說明
--------
  [原作者]  原 notebook 的寫法，v0 保留不改
  [新增]    為了搬到 repo 而加上的部分，不改變原本的行為
  P#        對應拆解筆記第 5 節「問題總表」的編號
"""

import string

import torch


class CharTokenizer:
    """字母級編碼器，把句子和數字互相轉換。"""

    def __init__(self):
        # [原作者] 三個特殊符號的編號
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

        # [原作者] 建立字母表：特殊符號 → a–z → 空白 → 標點
        self.chars = ['<PAD>', '<BOS>', '<EOS>']
        self.chars += list(string.ascii_lowercase)   # a–z，編號 3–28
        self.chars += [' ']                          # 空白，編號 29
        self.chars += list("'.,!?-")                 # 標點，編號 30–35

        # [原作者] 雙向對照表
        self.char2id = {c: i for i, c in enumerate(self.chars)}   # 'a' → 3
        self.id2char = {i: c for i, c in enumerate(self.chars)}   # 3 → 'a'
        self.vocab_size = len(self.chars)                          # 36

    def encode(self, text):
        """
        句子 → 數字

        輸入：字串，例如 "Hi!"
        輸出：(S,) 的 long 張量，例如 tensor([1, 10, 11, 33, 2])
              S = 字元數 + 2（前後各加一個 BOS、EOS）
        """
        text = text.lower()                    # [原作者] 全部轉小寫，所以字母表不需要大寫
        ids = [self.bos_id]                    # [原作者] 開頭放 <BOS>
        for c in text:
            if c in self.char2id:
                ids.append(self.char2id[c])
            else:
                # [原作者] 不認得的字元一律變成空白
                # 注意：數字 0–9、引號 "、冒號 : 等都會變成空白
                ids.append(self.char2id[' '])
        ids.append(self.eos_id)                # [原作者] 結尾放 <EOS>
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        """
        數字 → 句子

        輸入：一串數字，可以是 list、torch 張量或 numpy 陣列
              例如 [1, 10, 11, 33, 2, 0, 0]
        輸出：字串，例如 "hi!"

        規則：
          - 遇到 <EOS> 就停止，後面的內容（通常是補齊的 0）全部忽略
          - 跳過 <PAD>、<BOS>、<EOS>（編號 0、1、2）
        """
        # [新增] 統一轉成 Python 的 int 清單
        #   原 notebook 有兩種寫法：
        #   Cell 1 用 i.item()（假設輸入是張量），Cell 2 先轉成 numpy
        #   這裡合併成一種，三種輸入都能處理
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [int(i) for i in ids]

        chars = []
        for i in ids:
            if i == self.eos_id:               # [原作者] 遇到 <EOS> 就停
                break
            if i > 2:                          # [原作者] 跳過 PAD、BOS、EOS
                # [原作者，取自 Cell 2] 萬一出現超出範圍的編號，當成空白
                chars.append(self.id2char.get(i, ' '))
        return ''.join(chars)


# [新增] 直接執行這個檔案時，示範一次編碼和解碼
if __name__ == "__main__":
    tok = CharTokenizer()
    print("字母表大小：", tok.vocab_size)

    for s in ["Hi!", "I want some water.", "Room 101"]:
        ids = tok.encode(s)
        print(f"\n原句：  {s!r}")
        print(f"編碼：  {ids.tolist()}")
        print(f"解碼：  {tok.decode(ids)!r}")
