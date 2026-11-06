"""
CharTokenizer 的測試

執行方式（在 repo 根目錄）：
  python -m pytest tests/test_tokenizer.py -v
或沒有安裝 pytest 時：
  python tests/test_tokenizer.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer import CharTokenizer

tok = CharTokenizer()


def test_vocab_size():
    """字母表應有 36 個：3 個特殊符號 + 26 個字母 + 空白 + 6 種標點"""
    assert tok.vocab_size == 36


def test_special_ids():
    """特殊符號的編號：PAD=0、BOS=1、EOS=2"""
    assert (tok.pad_id, tok.bos_id, tok.eos_id) == (0, 1, 2)


def test_encode_example():
    """筆記裡的範例："Hi!" → [1, 10, 11, 33, 2]"""
    assert tok.encode("Hi!").tolist() == [1, 10, 11, 33, 2]


def test_roundtrip():
    """編碼後再解碼，應該得到小寫版本的原句"""
    for s in ["Hi!", "I want some water.", "it's a well-known fact, isn't it?"]:
        assert tok.decode(tok.encode(s)) == s.lower()


def test_unknown_chars_become_space():
    """不認得的字元（例如數字）會變成空白"""
    assert tok.decode(tok.encode("Room 101")) == "room    "


def test_decode_stops_at_eos():
    """遇到 EOS 就停，後面的補齊（0）和其他內容都忽略"""
    assert tok.decode([1, 10, 11, 2, 0, 0]) == "hi"
    assert tok.decode([1, 10, 2, 11, 12]) == "h"


def test_decode_accepts_list_and_tensor():
    """decode 要能接受 list 和張量"""
    ids = tok.encode("ok")
    assert tok.decode(ids) == tok.decode(ids.tolist()) == "ok"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n全部 {len(tests)} 項測試通過")
