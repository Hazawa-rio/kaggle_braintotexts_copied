"""
WER 與錯誤拆解的測試

執行方式（在 repo 根目錄）：
  python -m pytest tests/test_metrics.py -v
或沒有安裝 pytest 時：
  python tests/test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import count_errors, normalize_text, wer


def test_substitution():
    r = count_errors("i want some water", "i want sum water")
    assert (r['S'], r['D'], r['I'], r['N']) == (1, 0, 0, 4)


def test_deletion():
    r = count_errors("how are you", "how you")
    assert (r['S'], r['D'], r['I'], r['N']) == (0, 1, 0, 3)


def test_insertion():
    r = count_errors("thank you", "thank you very much")
    assert (r['S'], r['D'], r['I'], r['N']) == (0, 0, 2, 2)


def test_perfect_match():
    r = count_errors("hello world", "hello world")
    assert r['errors'] == 0


def test_aggregate_wer():
    """整體加總後再相除：(1 + 1 + 2) ÷ (4 + 3 + 2) = 4 ÷ 9"""
    result = wer(
        ["I want some water", "How are you?", "Thank you."],
        ["i want sum water", "how you", "thank you very much"],
    )
    assert (result['S'], result['D'], result['I'], result['N']) == (1, 1, 2, 9)
    assert abs(result['wer'] - 4 / 9) < 1e-9


def test_not_average_of_sentences():
    """確認不是「每句平均」：每句平均會是 (25% + 33.3% + 100%) ÷ 3 ≈ 52.8%"""
    result = wer(["i want some water", "how are you", "thank you"],
                 ["i want sum water", "how you", "thank you very much"])
    assert abs(result['wer'] - 0.528) > 0.05


def test_wer_can_exceed_100_percent():
    """插入也算錯，所以 WER 可以超過 100%"""
    result = wer(["hi"], ["oh hi there my friend"])
    assert result['wer'] > 1.0


def test_normalize_matches_official_rules():
    """和官方 remove_punctuation() 的行為一致"""
    assert normalize_text("I'm fine, thanks!") == "i'm fine thanks"
    assert normalize_text("Room 101") == "room"
    assert normalize_text("  Hello   World  ") == "hello world"


def test_case_and_punctuation_ignored():
    """大小寫和標點不影響分數"""
    r = count_errors("Hello, World!", "hello world")
    assert r['errors'] == 0


def test_mismatched_length_raises():
    try:
        wer(["a", "b"], ["a"])
    except ValueError:
        return
    raise AssertionError("句數不同時應該報錯")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n全部 {len(tests)} 項測試通過")
