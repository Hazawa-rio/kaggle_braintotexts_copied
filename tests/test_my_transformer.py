"""
驗收你自己寫的 Transformer（src/model/my_transformer.py）
========================================================

執行方式（在 repo 根目錄）：
  python tests/test_my_transformer.py

每一題的檢查方式都一樣：
  1. 建立 PyTorch 官方版本和你的版本
  2. 把官方版本的權重「原封不動」複製給你的版本（名稱和形狀必須完全相同）
  3. 關掉 dropout，餵入同樣的隨機資料
  4. 比對輸出，誤差在 1e-4 以內就算通過

還沒寫的題目會顯示「尚未完成」，不會中斷其他題目的檢查。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import my_transformer as mt

ATOL = 1e-4
B, L, S, D, H = 2, 5, 7, 32, 4      # 用小尺寸測試，跑得比較快


class Pending(Exception):
    """題目還沒寫完"""


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        raise Pending(str(e))


def _same(a, b, what):
    a, b = a.detach(), b.detach()
    assert a.shape == b.shape, f'{what}：形狀不同，你的是 {tuple(a.shape)}，官方是 {tuple(b.shape)}'
    diff = (a - b).abs().max().item()
    assert diff < ATOL, f'{what}：數值不同，最大誤差 {diff:.2e}'


def _causal(n):
    return nn.Transformer.generate_square_subsequent_mask(n)     # 浮點遮罩，上三角是 -inf


def _padding_mask(lengths, total):
    """(B, total) 的布林值，超過真實長度的位置是 True"""
    return torch.arange(total)[None, :] >= torch.tensor(lengths)[:, None]


# -----------------------------------------------------------------------------
# 練習 1
# -----------------------------------------------------------------------------

def test_1_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(B, L, 8), torch.randn(B, S, 8), torch.randn(B, S, 8)

    out, w = _call(mt.scaled_dot_product_attention, q, k, v)
    _same(out, F.scaled_dot_product_attention(q, k, v), '沒有遮罩的輸出')
    assert w.shape == (B, L, S), f'權重形狀應為 {(B, L, S)}，你的是 {tuple(w.shape)}'
    _same(w.sum(-1), torch.ones(B, L), '權重每一列的加總（應為 1）')

    # 布林遮罩：True = 不能看
    x = torch.randn(B, L, 8)
    bool_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    out_b, w_b = _call(mt.scaled_dot_product_attention, x, x, x, attn_mask=bool_mask)
    ref = F.scaled_dot_product_attention(x, x, x, is_causal=True)
    _same(out_b, ref, '布林因果遮罩的輸出')
    assert torch.all(w_b[:, 0, 1:] == 0), '第一個位置不該看到後面'

    # 浮點遮罩：直接加到分數上
    out_f, _ = _call(mt.scaled_dot_product_attention, x, x, x, attn_mask=_causal(L))
    _same(out_f, ref, '浮點因果遮罩的輸出')

    # 多一個「頭」的維度 (B, H, L, d) 也要能處理
    q4, k4, v4 = torch.randn(B, H, L, 8), torch.randn(B, H, S, 8), torch.randn(B, H, S, 8)
    out4, _ = _call(mt.scaled_dot_product_attention, q4, k4, v4)
    _same(out4, F.scaled_dot_product_attention(q4, k4, v4), '四維輸入 (B, H, L, d) 的輸出')


# -----------------------------------------------------------------------------
# 練習 2
# -----------------------------------------------------------------------------

def test_2_multihead_attention():
    torch.manual_seed(0)
    ref = nn.MultiheadAttention(D, H, dropout=0.0, batch_first=True).eval()
    mine = mt.MyMultiheadAttention(D, H, dropout=0.0, batch_first=True).eval()
    mine.load_state_dict(ref.state_dict())          # 名稱和形狀要完全一樣才載入得進去

    q, kv = torch.randn(B, L, D), torch.randn(B, S, D)
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        # 沒有遮罩（交叉注意力的情況：query 和 key/value 長度不同）
        out, _ = _call(mine, q, kv, kv)
        _same(out, ref(q, kv, kv, need_weights=False)[0], '沒有遮罩')

        # 因果遮罩（自注意力）
        out, _ = _call(mine, q, q, q, attn_mask=_causal(L))
        _same(out, ref(q, q, q, attn_mask=_causal(L), need_weights=False)[0], '因果遮罩')

        # 補齊遮罩：第 2 句只有前 4 個位置是真的
        kpm = _padding_mask([S, 4], S)
        out, _ = _call(mine, q, kv, kv, key_padding_mask=kpm)
        _same(out, ref(q, kv, kv, key_padding_mask=kpm, need_weights=False)[0], '補齊遮罩')

        # 兩種遮罩同時使用
        kpm2 = _padding_mask([L, 3], L)
        out, _ = _call(mine, q, q, q, attn_mask=_causal(L), key_padding_mask=kpm2)
        expect = ref(q, q, q, attn_mask=_causal(L), key_padding_mask=kpm2, need_weights=False)[0]
        _same(out, expect, '因果遮罩 + 補齊遮罩')


# -----------------------------------------------------------------------------
# 練習 3、4：編碼器
# -----------------------------------------------------------------------------

def _encoder_layer_args():
    return dict(d_model=D, nhead=H, dim_feedforward=4 * D, dropout=0.0,
                activation='gelu', batch_first=True, norm_first=True)


def test_3_encoder_layer():
    torch.manual_seed(0)
    ref = nn.TransformerEncoderLayer(**_encoder_layer_args()).eval()
    mine = mt.MyTransformerEncoderLayer(**_encoder_layer_args()).eval()
    mine.load_state_dict(ref.state_dict())

    x = torch.randn(B, L, D)
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        _same(_call(mine, x), ref(x), '編碼器層')
        kpm = _padding_mask([L, 3], L)
        _same(_call(mine, x, src_key_padding_mask=kpm), ref(x, src_key_padding_mask=kpm),
              '編碼器層 + 補齊遮罩')


def test_4_encoder_stack():
    torch.manual_seed(0)
    ref = nn.TransformerEncoder(nn.TransformerEncoderLayer(**_encoder_layer_args()),
                                num_layers=3, enable_nested_tensor=False).eval()
    mine = mt.MyTransformerEncoder(mt.MyTransformerEncoderLayer(**_encoder_layer_args()),
                                   num_layers=3).eval()
    mine.load_state_dict(ref.state_dict())

    x = torch.randn(B, L, D)
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        _same(_call(mine, x), ref(x), '3 層編碼器')


# -----------------------------------------------------------------------------
# 練習 5、6：解碼器
# -----------------------------------------------------------------------------

def test_5_decoder_layer():
    torch.manual_seed(0)
    ref = nn.TransformerDecoderLayer(**_encoder_layer_args()).eval()
    mine = mt.MyTransformerDecoderLayer(**_encoder_layer_args()).eval()
    mine.load_state_dict(ref.state_dict())

    tgt, memory = torch.randn(B, L, D), torch.randn(B, S, D)
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        _same(_call(mine, tgt, memory, tgt_mask=_causal(L)),
              ref(tgt, memory, tgt_mask=_causal(L)), '解碼器層（因果遮罩）')
        mkpm = _padding_mask([S, 4], S)
        _same(_call(mine, tgt, memory, tgt_mask=_causal(L), memory_key_padding_mask=mkpm),
              ref(tgt, memory, tgt_mask=_causal(L), memory_key_padding_mask=mkpm),
              '解碼器層（因果遮罩 + 腦訊號補齊遮罩）')


def test_6_decoder_stack():
    torch.manual_seed(0)
    ref = nn.TransformerDecoder(nn.TransformerDecoderLayer(**_encoder_layer_args()),
                                num_layers=3).eval()
    mine = mt.MyTransformerDecoder(mt.MyTransformerDecoderLayer(**_encoder_layer_args()),
                                   num_layers=3).eval()
    mine.load_state_dict(ref.state_dict())

    tgt, memory = torch.randn(B, L, D), torch.randn(B, S, D)
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        _same(_call(mine, tgt, memory, tgt_mask=_causal(L)),
              ref(tgt, memory, tgt_mask=_causal(L)), '3 層解碼器')


# -----------------------------------------------------------------------------
# 最終關卡：整個案例 B 模型
# -----------------------------------------------------------------------------

def test_7_full_model():
    """
    用官方版本建立完整的案例 B 模型，把權重整個搬進你的版本，輸出要一樣。
    通過之後，作者訓練好的權重也能直接載入你的版本：
      python scripts/infer.py --split val --limit 10 --set model.transformer_impl=mine
    """
    from src.config import load_config
    from src.model.dsdnla import build_model
    from src.tokenizer import CharTokenizer

    config = os.path.join(os.path.dirname(__file__), '..', 'configs', 'v0.yaml')
    cfg_ref, _ = load_config(config, ['model.transformer_impl=pytorch', 'model.dropout=0.0'])
    cfg_mine, _ = load_config(config, ['model.transformer_impl=mine', 'model.dropout=0.0'])
    vocab = CharTokenizer().vocab_size

    torch.manual_seed(0)
    ref = build_model(cfg_ref, vocab).eval()
    mine = build_model(cfg_mine, vocab).eval()
    mine.load_state_dict(ref.state_dict())          # 嚴格模式：一個參數都不能少

    x = torch.randn(2, 60, 512)
    tokens = torch.randint(3, vocab, (2, 12))
    with torch.enable_grad():   # 讓官方版本走一般的計算路徑（不用推論加速），方便逐位比對
        out_ref, _ = ref(x, target_tokens=tokens, training=False)
        out_mine, _ = _call(mine, x, target_tokens=tokens, training=False)
    _same(out_mine, out_ref, '完整模型的輸出')


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        name = t.__name__.replace('test_', '練習 ', 1).replace('_', ' ', 1)
        try:
            t()
            print(f'✅ 通過    {name}')
            passed += 1
        except Pending:
            print(f'⏳ 尚未完成 {name}')
        except AssertionError as e:
            print(f'❌ 未通過  {name}\n           {e}')
        except Exception as e:
            print(f'💥 執行錯誤 {name}\n           {type(e).__name__}: {e}')
    print(f'\n{passed} / {len(tests)} 題通過')
