"""
後處理
======

對應原 notebook：Cell 2 的 normalize_for_eval、clean_generated_text、spell_fix、
                  compute_gpt2_perplexity、select_best_candidate_by_lm
拆解筆記：4.11 ②③④

流程：候選句 → GPT-2 挑最通順的 → 拼字檢查 → 清理 → 正規化

[新增] 的部分：
  - GPT-2 和拼字檢查器改成「第一次用到才載入」，不用的實驗不必下載
  - 原作者寫死 .cuda()，這裡改成自動選擇 GPU 或 CPU
"""

import re

import torch

_gpt2 = {}        # 第一次用到時才載入
_spell = {}


# -----------------------------------------------------------------------------
# 正規化與清理
# -----------------------------------------------------------------------------

def normalize_for_eval(text):
    """
    [原作者] 轉小寫、只留 a–z、0–9、撇號、空白
    問題 P14：和官方規則不同（官方保留連字號、刪除數字）；正式評分以 src/metrics.py 為準
    """
    text = text.lower()
    text = text.replace('’', "'")
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_generated_text(text):
    """[原作者] 連續 3 個以上相同字元縮成 1 個、整理空白和撇號"""
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = ' '.join(text.split())
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()


# -----------------------------------------------------------------------------
# 拼字檢查（問題 P13：不看上下文，可能改出冷僻字）
# -----------------------------------------------------------------------------

def spell_fix(text):
    """[原作者] 把字典裡沒有的字，換成字典裡最接近的字"""
    if 'spell' not in _spell:
        from spellchecker import SpellChecker
        _spell['spell'] = SpellChecker()
    spell = _spell['spell']
    try:
        words = text.split()
        unknown_words = spell.unknown(words)
        corrected = [spell.correction(w) if w in unknown_words else w for w in words]
        text = ' '.join(filter(None, corrected))   # correction 可能回傳 None，過濾掉
    except Exception as e:
        print(f'Warning: Could not apply spell fix: {e}')
    return text


# -----------------------------------------------------------------------------
# GPT-2 挑選（問題 P12：只看通順度，完全沒用到模型自己的分數）
# -----------------------------------------------------------------------------

def _load_gpt2():
    if 'model' not in _gpt2:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading GPT-2 on {device}...')
        _gpt2['tokenizer'] = GPT2TokenizerFast.from_pretrained('gpt2')
        _gpt2['model'] = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        _gpt2['device'] = device
    return _gpt2['model'], _gpt2['tokenizer'], _gpt2['device']


def compute_gpt2_perplexity(text):
    """
    [原作者] 名稱寫 perplexity，實際回傳的是「平均每個 token 的損失」，越低越通順
    """
    if not text:
        return float('inf')
    model, tokenizer, device = _load_gpt2()
    try:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return outputs.loss.item()
    except Exception as e:
        print(f'Warning: Could not compute perplexity: {e}')
        return float('inf')


def select_best_candidate_by_lm(candidates):
    """[原作者] 從候選句中挑 GPT-2 損失最低的一句"""
    best_score = float('inf')
    best_text = candidates[0] if candidates else ''
    for text in candidates:
        text_clean = text.strip()
        ppl = compute_gpt2_perplexity(text_clean)
        if ppl < best_score:
            best_score = ppl
            best_text = text_clean
    return best_text
