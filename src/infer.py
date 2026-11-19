"""
推論
====

對應原 notebook：Cell 2 的 InferenceEngine、generate_submission
拆解筆記：4.11

每一句的流程（use_beam_search = true 時，原 notebook 的最終設定）：
  腦訊號 → 編碼器 → 集束搜尋（候選句）→ GPT-2 挑選 → 拼字檢查 → 清理 → 正規化

[新增] 的部分：
  - 設定從 yaml 讀取；GPT-2 挑選和拼字檢查可以用設定開關（預設和原本相同：都開）
  - GPT-2 關掉時，改取集束搜尋分數最高的候選句
  - 可以處理驗證集（有正確答案，能算 WER）
  - 自動選擇 GPU 或 CPU
"""

import h5py
import torch
from tqdm import tqdm

from src.dataset import find_split_files, read_sentence_label
from src.decode import BeamSearchGenerator
from src.model.dsdnla import build_model
from src.postprocess import (clean_generated_text, normalize_for_eval,
                             select_best_candidate_by_lm, spell_fix)
from src.tokenizer import CharTokenizer


def load_model(cfg, checkpoint_path, device):
    """
    建立模型並載入權重。

    使用嚴格模式（strict=True）：參數名稱或形狀只要有一個對不上就會報錯，
    確保真的載入了全部權重。
    推論時 dropout 設成 0（和原作者相同）。
    """
    tokenizer = CharTokenizer()
    model_cfg = dict(cfg, model=dict(cfg['model'], dropout=0.0))
    model = build_model(model_cfg, tokenizer.vocab_size)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def predict_one(model, tokenizer, neural_features, cfg, device, beam=None):
    """
    一句腦訊號 → 一句英文

    neural_features：(T, 512) 的 numpy 陣列
    回傳：(最終句子, 中間結果字典)
    """
    inf = cfg['inference']
    x = torch.tensor(neural_features, dtype=torch.float32).unsqueeze(0).to(device)   # (1, T, 512)
    info = {}

    if inf['use_beam_search'] and beam is not None:
        z_neural = model.neural_encoder(x)
        candidates = beam.generate_candidates(z_neural)
        info['candidates'] = candidates
        if inf['use_gpt2_rescoring']:
            text = select_best_candidate_by_lm(candidates)     # P12
        else:
            text = candidates[0].strip()                         # [新增] 取集束分數最高的
    else:
        # P3、P4：走有問題的 generate()
        tokens = model.inference(x, max_len=inf['greedy_max_len'])
        text = tokenizer.decode(tokens[0])
        text = ' '.join(text.strip().split())

    info['before_postprocess'] = text
    if inf['use_spellcheck']:
        text = spell_fix(text)                                   # P13
    text = clean_generated_text(text)
    text = normalize_for_eval(text)                              # P14
    return text, info


def run_inference(cfg, checkpoint_path, split='test', limit=None, device=None):
    """
    對某個切分的所有句子做推論。

    回傳：清單，每個元素是
      {'id', 'session', 'trial', 'reference', 'prediction', 'before_postprocess', 'candidates'}
    id 是 0 開始的流水號，順序為「日期由早到晚 → 每天依 trial 編號」（和原 notebook 相同）
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(cfg, checkpoint_path, device)
    inf = cfg['inference']
    beam = (BeamSearchGenerator(model, tokenizer, beam_width=inf['beam_width'],
                                max_len=inf['beam_max_len'])
            if inf['use_beam_search'] else None)

    files = find_split_files(cfg['paths']['data_dir'], split)
    print(f'Found {len(files)} {split} files. Device: {device}')

    results = []
    for path in tqdm(files, desc='Files'):
        session = path.replace('\\', '/').split('/')[-2]
        with h5py.File(path, 'r') as f:
            for key in sorted(f.keys()):
                if limit is not None and len(results) >= limit:
                    return results
                trial = f[key]
                text, info = predict_one(model, tokenizer, trial['input_features'][:], cfg, device, beam)
                results.append({
                    'id': len(results),
                    'session': session,
                    'trial': key,
                    'reference': read_sentence_label(trial) or '',
                    'prediction': text,
                    'before_postprocess': info.get('before_postprocess', ''),
                    'candidates': ' | '.join(info.get('candidates', [])),
                })
    return results
