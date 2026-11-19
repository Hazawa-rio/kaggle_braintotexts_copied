"""
模型的形狀與參數量測試（使用 PyTorch 內建的 Transformer）

執行方式（在 repo 根目錄）：
  python tests/test_model.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from src.config import load_config
from src.losses import DSDNLALoss
from src.model.dsdnla import build_model, count_parameters
from src.model.encoder import NeuralEncoder
from src.model.positional import PositionalEncoding
from src.tokenizer import CharTokenizer

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'v0.yaml')
torch.manual_seed(0)


def _model():
    cfg, _ = load_config(CONFIG, ['model.transformer_impl=pytorch'])
    return build_model(cfg, CharTokenizer().vocab_size).eval()


def test_parameter_counts():
    """各模組參數量要和原 notebook 完全一致"""
    counts = {r['module']: r['params'] for r in count_parameters(_model())}
    assert counts['neural_encoder'] == 26_532_352, counts
    assert counts['diffusion_prior'] == 21_276_672, counts
    assert counts['aligner'] == 2_103_296, counts
    assert counts['text_decoder'] == 25_261_092, counts
    assert counts['total'] == 75_173_412, counts


def test_positional_encoding_shape():
    pe = PositionalEncoding(512, dropout=0.0)
    x = torch.zeros(2, 75, 512)
    out = pe(x)
    assert out.shape == (2, 75, 512)
    # 位置 0：sin(0) = 0、cos(0) = 1
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(256))
    assert torch.allclose(out[0, 0, 1::2], torch.ones(256))


def test_encoder_halves_length():
    enc = NeuralEncoder(dropout=0.0).eval()
    for T in [150, 151, 7]:
        with torch.no_grad():
            z = enc(torch.randn(2, T, 512))
        assert z.shape == (2, (T + 1) // 2, 512), (T, z.shape)


def test_full_forward_and_loss():
    model = _model()
    tok = CharTokenizer()
    x = torch.randn(2, 150, 512)
    tokens = torch.nn.utils.rnn.pad_sequence(
        [tok.encode('hi there'), tok.encode('ok')], batch_first=True, padding_value=0)
    with torch.no_grad():
        logits, aux = model(x, target_tokens=tokens, training=True)
    assert logits.shape == (2, tokens.size(1), 36)
    assert aux == {}, '沒有提供語音資料時，不該有擴散和對齊損失（P2）'
    losses = DSDNLALoss()(logits, tokens, aux)
    assert losses['total'].item() > 0


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'通過：{t.__name__}')
    print(f'\n全部 {len(tests)} 項測試通過')
