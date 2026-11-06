"""
設定檔與實驗紀錄的測試

執行方式（在 repo 根目錄）：
  python tests/test_config.py
"""

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import flatten, load_config
from src.experiment import Experiment

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'v0.yaml')


def test_load_v0_values():
    """v0 設定要和原 notebook 的 Config 一致"""
    cfg, changes = load_config(CONFIG)
    assert changes == {}
    assert cfg['model']['d_model'] == 512
    assert cfg['model']['n_encoder_layers'] == 8
    assert cfg['train']['batch_size'] == 16
    assert cfg['train']['learning_rate'] == 0.0005
    assert cfg['inference']['beam_max_len'] == 80


def test_override_types():
    """--set 的數值會自動判斷型別"""
    cfg, changes = load_config(CONFIG, [
        'train.learning_rate=0.0001',   # 浮點數
        'train.num_epochs=10',          # 整數
        'augment.enabled=false',        # 布林值
        'experiment_name=my_test',      # 字串
    ])
    assert cfg['train']['learning_rate'] == 0.0001
    assert cfg['train']['num_epochs'] == 10
    assert cfg['augment']['enabled'] is False
    assert cfg['experiment_name'] == 'my_test'
    assert changes['train.learning_rate'] == (0.0005, 0.0001)


def test_typo_raises():
    """打錯參數名稱要報錯，而不是默默新增一個沒用的參數"""
    try:
        load_config(CONFIG, ['train.learnig_rate=0.1'])
    except KeyError:
        return
    raise AssertionError('打錯字應該報錯')


def test_flatten():
    assert flatten({'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}) == {'a.b': 1, 'a.c.d': 2, 'e': 3}


def test_experiment_outputs():
    """一次完整的實驗要產生所有檔案，並在總表新增一列"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, changes = load_config(CONFIG, ['experiment_name=unit_test'])
        cfg['paths']['runs_dir'] = os.path.join(tmp, 'runs')
        cfg['paths']['summary_csv'] = os.path.join(tmp, 'all_runs.csv')

        with Experiment(cfg, changes) as exp:
            exp.log_metrics({'epoch': 0, 'loss': 2.0})
            exp.log_metrics({'epoch': 1, 'loss': 1.5})
            r = exp.save_predictions(['how are you'], ['how you'])
            exp.finish({'wer': r['wer']})

        for name in ['config.yaml', 'meta.json', 'metrics.csv', 'predictions.csv', 'summary.json']:
            assert os.path.exists(os.path.join(exp.dir, name)), name
        with open(os.path.join(exp.dir, 'summary.json'), encoding='utf-8') as f:
            assert json.load(f)['status'] == 'finished'
        with open(cfg['paths']['summary_csv'], encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]['experiment_name'] == 'unit_test'


def test_failed_experiment_is_recorded():
    """中途出錯時，總表也要記下「failed」"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, changes = load_config(CONFIG, ['experiment_name=will_fail'])
        cfg['paths']['runs_dir'] = os.path.join(tmp, 'runs')
        cfg['paths']['summary_csv'] = os.path.join(tmp, 'all_runs.csv')
        try:
            with Experiment(cfg, changes):
                raise RuntimeError('模擬錯誤')
        except RuntimeError:
            pass
        with open(cfg['paths']['summary_csv'], encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['status'] == 'failed'


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f"通過：{t.__name__}")
    print(f"\n全部 {len(tests)} 項測試通過")
