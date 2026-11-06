"""
設定檔讀取與覆蓋
================

[新增] 取代原 notebook 的 Config 類別。

為什麼不用原本的 Config 類別？
  - 原本的設定寫死在程式碼裡，每次調參數都要改程式
  - 原本存進模型檔的設定其實是空的（問題 P9）
  改用 yaml 檔之後，設定和程式分開，而且每次實驗都能完整存檔。

用法
----
  from src.config import load_config
  cfg = load_config('configs/v0.yaml', overrides=['train.learning_rate=0.0001'])
  print(cfg['train']['learning_rate'])    # 0.0001

覆蓋的寫法：「區塊.參數=數值」
  train.learning_rate=0.0001
  inference.beam_width=5
  augment.enabled=false
  experiment_name=lr_1e-4
"""

import copy

import yaml


def load_config(path, overrides=None):
    """
    讀取 yaml 設定檔，再套用命令列的覆蓋。

    輸入：
      path：設定檔路徑，例如 'configs/v0.yaml'
      overrides：字串清單，例如 ['train.learning_rate=0.0001']

    輸出：
      (cfg, changes)
      cfg：     套用覆蓋後的完整設定（字典）
      changes： 這次被改掉的參數 {'train.learning_rate': (原值, 新值)}
                會寫進實驗總表，方便日後一眼看出每次實驗改了什麼
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    changes = {}
    for item in overrides or []:
        key, value = _parse_override(item)
        old = set_by_path(cfg, key, value)
        changes[key] = (old, value)
    return cfg, changes


def _parse_override(item):
    """
    把 'train.learning_rate=0.0001' 拆成 ('train.learning_rate', 0.0001)。

    數值用 yaml 的規則自動判斷型別：
      '0.0001' → 浮點數    '5' → 整數    'false' → 布林值    'abc' → 字串
    """
    if '=' not in item:
        raise ValueError(f'覆蓋格式錯誤：{item!r}，應為「區塊.參數=數值」')
    key, raw = item.split('=', 1)
    return key.strip(), yaml.safe_load(raw)


def set_by_path(cfg, key, value):
    """
    用「點」分隔的路徑修改巢狀字典，回傳原本的值。

    例如 key = 'train.learning_rate' 會修改 cfg['train']['learning_rate']。
    路徑不存在時會報錯，避免打錯字卻沒發現（例如寫成 train.learnig_rate）。
    """
    parts = key.split('.')
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f'設定檔裡沒有「{key}」，請檢查拼字')
        node = node[p]
    last = parts[-1]
    if last not in node:
        raise KeyError(f'設定檔裡沒有「{key}」，請檢查拼字')
    old = node[last]
    node[last] = value
    return old


def flatten(cfg, prefix=''):
    """
    把巢狀字典攤平成一層，方便寫進 CSV。

    {'train': {'learning_rate': 0.0005}}  →  {'train.learning_rate': 0.0005}
    """
    flat = {}
    for k, v in cfg.items():
        name = f'{prefix}{k}'
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=name + '.'))
        else:
            flat[name] = v
    return flat


def save_config(cfg, path):
    """把實際使用的完整設定存成 yaml（保留原本的順序）。"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, allow_unicode=True, sort_keys=False)
