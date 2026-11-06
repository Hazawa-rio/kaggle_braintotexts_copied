# Brain-to-text '25 案例 B 復現

復現 Kaggle Brain-to-text '25 比賽中的公開 notebook「no-phonemes」（作者自稱 DSD-NLA）：
一個把腦訊號直接解碼成英文字母的 Transformer 編碼器－解碼器。

- 比賽頁面：https://www.kaggle.com/competitions/brain-to-text-25
- 官方基準程式：https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- 原 notebook：（請填入 Kaggle 連結與原作者，並確認授權條款）

## 復現計畫

| 版本 | 目標 |
|---|---|
| **v0** | 忠實復現原 notebook，保留原本的寫法（包括已知問題） |
| **v1** | 一次修正一個問題，記錄每個修正帶來的改變 |

已知問題的編號（P1–P17）見拆解筆記第 5 節，程式碼註解中以 `P#` 標示。

## 目前進度

- [x] 專案結構、設定檔、實驗紀錄與數據匯出
- [x] 字母編碼器 `src/tokenizer.py`
- [x] WER 與錯誤拆解 `src/metrics.py`
- [ ] 模型（編碼器、解碼器）
- [ ] 資料讀取
- [ ] 訓練
- [ ] 推論

## 目錄結構

```
configs/            設定檔（所有實驗參數都在這裡）
src/                主要程式
  config.py           讀取設定、命令列覆蓋
  experiment.py       實驗紀錄與數據匯出
  tokenizer.py        字母級編碼器
  metrics.py          WER 與三種錯誤的拆解
scripts/            可以直接執行的實驗
  evaluate_predictions.py   評估預測結果
  lr_schedule.py            學習率排程（問題 P7）
tests/              測試
examples/           範例資料
experiments/        所有實驗的總表 all_runs.csv（會上傳）
runs/               每次實驗的詳細輸出（不上傳）
```

## 安裝

```bash
pip install -r requirements.txt
```

## 怎麼做實驗

### 1. 執行

所有指令都在 repo 最上層資料夾執行。

```bash
# 評估一份預測結果
python scripts/evaluate_predictions.py --input examples/sample_predictions.csv

# 計算原作者的學習率排程
python scripts/lr_schedule.py
```

### 2. 調整參數

**不要直接改 `configs/v0.yaml`**，用 `--set` 臨時覆蓋，可以用很多次：

```bash
python scripts/lr_schedule.py --set train.use_assumed_lr_steps=false
python scripts/lr_schedule.py --set train.num_epochs=10 --set train.learning_rate=0.0001
python scripts/evaluate_predictions.py --input examples/sample_predictions.csv --set eval.normalize=false
```

格式是「區塊.參數=數值」，區塊和參數名稱對照 `configs/v0.yaml`。打錯字會直接報錯。

要長期使用的設定，複製一份設定檔再改：

```bash
cp configs/v0.yaml configs/exp_my_idea.yaml
python scripts/lr_schedule.py --config configs/exp_my_idea.yaml
```

### 3. 看結果與匯出數據

每次執行都會產生一個資料夾 `runs/日期-時間_實驗名稱/`：

| 檔案 | 內容 |
|---|---|
| `config.yaml` | 這次實際使用的完整設定 |
| `meta.json` | 執行時間、Git 版本、改了哪些參數 |
| `metrics.csv` | 過程數據（例如每一輪的損失、學習率） |
| `predictions.csv` | 每一句的預測和錯誤拆解（評估類實驗才有） |
| `summary.json` | 最終成績 |

另外，**每次實驗都會在 `experiments/all_runs.csv` 新增一列**，包含改了什麼、最終成績和完整設定。
用 Excel 打開就能排序、篩選、畫圖比較。

## 測試

```bash
python tests/test_tokenizer.py
python tests/test_metrics.py
python tests/test_config.py
```

或安裝 pytest 後：`python -m pytest tests -v`
