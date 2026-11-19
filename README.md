# Brain-to-text '25 案例 B 復現

復現 Kaggle Brain-to-text '25 比賽中的公開 notebook「no-phonemes」（作者自稱 DSD-NLA）：
一個把腦訊號直接解碼成英文字母的 Transformer 編碼器－解碼器。

**這個專案的特色**：除了 Transformer 之外的部分都照原 notebook 搬好了；
Transformer 本身（`src/model/my_transformer.py`）留給自己從零實作，並用自動測試和 PyTorch 官方版本逐位比對。

- 比賽頁面：https://www.kaggle.com/competitions/brain-to-text-25
- 官方基準程式：https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- 原 notebook：（請填入 Kaggle 連結與原作者，並確認授權條款）

## 復現計畫

| 版本 | 目標 |
|---|---|
| **v0** | 忠實復現原 notebook，保留原本的寫法（包括已知問題） |
| **v1** | 一次修正一個問題，記錄每個修正帶來的改變 |

已知問題的編號（P1–P17）見拆解筆記第 5 節，程式碼註解中以 `P#` 標示。
註解中的 `[原作者]` 代表原 notebook 的寫法，`[新增]` 代表為了搬到 repo 而加的部分。

## 目前進度

- [x] 設定檔、實驗紀錄與數據匯出
- [x] 字母編碼器、WER 與錯誤拆解
- [x] 模型（使用 PyTorch 內建 Transformer）
- [x] 資料讀取、訓練、推論
- [ ] **自己實作的 Transformer**（`src/model/my_transformer.py`，練習 1–6）
- [ ] 下載資料並執行 v0

## 目錄結構

```
configs/v0.yaml         所有實驗參數
src/
  config.py               讀取設定、命令列覆蓋
  experiment.py           實驗紀錄與數據匯出
  tokenizer.py            字母級編碼器
  metrics.py              WER 與三種錯誤的拆解
  dataset.py              讀資料、資料增強
  losses.py               損失函數
  train.py                訓練器
  decode.py               集束搜尋
  postprocess.py          GPT-2 挑選、拼字檢查、正規化
  infer.py                推論
  model/
    my_transformer.py     ★ 自己實作的 Transformer（練習）
    positional.py         位置編碼
    encoder.py            神經編碼器
    decoder.py            文字解碼器
    diffusion.py          擴散模型（沒有運作，為了載入權重而保留）
    aligner.py            對齊器（同上）
    dsdnla.py             組合外殼
scripts/                可以直接執行的實驗（見下方）
tests/                  測試
examples/               範例資料
experiments/all_runs.csv  所有實驗的總表（會上傳）
runs/                   每次實驗的詳細輸出（不上傳）
```

## 安裝

```bash
pip install -r requirements.txt
# PyTorch 另外裝。沒有 GPU（例如 Codespaces）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 自己實作 Transformer

1. 打開 `src/model/my_transformer.py`，照練習 1 到 6 的順序，把每個 `raise NotImplementedError(...)` 換成自己的程式碼
2. 每寫完一題就檢查：

```bash
python tests/test_my_transformer.py
```

```
✅ 通過    練習 1 attention
⏳ 尚未完成 練習 2 multihead_attention
...
```

3. 全部通過後，切換成自己的版本：

```bash
python scripts/model_summary.py --set model.transformer_impl=mine
python scripts/infer.py --split val --limit 10 --set model.transformer_impl=mine
```

## 實驗一覽

所有指令都在 repo 最上層資料夾執行。

| 實驗 | 指令 | 需要 |
|---|---|---|
| 評分 | `python scripts/evaluate_predictions.py --input examples/sample_predictions.csv` | — |
| 學習率排程 | `python scripts/lr_schedule.py` | — |
| 模型摘要 | `python scripts/model_summary.py` | PyTorch |
| 資料檢查 | `python scripts/check_data.py` | 資料 |
| 訓練 | `python scripts/train.py` | 資料、建議 GPU |
| 推論與評分 | `python scripts/infer.py --split val` | 資料、模型檔 |

### 調整參數

**不要直接改 `configs/v0.yaml`**，用 `--set` 臨時覆蓋，可以用很多次：

```bash
python scripts/model_summary.py --set model.n_encoder_layers=4
python scripts/train.py --set train.max_trials=100 --set train.num_epochs=2 --set experiment_name=smoke_test
python scripts/infer.py --split val --set inference.use_spellcheck=false
```

格式是「區塊.參數=數值」，對照 `configs/v0.yaml`。打錯字會直接報錯。

### 看結果與匯出數據

每次執行都會產生 `runs/日期-時間_實驗名稱/`：

| 檔案 | 內容 |
|---|---|
| `config.yaml` | 這次實際使用的完整設定 |
| `meta.json` | 執行時間、Git 版本、改了哪些參數 |
| `metrics.csv` | 過程數據（每一輪的損失、學習率） |
| `predictions.csv` | 每一句的預測和錯誤拆解（評分、推論驗證集） |
| `details.csv` | 每一句的候選句、後處理前的句子（推論） |
| `parameters.csv` | 各模組參數量 |
| `summary.json` | 最終成績 |

每次實驗也會在 `experiments/all_runs.csv` 新增一列，用 Excel 打開就能排序、篩選、畫圖比較。

## 資料與模型檔

- 腦訊號資料：從 [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85) 下載 `t15_copyTask_neuralData.zip`，
  或從 Kaggle 比賽頁下載，解壓縮後放在 `data/hdf5_data_final/`（應有 45 個 `t15.*` 資料夾）
- 作者訓練好的模型：從 Kaggle 資料集 `dsd-nla` 下載 `best_dsdnla_model.pt`，放在 `checkpoints/`

兩者都已列入 `.gitignore`，不會上傳到 GitHub。

## 測試

```bash
python tests/test_metrics.py         # 不需要 PyTorch
python tests/test_config.py          # 不需要 PyTorch
python tests/test_tokenizer.py
python tests/test_model.py           # 參數量要剛好 75,173,412
python tests/test_my_transformer.py  # 自己實作的 Transformer
```
