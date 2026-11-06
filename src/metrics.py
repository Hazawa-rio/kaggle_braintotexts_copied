"""
字詞錯誤率（WER）與錯誤拆解
==============================

[新增] 原 notebook 沒有這個檔案。
原作者只用驗證損失挑模型，沒有計算 WER（拆解筆記問題 P8）。
這裡自己實作，除了算出 WER，還能把錯誤拆成三種：

  替換（Substitution, S）：該寫 A 卻寫成 B      want some water → want sum water
  刪除（Deletion, D）    ：漏掉一個字           how are you     → how you
  插入（Insertion, I）   ：多寫一個字           thank you       → thank you very much

計算方式（和官方 evaluate_model.py 相同）
---------------------------------------
  WER = 所有句子的 (S + D + I) 加總 ÷ 所有正確句子的總字數 N

  注意是「整體加總後再相除」，不是「每句算完再平均」。
  所以長句子的影響比較大；而且因為插入也算錯，WER 可以超過 100%。

核心演算法：編輯距離（動態規劃）
------------------------------
  dp[i][j] = 把「正確句的前 i 個字」變成「預測句的前 j 個字」最少要幾步

  每一格從三個方向而來，取最小：
    左上 dp[i-1][j-1] + (0 或 1)   兩個字相同 → 對上（0 步）；不同 → 替換（1 步）
    上方 dp[i-1][j]   + 1          刪除：正確句的字在預測句裡沒有
    左方 dp[i][j-1]   + 1          插入：預測句多出一個字

  填完整張表後，右下角就是編輯距離。
  再從右下角往回走（回溯），就能知道每一步是哪一種錯誤。
"""

import re


# ---------------------------------------------------------------------------
# 1. 正規化：計算前先統一格式
# ---------------------------------------------------------------------------

def normalize_text(sentence):
    """
    把句子整理成統一格式再比較。

    [新增] 規則照抄官方 repo 的 remove_punctuation()
    （model_training/evaluate_model_helpers.py），確保評分方式和官方一致：
      1. 只保留英文字母、連字號 -、空白、撇號 '，其他字元直接刪除
      2. 處理連字號和撇號的特殊情況
      3. 全部轉小寫，去掉多餘空白

    範例：
      "I'm fine, thanks!"  →  "i'm fine thanks"
      "Room 101"           →  "room"          （數字被刪除）

    注意：原 notebook 的 normalize_for_eval() 規則不同，
    它保留數字、把連字號換成空白（問題 P14）。
    評分時一律以這裡的官方規則為準。
    """
    sentence = re.sub(r"[^a-zA-Z\- ']", '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])
    return sentence


# ---------------------------------------------------------------------------
# 2. 對齊：找出兩句話之間的最少修改步驟
# ---------------------------------------------------------------------------

def align(ref_words, hyp_words):
    """
    用動態規劃對齊兩個字詞清單，並回溯出每一步的操作。

    輸入：
      ref_words：正確句的字詞清單，例如 ['i', 'want', 'some', 'water']
      hyp_words：預測句的字詞清單，例如 ['i', 'want', 'sum', 'water']

    輸出：操作清單，每個元素是 (操作, 正確的字, 預測的字)
      操作有四種：
        'C'  對上（Correct）       ('C', 'i', 'i')
        'S'  替換（Substitution）  ('S', 'some', 'sum')
        'D'  刪除（Deletion）      ('D', 'are', None)
        'I'  插入（Insertion）     ('I', None, 'very')
    """
    n, m = len(ref_words), len(hyp_words)

    # 建立 (n+1) × (m+1) 的表格
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 第一欄：正確句有 i 個字、預測句是空的 → 要刪除 i 次
    for i in range(n + 1):
        dp[i][0] = i
    # 第一列：正確句是空的、預測句有 j 個字 → 要插入 j 次
    for j in range(m + 1):
        dp[0][j] = j

    # 一格一格填表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,   # 左上：對上或替換
                dp[i - 1][j] + 1,          # 上方：刪除
                dp[i][j - 1] + 1,          # 左方：插入
            )

    # 從右下角往回走，記錄每一步
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        # 同分時的優先順序：對上／替換 → 刪除 → 插入
        # 編輯距離的總數不受影響，但 S、D、I 各自的數量可能因順序而不同
        # 這裡的順序和常見工具（例如 jiwer）的慣例相同
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                op = 'C' if cost == 0 else 'S'
                ops.append((op, ref_words[i - 1], hyp_words[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('D', ref_words[i - 1], None))
            i -= 1
            continue
        ops.append(('I', None, hyp_words[j - 1]))
        j -= 1

    ops.reverse()   # 回溯是從尾巴往前，所以要反轉回正常順序
    return ops


def count_errors(ref, hyp, normalize=True):
    """
    計算一句話的錯誤數。

    輸入：
      ref：正確句（字串）
      hyp：預測句（字串）
      normalize：是否先用官方規則正規化（預設會）

    輸出：字典
      {'S': 替換數, 'D': 刪除數, 'I': 插入數,
       'N': 正確句字數, 'errors': S + D + I, 'ops': 對齊結果}
    """
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    ref_words, hyp_words = ref.split(), hyp.split()

    ops = align(ref_words, hyp_words)
    s = sum(1 for op, _, _ in ops if op == 'S')
    d = sum(1 for op, _, _ in ops if op == 'D')
    i = sum(1 for op, _, _ in ops if op == 'I')
    return {'S': s, 'D': d, 'I': i, 'N': len(ref_words),
            'errors': s + d + i, 'ops': ops}


# ---------------------------------------------------------------------------
# 3. 整體 WER
# ---------------------------------------------------------------------------

def wer(refs, hyps, normalize=True):
    """
    計算一整組句子的 WER，並拆解三種錯誤。

    輸入：
      refs：正確句清單
      hyps：預測句清單（順序要和 refs 對應）

    輸出：字典
      'wer'        整體 WER（0.25 代表 25%）
      'S', 'D', 'I' 三種錯誤的總數
      'N'          正確句的總字數
      'per_sentence' 每一句的詳細結果（方便找出錯最多的句子）
    """
    if len(refs) != len(hyps):
        raise ValueError(f'句數不同：正確句 {len(refs)} 句，預測句 {len(hyps)} 句')

    total = {'S': 0, 'D': 0, 'I': 0, 'N': 0}
    per_sentence = []
    for ref, hyp in zip(refs, hyps):
        r = count_errors(ref, hyp, normalize)
        for k in total:
            total[k] += r[k]
        per_sentence.append(r)

    errors = total['S'] + total['D'] + total['I']
    total['wer'] = errors / total['N'] if total['N'] > 0 else float('inf')
    total['per_sentence'] = per_sentence
    return total


# ---------------------------------------------------------------------------
# 4. 顯示對齊結果（分析用）
# ---------------------------------------------------------------------------

def format_alignment(ops):
    """
    把對齊結果排成三列，方便肉眼比對。

    範例輸出：
      正確：  i  want  some  water
      預測：  i  want  sum   water
      錯誤：           S
    """
    ref_row, hyp_row, op_row = [], [], []
    for op, r, h in ops:
        r = r if r is not None else '*' * len(h)   # 插入時，正確句那格用 * 填
        h = h if h is not None else '*' * len(r)   # 刪除時，預測句那格用 * 填
        width = max(len(r), len(h))
        ref_row.append(r.ljust(width))
        hyp_row.append(h.ljust(width))
        op_row.append(('' if op == 'C' else op).ljust(width))
    return ('正確：  ' + '  '.join(ref_row) + '\n'
            '預測：  ' + '  '.join(hyp_row) + '\n'
            '錯誤：  ' + '  '.join(op_row).rstrip())


# 直接執行這個檔案時，跑一次範例
if __name__ == "__main__":
    refs = ["I want some water", "How are you?", "Thank you."]
    hyps = ["i want sum water",  "how you",      "thank you very much"]

    result = wer(refs, hyps)
    for k, r in enumerate(result['per_sentence']):
        print(f"第 {k + 1} 句（S={r['S']} D={r['D']} I={r['I']} / N={r['N']}）")
        print(format_alignment(r['ops']))
        print()

    print(f"替換 {result['S']}、刪除 {result['D']}、插入 {result['I']}，"
          f"總字數 {result['N']}")
    print(f"WER = {result['wer']:.1%}")
