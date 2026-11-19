"""
自己動手寫的 Transformer  ★ 這個檔案是你的練習 ★
==================================================

這個專案裡，其他部分都是現成的，只有這個檔案要你自己完成。

在設定檔切換要用哪一個版本：
  model.transformer_impl: pytorch   ← 用 PyTorch 內建版（預設，現在就能跑）
  model.transformer_impl: mine      ← 用你在這裡寫的版本

驗收方式
--------
  python tests/test_my_transformer.py

每一題都會把「你的版本」和「PyTorch 官方版本」放進一模一樣的權重、
餵一模一樣的輸入，比對輸出。數值一樣，就代表你寫對了。

全部完成後的最終目標：用你自己寫的 Transformer，直接載入作者訓練好的權重。

練習清單（建議照順序做）
----------------------
  練習 1  scaled_dot_product_attention     縮放點積注意力（核心公式）
  練習 2  MyMultiheadAttention.forward     多頭注意力
  練習 3  MyTransformerEncoderLayer.forward 編碼器層（Pre-LN）
  練習 4  MyTransformerEncoder.forward     把編碼器層疊起來
  練習 5  MyTransformerDecoderLayer.forward 解碼器層（因果遮罩 + 交叉注意力）
  練習 6  MyTransformerDecoder.forward     把解碼器層疊起來

規則
----
  - 每個 __init__ 都已經寫好了。參數的「名稱」和「形狀」要和 PyTorch 官方完全相同，
    才能載入作者的權重，所以請不要改 __init__ 裡的變數名稱。
  - 你只需要把每個 forward（和練習 1 的函式）裡的
        raise NotImplementedError(...)
    換成你的程式碼。
  - 不可以呼叫 PyTorch 現成的注意力或 Transformer
    （例如 F.scaled_dot_product_attention、nn.MultiheadAttention），
    其他基本運算（矩陣乘法、softmax、Linear、LayerNorm、dropout）都可以用。

形狀符號
--------
  B：批次大小    L：查詢的長度    S：鍵／值的長度
  D：模型寬度（d_model = 512）    H：注意力頭數（8）    d：每個頭的寬度（D / H = 64）

遮罩的規則（和 PyTorch 相同，所有練習都照這個規則）
-------------------------------------------------
  attn_mask：形狀可以廣播成 (..., L, S)
    - 布林值：True 代表「不能看」
    - 浮點數：直接加到分數上（不能看的位置是 -inf，能看的位置是 0）
      例如解碼器的因果遮罩，就是浮點數版本：
        nn.Transformer.generate_square_subsequent_mask(4) =
          [[0., -inf, -inf, -inf],
           [0.,   0., -inf, -inf],
           [0.,   0.,   0., -inf],
           [0.,   0.,   0.,   0.]]
  key_padding_mask：形狀 (B, S) 的布林值，True 代表「這個位置是補上去的，不要看」
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 練習 1：縮放點積注意力
# =============================================================================

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V

    輸入：
      q：(..., L, d)   查詢
      k：(..., S, d)   鍵
      v：(..., S, d)   值
         「...」可以是任何前置維度，例如 (B,) 或 (B, H)
      attn_mask：見檔案開頭的遮罩規則；None 代表全部都能看
      dropout_p：對注意力權重做 dropout 的機率
      training：是否在訓練模式（只有 True 時才做 dropout）

    輸出：
      out：    (..., L, d)   混合後的結果
      weights：(..., L, S)   注意力權重（dropout 之前的），最後一維加總為 1

    步驟：
      1. 每個查詢和每個鍵做內積，得到分數 (..., L, S)
      2. 除以 √d（d 是 q 的最後一維）
      3. 套用遮罩：布林遮罩 → 不能看的位置設成 -inf；浮點遮罩 → 直接相加
      4. 在最後一維做 softmax，得到權重
      5. 對權重做 dropout（F.dropout(權重, p=dropout_p, training=training)）
      6. 權重乘上 v

    提示：矩陣乘法 @、交換最後兩維 .transpose(-2, -1)、
          .masked_fill(條件, float('-inf'))、torch.softmax(x, dim=-1)、
          判斷是不是布林值：attn_mask.dtype == torch.bool
    """
        # Step 1：內積（q 和每個 k 有多像）
    scores = q @ k.transpose(-2, -1)

    # Step 2：除以 √d
    d = q.size(-1)
    scores = scores / math.sqrt(d)

    # 遮罩（有給的話才做）
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        else:
            scores = scores + attn_mask

    # Step 3：softmax，變成加總為 1 的權重
    weights = torch.softmax(scores, dim=-1)

    # dropout（只在訓練時有作用）
    dropped = F.dropout(weights, p=dropout_p, training=training)

    # Step 4：用權重混合 v
    out = dropped @ v

    return out, weights


# =============================================================================
# 練習 2：多頭注意力
# =============================================================================

class MyMultiheadAttention(nn.Module):
    """
    對應 PyTorch 的 nn.MultiheadAttention（batch_first=True）。

    概念：把 D = 512 維切成 H = 8 個頭，每個頭 64 維，
          各自做一次練習 1 的注意力，最後再合併回 512 維。
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim 必須能被 num_heads 整除'
        assert batch_first, '這個專案只用 batch_first=True'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Q、K、V 三個投影矩陣「疊在一起」存成一個參數（和 PyTorch 官方相同）
        #   in_proj_weight[0:D]     → W_q
        #   in_proj_weight[D:2D]    → W_k
        #   in_proj_weight[2D:3D]   → W_v
        # 投影的算法：x @ W.T + b   （和 nn.Linear 一樣，也可以用 F.linear(x, W, b)）
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        # 合併所有頭之後的輸出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 初始化方式和 PyTorch 官方相同
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                need_weights=False):
        """
        輸入：
          query：(B, L, D)
          key、value：(B, S, D)
          key_padding_mask：(B, S) 布林值，True = 補上去的位置
          attn_mask：(L, S)，規則見檔案開頭
          need_weights：True 時回傳「8 個頭平均後」的注意力權重

        輸出：
          (out, weights)
          out：(B, L, D)
          weights：need_weights=True 時是 (B, L, S)，否則是 None
                   （回傳兩個值，是為了和 PyTorch 官方的呼叫方式相同）

        步驟：
          1. 把 in_proj_weight / in_proj_bias 切成 Q、K、V 三份，分別投影
             q = query 投影後 (B, L, D)；k、v 同理 (B, S, D)
          2. 切成多個頭：(B, L, D) → (B, L, H, d) → (B, H, L, d)
             提示：.view(B, L, H, d).transpose(1, 2)
          3. 合併兩種遮罩，做成一個布林或浮點遮罩，能廣播到 (B, H, L, S)
             key_padding_mask (B, S) → (B, 1, 1, S)
             attn_mask (L, S) 可以直接廣播
             兩個都有的時候：可以先都轉成浮點（True → -inf，False → 0）再相加
          4. 呼叫練習 1 的函式（記得傳 dropout_p=self.dropout、training=self.training）
          5. 合併多個頭：(B, H, L, d) → (B, L, H, d) → (B, L, D)
             提示：.transpose(1, 2).reshape(B, L, D)
          6. 通過 self.out_proj
        """
        raise NotImplementedError('練習 2：MyMultiheadAttention.forward 尚未完成')


# =============================================================================
# 練習 3：編碼器層
# =============================================================================

class MyTransformerEncoderLayer(nn.Module):
    """
    對應 nn.TransformerEncoderLayer(..., activation='gelu', batch_first=True, norm_first=True)

    案例 B 的編碼器用的是「Pre-LN」寫法（norm_first=True）：
        x = x + 自注意力區塊( LayerNorm1(x) )
        x = x + 前饋區塊(   LayerNorm2(x) )
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', batch_first=True, norm_first=True):
        super().__init__()
        assert activation == 'gelu' and batch_first and norm_first, \
            '這個專案只用 gelu、batch_first=True、norm_first=True'
        # 自注意力
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前饋網路：D → dim_feedforward → D
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)        # 前饋網路中間的 dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 兩個 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 兩個殘差前的 dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        輸入：src (B, L, D)；src_mask (L, L)；src_key_padding_mask (B, L)
        輸出：(B, L, D)

        步驟：
          自注意力區塊：
            a = self.norm1(x)
            a = self.self_attn(a, a, a, attn_mask=..., key_padding_mask=...) 的第 0 個輸出
            x = x + self.dropout1(a)
          前饋區塊：
            f = self.norm2(x)
            f = self.linear2( self.dropout( self.activation( self.linear1(f) ) ) )
            x = x + self.dropout2(f)
        """
        raise NotImplementedError('練習 3：MyTransformerEncoderLayer.forward 尚未完成')


# =============================================================================
# 練習 4：把編碼器層疊起來
# =============================================================================

class MyTransformerEncoder(nn.Module):
    """
    對應 nn.TransformerEncoder(encoder_layer, num_layers)

    把同一個層「複製」num_layers 份（每份有自己獨立的參數），依序通過。
    案例 B：8 層，最後沒有額外的 LayerNorm（那個 LayerNorm 在 NeuralEncoder 裡）。
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 名稱必須叫 layers，載入權重時才對得上（transformer.layers.0.self_attn....）
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        輸入：src (B, L, D)；mask (L, L)；src_key_padding_mask (B, L)
        輸出：(B, L, D)

        步驟：依序把資料通過 self.layers 裡的每一層，每一層都傳入同樣的遮罩。
        """
        raise NotImplementedError('練習 4：MyTransformerEncoder.forward 尚未完成')


# =============================================================================
# 練習 5：解碼器層
# =============================================================================

class MyTransformerDecoderLayer(nn.Module):
    """
    對應 nn.TransformerDecoderLayer(..., activation='gelu', batch_first=True, norm_first=True)

    比編碼器層多一個「交叉注意力」區塊：
        x = x + 遮罩自注意力區塊( LayerNorm1(x) )         ← 只能看自己和前面的字
        x = x + 交叉注意力區塊(   LayerNorm2(x), memory ) ← 去看編碼器的輸出（腦訊號）
        x = x + 前饋區塊(        LayerNorm3(x) )
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', batch_first=True, norm_first=True):
        super().__init__()
        assert activation == 'gelu' and batch_first and norm_first, \
            '這個專案只用 gelu、batch_first=True、norm_first=True'
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        輸入：
          tgt：   (B, T, D)   已經寫出的文字（嵌入後）
          memory：(B, S, D)   編碼器的輸出
          tgt_mask：(T, T)    因果遮罩
          memory_mask：(T, S)
          tgt_key_padding_mask：(B, T)
          memory_key_padding_mask：(B, S)
        輸出：(B, T, D)

        步驟：
          自注意力：   a = norm1(x)；a = self_attn(a, a, a, tgt_mask, tgt_key_padding_mask)
                       x = x + dropout1(a)
          交叉注意力： c = norm2(x)；c = multihead_attn(c, memory, memory, memory_mask, memory_key_padding_mask)
                       x = x + dropout2(c)
                       注意：query 來自文字（c），key 和 value 來自腦訊號（memory）
          前饋：       f = norm3(x)；f = linear2(dropout(activation(linear1(f))))
                       x = x + dropout3(f)
        """
        raise NotImplementedError('練習 5：MyTransformerDecoderLayer.forward 尚未完成')


# =============================================================================
# 練習 6：把解碼器層疊起來
# =============================================================================

class MyTransformerDecoder(nn.Module):
    """
    對應 nn.TransformerDecoder(decoder_layer, num_layers)
    案例 B：6 層，最後沒有額外的 LayerNorm。
    """

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        輸入、輸出同練習 5。
        步驟：依序通過每一層；每一層都用同一個 memory 和同樣的遮罩。
        """
        raise NotImplementedError('練習 6：MyTransformerDecoder.forward 尚未完成')
