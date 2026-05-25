# Transformer 的注意力機制：讓每個詞同時讀懂整句話

## TL;DR

- Transformer 最核心的創新是 **Attention（注意力機制）**：讓每個詞在處理時，都能直接「看到」句子中所有其他詞，並自己決定要關注哪些。
- 傳統 RNN 是逐字讀取，距離越遠記憶越模糊；Transformer 是全局並行計算，沒有距離懲罰。
- 每個詞被拆成三個角色：**Query（問）、Key（被問）、Value（給的內容）**，三者互動決定注意力分佈。
- **Multi-Head Attention** 同時做多組不同視角的注意力計算，捕捉語法、指代、語意等多維關係。
- 現代大語言模型（GPT、Claude、Gemini）的底層都是 Transformer 架構的變體。

---

## 核心譬喻：開會與會議記錄分析師

### 適用範圍

傳統語言模型像一個**逐字記錄的秘書**——聽到後面的詞時，前面的已經開始淡忘。

Transformer 像一個**同時看到整份會議記錄的分析師**：讀到任何一個詞時，都能直接對照整份記錄，決定哪些詞跟它最相關，然後把那些詞的資訊加權匯總進來。

### 失準之處

這個比喻暗示「分析師」是個外部觀察者，但 Transformer 的 Q、K、V 是**從輸入本身學出來的**——更像每個人在開會前，根據當下話題臨時重新定義自己的「標籤與名片」，而不是用固定的身份被查詢。

---

## 機制拆解

### 1. Query、Key、Value：三張名片

每個詞的向量輸入後，通過三個不同的線性投影，變成三個角色：

| 角色 | 意義 | 類比 |
|---|---|---|
| **Query (Q)** | 我想找什麼？ | 圖書館搜尋的關鍵字 |
| **Key (K)** | 我有什麼可以被找到？ | 書籍的索引標籤 |
| **Value (V)** | 找到我之後，給你的實際內容 | 書的正文 |

Q 去跟所有詞的 K 做內積，得到相關性分數；分數過 softmax 後，加權匯總所有詞的 V，就是這個詞「吸收完周圍資訊」後的新表示。

### 2. 注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$：每個詞的 Q 對所有詞的 K 做內積，產生 $n \times n$ 的分數矩陣（$n$ 是序列長度）。
- $\sqrt{d_k}$：縮放因子，防止向量維度大時內積值過大，導致 softmax 梯度消失。
- softmax 之後得到機率分佈，再乘以 V 做加權平均。

### 3. Multi-Head Attention：多視角並行

單組 Q、K、V 視角太單一。Transformer 把整個注意力過程**同時執行 $h$ 次**（原始論文 $h=8$），每個 head 用不同的投影矩陣：

- Head 1 可能在看**語法依存**（主詞動詞）
- Head 2 可能在看**指代關係**（「它」指哪個名詞）
- Head 3 可能在看**語意相近**（同義詞、近義詞）

最後把 $h$ 個 head 的輸出串接，再投影回原始維度。這讓模型能同時理解多種語言結構。

### 4. 位置編碼（Positional Encoding）

注意力本身**沒有順序概念**——「貓吃魚」和「魚吃貓」的詞集合完全一樣，純注意力無法區分。

解法是在輸入嵌入上**直接疊加位置資訊**：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

現代模型多改用 **RoPE（Rotary Position Embedding，旋轉位置編碼）**，將位置資訊直接融入 Q、K 的旋轉中，對長序列泛化更好。

### 5. Feed-Forward 層與殘差連接

每個注意力層之後，接一個**兩層的前饋網路（FFN）**，負責在每個位置做非線性變換，「消化」注意力收集來的資訊：

$$\text{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2$$

兩個穩定訓練的設計：
- **殘差連接（Residual Connection）**：`output = LayerNorm(x + Sublayer(x))`，讓梯度能直接流回早期層。
- **Layer Normalization**：對每層輸出做正規化，穩定訓練動態。

### 6. Encoder vs Decoder

原始 Transformer（Vaswani et al., 2017）為翻譯設計，分兩半：

| | Encoder | Decoder |
|---|---|---|
| **注意力方向** | 雙向（每個詞看所有詞） | 單向（只看左邊已生成的詞） |
| **用途** | 理解輸入 | 逐步生成輸出 |
| **代表模型** | BERT、RoBERTa | GPT 系列 |

現代大語言模型的架構選擇：
- **僅 Decoder**：GPT、Claude、LLaMA——適合自回歸生成
- **僅 Encoder**：BERT——適合分類、語義理解
- **完整 Encoder-Decoder**：T5、Bart——適合翻譯、摘要

---

## 跨領域連結

### 【同構】Attention ↔ 人類工作記憶

認知科學的**工作記憶（working memory）**不是把所有資訊都存起來，而是根據當前任務**動態激活最相關的記憶**——這跟注意力機制的設計幾乎完全同構。

人類工作記憶容量約 7±2 個「區塊」（Miller's Law），Transformer 的 context window 也是有限的，只是大得多。兩者都面對同一個根本問題：**你永遠無法在當下同時 attend 到所有事情**。

現在 AI 研究的 sparse attention、sliding window attention，其實正在模仿人類不均勻分配注意力的策略。

### 【時間錯位】Transformer 協議戰 ↔ 早期電力系統標準戰

1890 年代，愛迪生（直流電）vs 特斯拉（交流電）打了一場電力標準戰；現在 AI 社群也在打「模型協議戰」——MCP、OpenAPI、各家 agent 框架都在搶成為大語言模型的「插座規格」。兩場戰爭的核心邏輯一樣：技術優劣只是一部分，生態系鎖定才是決勝點。（→ 尚未建檔）

---

## 延伸追問

1. **為什麼 Transformer 這麼吃記憶體？** — Attention 的計算是 $O(n^2)$，序列長 2 倍，計算量暴增 4 倍。Flash Attention、Mamba 等架構都在解這個問題。
2. **BERT 和 GPT 的根本差異是什麼？** — 雙向理解 vs 單向生成，適合的任務完全不同。
3. **RoPE 怎麼運作，為什麼比原版位置編碼好？** — 旋轉矩陣的幾何直覺。
4. **Attention Head 真的在學語法、語意？** — Mechanistic Interpretability（可解釋性研究）的最新進展，Anthropic 有專門的研究團隊在做。
5. **有沒有不需要 Attention 的替代架構？** — Mamba（State Space Model）、RWKV 等，試圖用線性複雜度挑戰 Transformer 的主導地位。

---

## 參考資料

| 來源 | 時間 | 可信度 |
|---|---|---|
| Vaswani et al., *Attention Is All You Need* | 2017 | ⭐⭐⭐⭐⭐ 同行評審論文，Transformer 原始論文 |
| Anthropic, *Transformer Circuits Thread* | 2021–現在 | ⭐⭐⭐⭐⭐ 研究部落格，Mechanistic Interpretability 系列 |
| Illustrated Transformer（Jay Alammar） | 2018 | ⭐⭐⭐⭐ 技術部落格，視覺化解釋極佳 |

---

## 相關主題

- [傅立葉轉換](../../science/physics/fourier-transform.md) — 同樣是「把複雜的東西分解成一組基底」的思路，但在頻率域
- [小波轉換](../../science/physics/wavelet-transform.md) — 加上時間局部性的傅立葉，與 sliding window attention 的思路呼應
- MCP 協議（尚未建檔）— 跨領域連結中提到的「AI 模型協議戰」
- Mechanistic Interpretability（尚未建檔）— Attention Head 真正在學什麼
