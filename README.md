# PAD_Sentiment_Analyzer
A robust sentiment analysis tool based on the Pleasure-Arousal-Dominance (PAD) model. It integrates the ANEW lexicon with spaCy/Word2Vec vector embedding to effectively handle Out-of-Vocabulary (OOV) words using cosine similarity and weighted interpolation.

# PAD_Sentiment_Analyzer：基於 P-A-D 模型的情緒分析工具

## 專案概述

這個 Python 專案提供了一個強大且魯棒的情緒分析工具，用於評估英文文本的 **Pleasure-Arousal-Dominance (P-A-D) 維度情緒**。

**核心優勢：**
* **整合 ANEW 詞典**：專案內部已包含 **ANEW (Affective Norms for English Words)** 詞典作為情緒基準。
* **Word2Vec robustic**：結合 **spaCy** 提供的 Word2Vec 詞向量技術，能有效處理詞典中不存在的單字 (OOV)，從而提高分析的準確性和覆蓋範圍。

---

## P-A-D 情緒模型詳解 (Pleasure-Arousal-Dominance)

P-A-D 模型（也稱為**維度情緒模型**）是心理學中一個廣泛使用的框架，它將任何情緒描述為三個基本、獨立的維度組合，實現了比簡單「正面/負面」分類更細緻的描述。

| 維度 | 意義 | 功能描述 |
| :--- | :--- | :--- |
| **Pleasure (P) / 愉悅度 (價價, Valence)** | 衡量情感的**正負極性**或好壞程度。 | 評估文本傳達的快樂、滿意或悲傷、沮喪程度。 |
| **Arousal (A) / 活躍度** | 衡量情感反應的**強度**或生理活躍程度。 | 評估文本的興奮、緊張（高 A）或平靜、無聊（低 A）。 |
| **Dominance (D) / 支配度** | 衡量個體在情境中感受到的**控制感**或主導權。 | 評估文本所暗示的無助感（低 D）或權力感（高 D）。 |

這三個維度共同創建了一個**三維情緒空間**，能精確定位複雜的情緒狀態。

---

## 技術核心：OOV 單字處理

本專案解決了詞典方法常見的 **OOV (Out-Of-Vocabulary) 問題**。當分析文本中出現 ANEW 詞典裡沒有的單字時，程式碼會執行以下步驟進行分數估計：

1.  **取得 OOV 向量與詞性**：使用 **spaCy (en\_core\_web\_md)** 取得 OOV 單字的詞向量和詞性 (POS)。
2.  **餘弦相似度計算**：計算 OOV 向量與詞典中所有詞向量的**餘弦相似度**。
3.  **相似度加權**：只保留相似度**高於 $0.5$** 且詞性相符的詞典單字。
4.  **分數內插**：使用這些相似單字的**情緒分數**，並以它們的**相似度作為權重**，計算出 OOV 單字的**加權平均 P-A-D 分數**。

這種方法有效地將**語義知識**整合到情緒評估中，提高了分析的準確性。

---

## 安裝與設定

### 1. 必備環境

您需要安裝 Python (建議 3.8+)。

### 2. 安裝 Python 套件

使用 `pip` 安裝所需的函式庫：

```bash
pip install pandas spacy scipy
```

### 3. 下載 spaCy 詞向量模型
本專案依賴於 en_core_web_md 模型來獲取單字的詞向量和詞性標註：

```bash
python -m spacy download en_core_web_md
```

### 4. 詞典檔案
注意： ANEW 詞典檔案 ('ANEW_with_POS.csv') 已包含在您的 GitHub Repository 中，無需額外下載或設定。請確保主程式碼中 'load_anew_with_pos' 函式的參數與實際檔案名稱匹配。
