import pandas as pd
import re
import spacy
from scipy.spatial.distance import cosine

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("請先執行：python -m spacy download en_core_web_md")
    exit()

def load_anew_with_pos(file_path):
    """
    從 CSV 檔案載入 ANEW 詞典，包含已標註的 POS (詞性)。
    """
    try:
        df = pd.read_csv(file_path)
        anew_lexicon = {}
        for index, row in df.iterrows():
            word = str(row['Description']).lower()
            doc = nlp(word)
            if doc and doc[0].has_vector:
                anew_lexicon[word] = {
                    'valence': row['Valence_Mean'],
                    'arousal': row['Arousal_Mean'],
                    'dominance': row['Dominance_Mean'],
                    'pos': row['POS'],
                    'vector': doc[0].vector
                }
        return anew_lexicon
    except FileNotFoundError:
        print(f"錯誤：找不到檔案在 {file_path}")
        return None

def get_oov_score(oov_word, lexicon):
    """
    針對字典中不存在的單字，使用 Word2Vec 相似度進行分數內插。
    """
    doc = nlp(oov_word.lower())
    if not doc or not doc[0].has_vector:
        return None, None, None

    oov_token = doc[0]
    scores = []

    for word_data in lexicon.values():
        if word_data['pos'] == oov_token.pos_ and word_data['vector'] is not None:
            similarity = 1 - cosine(oov_token.vector, word_data['vector'])

            if similarity > 0.5:
                scores.append({
                    'similarity': similarity,
                    'valence': word_data['valence'],
                    'arousal': word_data['arousal'],
                    'dominance': word_data['dominance']
                })

    if not scores:
        return None, None, None

    total_similarity = sum(s['similarity'] for s in scores)
    if total_similarity == 0:
        return None, None, None

    avg_valence = sum(s['valence'] * s['similarity'] for s in scores) / total_similarity
    avg_arousal = sum(s['arousal'] * s['similarity'] for s in scores) / total_similarity
    avg_dominance = sum(s['dominance'] * s['similarity'] for s in scores) / total_similarity

    return avg_valence, avg_arousal, avg_dominance

def analyze_text_with_anew(text, lexicon):
    """
    對文本進行 P-A-D 情緒分析，並處理字典中不存在的單字。
    """
    words = re.findall(r'\b\w+\b', text.lower())

    valence_scores = []
    arousal_scores = []
    dominance_scores = []

    for word in words:
        if word in lexicon:
            data = lexicon[word]
            valence_scores.append(data['valence'])
            arousal_scores.append(data['arousal'])
            dominance_scores.append(data['dominance'])
        else:
            valence, arousal, dominance = get_oov_score(word, lexicon)
            if valence is not None:
                valence_scores.append(valence)
                arousal_scores.append(arousal)
                dominance_scores.append(dominance)

    if not valence_scores:
        return 0, 0, 0

    avg_valence = sum(valence_scores) / len(valence_scores)
    avg_arousal = sum(arousal_scores) / len(arousal_scores)
    avg_dominance = sum(dominance_scores) / len(dominance_scores)

    return avg_valence, avg_arousal, avg_dominance

if __name__ == '__main__':
    anew_lexicon_with_pos = load_anew_with_pos('ANEW_with_POS.csv')

    if anew_lexicon_with_pos:
        sample_text = "A sudden noise made me feel uncertain and helpless."
        valence, arousal, dominance = analyze_text_with_anew(sample_text, anew_lexicon_with_pos)

        print("\n分析結果：")
        print(f"文本的平均愉悅度 (Pleasure): {valence}")
        print(f"文本的平均激活度 (Arousal): {arousal}")
        print(f"文本的平均支配度 (Dominance): {dominance}")
