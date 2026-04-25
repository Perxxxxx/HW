import re
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
DATA_DIR = Path('../datasets')
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

REPEAT = 10
TEST_SIZE = 0.2

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """比老师的预处理更彻底:多了代码块剔除 + 词形还原"""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split()
              if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


def load_project(project_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'{project_name}.csv')
    df = df.sample(frac=1, random_state=999)
    df['text'] = (df['Title'].fillna('').astype(str) + ' '
                  + df['Body'].fillna('').astype(str)).apply(clean_text)
    df['label'] = df['class'].astype(int)
    return df[['text', 'label']]


def run_once(df: pd.DataFrame, seed: int) -> dict:
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=TEST_SIZE, random_state=seed
    )

    vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)

    n_pos = int((y_train == 1).sum())
    k = min(5, max(1, n_pos - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k)
    X_train_r, y_train_r = smote.fit_resample(X_train, y_train)

    clf = ComplementNB()
    clf.fit(X_train_r, y_train_r)
    y_pred = clf.predict(X_test)

    return {
        'accuracy':        accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro':    recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro':        f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_pos':   precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_pos':      recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1_pos':          f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }


def main():
    all_rows = []

    for proj in PROJECTS:
        print(f'\n>>> Running improved (aligned) on {proj} ...')
        df = load_project(proj)

        per_proj = []
        for seed in range(REPEAT):
            r = run_once(df, seed)
            r['project'] = proj
            r['repeat'] = seed
            all_rows.append(r)
            per_proj.append(r)

        sub = pd.DataFrame(per_proj)
        print(f"  Accuracy    : {sub['accuracy'].mean():.4f}")
        print(f"  F1 (macro)  : {sub['f1_macro'].mean():.4f}")
        print(f"  F1 (pos)    : {sub['f1_pos'].mean():.4f}")
        print(f"  Recall(pos) : {sub['recall_pos'].mean():.4f}")

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / 'improved_aligned_results.csv', index=False)
    print(f'\n完整结果已保存到 {RESULTS_DIR / "improved_aligned_results.csv"}')


if __name__ == '__main__':
    main()