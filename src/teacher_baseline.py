import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.naive_bayes import GaussianNB

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
DATA_DIR = Path('../datasets')
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

REPEAT = 10
STOPWORDS = stopwords.words('english') + ['...']


def remove_html(t): return re.compile(r'<.*?>').sub('', str(t))
def remove_emoji(t):
    p = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                   "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
                   "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
                   flags=re.UNICODE)
    return p.sub('', str(t))
def remove_stopwords(t): return ' '.join(w for w in str(t).split() if w not in STOPWORDS)
def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[\\\"\']", "", s)
    return s.strip().lower()


def run_project(project_name: str) -> list:
    df = pd.read_csv(DATA_DIR / f'{project_name}.csv')
    df = df.sample(frac=1, random_state=999)

    df['text'] = df.apply(
        lambda r: (str(r['Title']) + '. ' + str(r['Body'])) if pd.notna(r['Body']) else str(r['Title']),
        axis=1
    )
    df['sentiment'] = df['class']

    for fn in (remove_html, remove_emoji, remove_stopwords, clean_str):
        df['text'] = df['text'].apply(fn)

    params = {'var_smoothing': np.logspace(-12, 0, 13)}
    rows = []

    for seed in range(REPEAT):
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        X_tr_txt, X_te_txt = df['text'].iloc[tr], df['text'].iloc[te]
        y_tr, y_te = df['sentiment'].iloc[tr], df['sentiment'].iloc[te]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X_tr = tfidf.fit_transform(X_tr_txt).toarray()
        X_te = tfidf.transform(X_te_txt).toarray()

        clf = GaussianNB()
        grid = GridSearchCV(clf, params, cv=5, scoring='roc_auc')
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_
        best.fit(X_tr, y_tr)
        y_pred = best.predict(X_te)

        fpr, tpr, _ = roc_curve(y_te, y_pred, pos_label=1)

        rows.append({
            'project': project_name,
            'repeat': seed,
            'accuracy':        accuracy_score(y_te, y_pred),
            'precision_macro': precision_score(y_te, y_pred, average='macro', zero_division=0),
            'recall_macro':    recall_score(y_te, y_pred, average='macro', zero_division=0),
            'f1_macro':        f1_score(y_te, y_pred, average='macro', zero_division=0),
            'precision_pos':   precision_score(y_te, y_pred, pos_label=1, zero_division=0),
            'recall_pos':      recall_score(y_te, y_pred, pos_label=1, zero_division=0),
            'f1_pos':          f1_score(y_te, y_pred, pos_label=1, zero_division=0),
            'auc':             auc(fpr, tpr),
        })
    return rows


def main():
    all_rows = []
    for proj in PROJECTS:
        print(f'\n>>> Running teacher baseline v2 on {proj} ...')
        rows = run_project(proj)
        all_rows.extend(rows)
        sub = pd.DataFrame(rows)
        print(f"  F1 (macro) : {sub['f1_macro'].mean():.4f}")
        print(f"  F1 (pos)   : {sub['f1_pos'].mean():.4f}")
        print(f"  Recall(pos): {sub['recall_pos'].mean():.4f}")

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / 'teacher_baseline_detailed.csv', index=False)
    print(f'\n完整结果已保存')


if __name__ == '__main__':
    main()