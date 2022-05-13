from bs4 import BeautifulSoup
import string
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import urllib.request as req
import urllib
import os
import time
from urllib.parse import urljoin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from downloader import get_data


# 簡易機械学習よう Tf-idf


def train_and_eval(x_train, y_train, x_test, y_test, vectorizer):
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))


def filter_by_ascii_rate(text, threshold=0.9):
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold


def load_dataset(filename, n=5000, state=6):
    df = pd.read_csv(filename, sep='\t')

    # マルチクラスを二値にするやつ
    mapping = {1: 0, 2: 0, 4: 1, 5: 1}
    df = df[df.star_rating != 3]
    df.star_rating = df.star_rating.map(mapping)

    # 日本語抽出するやつ
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # サンプリング
    df = df.sample(frac=1, random_state=state)  # シャッフル
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n)
    return df.review_body.values, df.star_rating.values
