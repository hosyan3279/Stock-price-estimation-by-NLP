import random
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit

from utils import plot_learning_curve


def cleaning_tokenize_split(text):
    return text


def train_and_eval():
    df = pd.read_csv("C:\\Users\\hiroyuki\\Desktop\\same\\copy\\train_test_data\\corpus.tsv", sep="\t")

    df.columns = ["date", "id", "text", "label"]
    df = df * 1
    text = df["text"]
    # corpusの整形
    text = text.str.replace("[", "")
    text = text.str.replace("]", "")
    print(df["label"])

    # labelの整形
    for i in range(len(df["label"])):
        if df["label"].iloc[i] == -1:
            df["label"][i] = 0

    # シャッフル
    df = df.sample(frac=1, random_state=1)

    X = df["text"].values
    Y = df["label"].values

    # べくとらいざー
    lowercase = False
    tokenize = None
    preprocessor = None

    vectorizer = TfidfVectorizer(max_features=1000, lowercase=lowercase, tokenizer=tokenize, preprocessor=preprocessor)

    X = vectorizer.fit_transform(X)

    # x_train_vec = vectorizer.fit_transform(x_train)
    # x_test_vec = vectorizer.fit_transform(x_test)

    x_train_vec, x_test_vec, y_train, y_test = train_test_split(X, Y,
                                                                test_size=0.2, random_state=1)

    x_train_vec = np.nan_to_num(np.array(x_train_vec.todense()))
    x_test_vec = np.nan_to_num(np.array(x_test_vec.todense()))
    y_train = np.nan_to_num(np.array(y_train))
    y_test = np.nan_to_num(np.array(y_test))

    print("  shape  \nx_train_vec x_test_vec \n", np.shape(x_train_vec), np.shape(x_test_vec))
    print("  shape  \ny_train y_test \n", np.shape(y_train), np.shape(y_test))
    print("x_train_vec x_test_vec \n", x_train_vec, x_test_vec)
    print("y_train y_test \n", y_train, y_test)

    print("  shape  \nx_train_vec x_test_vec \n", np.shape(x_train_vec), np.shape(x_test_vec))
    print("  shape  \ny_train y_test \n", np.shape(y_train), np.shape(y_test))
    print("x_train_vec x_test_vec \n", x_train_vec, x_test_vec)
    print("y_train y_test \n", y_train, y_test)

    # clf = LogisticRegression(solver="liblinear",penalty="l2",C=0.1)
    # clf.fit(x_train_vec, y_train)
    clf = svm.SVC()
    clf.fit(x_train_vec, y_train)
    y_predict = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_predict)
    print("accuracy_score " + "{:.4f}".format(score))

    # 学習曲線を描画する関数 utils.pyにある
    title = "learning curve"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(clf, title, x_train_vec, y_train, cv=cv)

    return 0


if __name__ == '__main__':
    train_and_eval()
