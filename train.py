import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

df3 = pd.read_csv("C:\\Users\\p-user\\Desktop\\pythonProject1\\testdata\\test_corpus.tsv",
                  sep=",", encoding="cp932")
df = pd.read_csv("C:\\Users\\p-user\\Desktop\\pythonProject1\\testdata\\text.tsv")
# dfに列名を付ける
df.columns = ["date", "id", "text", "label"]
df = df * 1
text = df["text"]
df = df.sample(frac=1, random_state=1)  # シャッフル
grouped = df.groupby("label")
df = grouped.head(n=4000)


x_train, x_test, y_train, y_test = train_test_split(df["text"], df["label"],
                                                    test_size=0.2, random_state=1)

print(np.shape(x_train), np.shape(x_test))
print(y_train)

lowercase = False
tokenize = None
preprocessor = None

vectorizer = TfidfVectorizer()

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.fit_transform(x_test)

clf = LogisticRegression(solver="liblinear")
clf.fit(x_train_vec, y_train)

# clf = svm.SVC()
# clf.fit(x_train_vec, y_train)

y_pred = clf.predict(x_test_vec)
score = accuracy_score(y_test, y_pred)
print("{:.2f}".format(score))
