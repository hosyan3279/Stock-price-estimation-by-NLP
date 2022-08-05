"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['Species'] = pd.DataFrame(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']], iris_df['Species'], test_size=0.2, random_state=0)
logit2 = LogisticRegression()
logit2.fit(X_train, y_train)

#偏回帰係数
print(logit2.coef_)
print("\n")
#切片
print(logit2.intercept_)
print("\n")
#予測
y_pred = logit2.predict(X_test)
print(y_pred)
print("\n")
#混合行列
print(confusion_matrix(y_test, y_pred))
print("\n")
#正解率
print(accuracy_score(y_test, y_pred))
print("\n")
#適合率
print(precision_score(y_test, y_pred))
print("\n")
"""

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import random

df = pd.read_csv("C:\\Users\\hiroyuki\\Desktop\\same\\copy\\train_test_data\\test_corpus.tsv", sep="\t")

df.columns = ["date", "id", "text", "label"]
df = df * 1
text = df["text"]
# corpusの整形
text = text.str.replace("[", "")
text = text.str.replace("]", "")
print(df["label"])

# labelの整形
# 各データの特徴数を揃える
for i in range(len(df["label"])):
    if df["label"].iloc[i] == -1:
        df["label"][i] = 0

for k in range(len(df["text"])):
    print(tuple(random.sample(df["text"][k], 14)))
    df["text"][k] = tuple(random.sample(df["text"][k], 14))
0
print(df)
# print(df["label"])
# print(df["text"])

# シャッフル
df = df.sample(frac=1, random_state=1)
# grouped = df.groupby("label")
# df = grouped.head(n=4000)
X = df["text"].values
Y = df["label"].values

manual_data = pd.DataFrame(X, Y, columns=["text", "label"])
print(manual_data)
manual_data.to_csv("C:\\Users\\hiroyuki\\Desktop\\same\\copy\\train_test_data\\manual_data.tsv", sep="\t")
