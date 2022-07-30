import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("C:\\Users\\hiroyuki\\Desktop\\same\\copy\\corpus.tsv", sep="\t")

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
#print(df["label"])
#print(df["text"])

# シャッフル
df = df.sample(frac=1, random_state=1)
# grouped = df.groupby("label")
# df = grouped.head(n=4000)
X = df["text"].values
Y = df["label"].values

x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=1)

print("x_train x_test \n", np.shape(x_train), np.shape(x_test))
print("y_train y_test \n", np.shape(y_train), np.shape(y_test))

print("x\n",x_train, x_test)
print("y\n",y_train, y_test)


lowercase = False
tokenize = None
preprocessor = None

vectorizer = TfidfVectorizer()

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.fit_transform(x_test)

x_train_vec = np.nan_to_num(np.array(x_train_vec.todense()))
x_test_vec = np.nan_to_num(np.array(x_test_vec.todense()))
y_train = np.nan_to_num(np.array(y_train))
y_test = np.nan_to_num(np.array(y_test))

print("  shape  \nx_train_vec x_test_vec \n", np.shape(x_train_vec), np.shape(x_test_vec))
print("  shape  \ny_train y_test \n", np.shape(y_train), np.shape(y_test))
print("x_train_vec x_test_vec \n", x_train_vec, x_test_vec)
print("y_train y_test \n", y_train, y_test)


clf = LogisticRegression(solver="liblinear")
clf.fit(x_train_vec, y_train)
# clf = svm.SVC()
# clf.fit(x_train_vec, y_train)
y_predict = clf.predict(x_test_vec)
score = accuracy_score(y_test, y_predict)
print("{:.2f}".format(score))
