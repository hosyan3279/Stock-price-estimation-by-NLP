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