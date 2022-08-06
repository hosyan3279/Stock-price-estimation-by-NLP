from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#from utils import load_dataset


def main():
    # まだ
    x, y = load_dataset('###############', n=5000)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # ベクトラいざー
    vectorizer = CountVectorizer(tokenizer=tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    print(x_train.shape)
    print(x_test.shape)

    # 特徴量選択
    print('Selecting features...')
    selector = SelectKBest(k=7000, score_func=mutual_info_classif)
    # selector = SelectKBest(k=7000)
    selector.fit(x_train, y_train)
    x_train_new = selector.transform(x_train)
    x_test_new = selector.transform(x_test)
    print(x_train_new.shape)
    print(x_test_new.shape)

    # 評価
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_new, y_train)
    y_pred = clf.predict(x_test_new)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))


if __name__ == '__main__':
    main()
