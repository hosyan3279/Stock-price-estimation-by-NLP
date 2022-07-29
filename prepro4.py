import os
import numpy as np
import pandas as pd
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
import glob
import io
import processing_text
import csv
import datetime
import collections


def   convert_pdf_to_txt(path):
    for filename in os.listdir("C:\\Users\\p-user\\Desktop\\pythonProject1\\data"):

        if filename.endswith(".pdf"):
            rsrcmgr = PDFResourceManager()
            retstr = io.StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            fp = open(path, 'rb')
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()

            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                          check_extractable=False):
                interpreter.process_page(page)

            text = retstr.getvalue()

            fp.close()
            device.close()
            retstr.close()

            return text


if __name__ == "__main__":
    # 株価データの取得
    stock_files = glob.glob("C:\\Users\\p-user\\Desktop\\pythonProject1\\stockdata/*.csv")
    data_list_stock = []
    data_list = []
    # stockdataを一行ずつ読み込んでlistに格納
    for file in stock_files:
        with open(file, "r", encoding="shift_jis") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    date = datetime.datetime.strptime(row[0], "%Y/%m/%d")
                    binary = np.sign(int(float(row[5]) - float(row[8])))
                    data = [date, row[1], binary]
                    data_list_stock.append(data)
                except:
                    pass

    text_files = glob.glob("C:\\Users\\p-user\\Desktop\\pythonProject1\\data/*.pdf")
    k = 0
    i = len(data_list_stock)
    stopwords = []
    list_output = []
    for file in text_files:
        try:
            # 前処理もろもろ&ストップワード用辞書作成
            text = convert_pdf_to_txt(file)
            text = processing_text.clean_text(text)
            text = processing_text.normalize(text)
            text = processing_text.normalize_number(text)
            text = processing_text.wakati(text)
            list_output = text.split(" ")
            stopwords.extend(list_output)
            date = datetime.datetime.strptime(file[-20:-10], "%Y-%m-%d")
            code = file[-9:-5]
            print(date, code)
        except:
            pass

        for data in data_list_stock:
            try:
                if data[1] == code and data[0] == date:
                    data_list.append([date, code, list_output, data[2]])
            except:
                pass

    c = collections.Counter(stopwords)
    # stopwords上位n件のパラメータ↓
    c = dict(c.most_common(100))
    c = list(c.keys())
    data_list2 = []
    print(c)
    print(data_list)
    for i in range(len(data_list)):
        data_list[i][2] = [word for word in data_list[i][2] if word not in c]

    for data in data_list:
        print(data)

    corpus = pd.DataFrame(data_list, columns=["date", "code", "text", "label"])

    print(corpus)

    corpus.to_csv("C:\\Users\\p-user\\Desktop\\pythonProject1\\corpus.tsv", sep="\t", index=False, encoding="utf-8")

    # day_beforeは負がTrueで正と0がFalse
