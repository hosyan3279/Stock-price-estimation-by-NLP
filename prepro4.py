import os
import numpy
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


def convert_pdf_to_txt(path):
    for filename in os.listdir("C:\\Users\\p-user\\Desktop\\pythonProject1\\testdata"):
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
    stack_files = glob.glob("C:\\Users\\p-user\\Desktop\\pythonProject1\\stockdata/*.csv")
    data_list_stock = []
    data_list = []
    # stockdataを一行ずつ読み込んでlistに格納
    for file in stack_files:
        with open(file, "r", encoding="shift_jis") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    date = datetime.datetime.strptime(row[0], "%Y/%m/%d")
                    binary = numpy.sign(int(float(row[5]) - float(row[8])))
                    data = [date, row[1], binary]
                    data_list_stock.append(data)
                except:
                    pass

        # print(data)

    text_files = glob.glob("C:\\Users\\p-user\\Desktop\\pythonProject1\\testdata/*.pdf")
    k = 0
    i = len(data_list_stock)

    for file in text_files:
        # 前処理もろもろ
        text = convert_pdf_to_txt(file)
        text = processing_text.clean_text(text)
        text = processing_text.normalize(text)
        text = processing_text.normalize_number(text)
        text = processing_text.wakati(text)
        date = datetime.datetime.strptime(file[-20:-10], "%Y-%m-%d")
        code = file[-9:-5]
        print(date, code)

        for data in data_list_stock:
            try:
                # 銘柄コードと日付が一致するものを検索
                if data[1] == code and data[0] == date:
                    data_list.append([date, code, text, data[2]])
                    print('{:.2f}'.format(100 * (k / i)) + "%")
                    print(k)
                    k += 1
            except:
                pass

    corpus = pd.DataFrame(data_list, columns=["date", "code", "text", "day_before"])
    print(corpus["text"])
    dictionary = processing_text.create_dictionary(corpus["text"])
    print(dictionary)

    stop_words = processing_text.get_stop_words(dictionary)

    for data in corpus:
        corpus = processing_text.remove_stopwords(corpus["text"])

    print(corpus)

    print("end")
