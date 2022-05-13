import os
import pandas as pd
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from janome.tokenizer import Tokenizer
import glob
import io
import MeCab
import clean_text
import csv
import datetime
import numpy as np

columns = ["date", "code", "text", "stack_binary"]
corpus = pd.DataFrame(columns=columns)
t = Tokenizer(wakati=True)
m = MeCab.Tagger("-Owakati")


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


def tokenize(text):  # janome
    return t.tokenize(text)


if __name__ == "__main__":
    stack_files = glob.glob("/stockdata/*.csv")
    data_list_stack = []

    for file in stack_files:
        with open(file, "r", encoding="shift_jis") as f:
            reader = csv.reader(f)
            for row in reader:
                data_list_stack.append(row)
    print(len(data_list_stack))
    files = glob.glob("C:\\Users\\p-user\\Desktop\\pythonProject1\\data/*.pdf")
    corpus = []

    for file in files:

        text = convert_pdf_to_txt(file)
        text = clean_text.clean_text(text)
        text = clean_text.normalize(text)
        text = clean_text.normalize_number(text)
        text = m.parse(text)

        date1 = datetime.datetime.strptime(file[-20:-10], "%Y-%m-%d")
        code1 = file[-9:-5]

        corpus.append([date1, code1, text, 0])
        for data_stack in data_list_stack:
            # print(data_stack[0], data_stack[1])
            try:
                date2 = datetime.datetime.strptime(data_stack[0], "%Y/%m/%d")
                date3 = date2 + datetime.timedelta(days=-1)
            except:
                pass
            for data_stack2 in data_list_stack:
                try:  # こいつをどうにかする↓
                    date = datetime.datetime.strptime(data_stack2[0], "%Y/%m/%d")
                    if date3 == date and code1 == data_stack[1]:
                        print("S")
                        stack_yesterday = data_stack[8]
                except:
                    pass

            try:
                if date1 == date2 and code1 == data_stack[1]:
                    print(date1, date2, code1, data_stack[1])
                    stack_today = data_stack[8]
                    stack3 = int(stack_today) - int(stack_yesterday)
                    print(np.sign(stack3))
                    corpus.append([date1, code1, text, np.sign(stack3)])
            except:
                pass

    # print(len(corpus))

    corpus = pd.DataFrame(corpus)

    corpus.to_csv("C:\\Users\\p-user\\Desktop\\pythonProject1\\testdata\\corpus.csv", index=False)
