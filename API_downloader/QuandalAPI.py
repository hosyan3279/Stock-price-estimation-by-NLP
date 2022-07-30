import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def get_data(code, start_date, end_date):
    try:

        data = quandl.get(code,date = { "gte": f"start_date", "lte": f"end_date" })
        sleep_time = np.random.randint(6, 15)
        time.sleep(sleep_time)
        print(data)
    except:
        print(f"{code} is not found")
        data = None
        pass

    return data


quandl.ApiConfig.api_key = "jtfUsP_kX1yTxwJzcBMZ"
data_list = pd.read_csv("C:\\Users\\hiroyuki\\Desktop\\same\copy\\train_test_data\\test_corpus.tsv", sep="\t")
data_list_code = data_list["code"]

for i in range(len(data_list_code)):
    code = "TSE/" + str(data_list_code[i])
    start_date = "2019-01-01"
    end_date = "2020-01-01"
    print(code, start_date, end_date)

    get_data = get_data(code, start_date, end_date)
    get_data.to_csv("C:\\Users\\hiroyuki\\Desktop\\same\copy\\train_test_data\\test_corpus_data.tsv", sep="\t")
"""
    if get_data is not None:
        data_list["data"][i] = get_data
        print(f"{code} is downloaded")
    else:
        print(f"{code} is not downloaded")
        data_list["data"][i] = None
        pass
"""

