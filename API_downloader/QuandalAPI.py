import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

quandl.ApiConfig.api_key = "jtfUsP_kX1yTxwJzcBMZ"
data_list = pd.read_csv("C:\\Users\\hiroyuki\\Desktop\\same\copy\\train_test_data\\test_corpus.tsv", sep="\t")
data_list_code = data_list["code"]
for i in range(len(data_list_code)):
    get_data = "TES/" + str(data_list_code[i])
    sleep_time = np.random.randint(6, 15)
    quandl_data = quandl.get(get_data, start_date="2019-01-01", end_date="2019-12-31")
    time.sleep(sleep_time)
    # グラフを描画
    plt.plot(quandl_data["Close"])
    plt.show()

