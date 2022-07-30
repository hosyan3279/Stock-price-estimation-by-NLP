import quandl
import pandas as pd
import numpy as np

quandl.ApiConfig.api_key = "jtfUsP_kX1yTxwJzcBMZ"
data = quandl.get('WIKI/KO')
print(data)
