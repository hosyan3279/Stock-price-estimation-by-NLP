import time
import requests
import json
import random

def get_data(limit_size=300):
    url = "https://webapi.yanoshin.jp/webapi/tdnet/list/"

    condition = "20220305-20220422"
    format = "json"
    query = f"limit={str(limit_size)}"

    res = requests.get(f"{url}{condition}.{format}?{query}")

    response_loads = json.loads(res.content)

    data_list = [data["Tdnet"] for data in response_loads["items"]]
    return data_list


if __name__ == "__main__":
    data_list = get_data(1)
    data_list_url = []
    for data in data_list:
        data_list_url.append((data["pubdate"], data["company_code"], data["document_url"]))

    data_list_url_rev2 = []
    for data in data_list_url:
        if data[1][-1] == "0":
            data_list_url_rev2.append(data)

    print(len(data_list_url_rev2))

    for data in data_list_url_rev2:
        time.sleep(random.randint(6, 15))
        res = requests.get(data[2])
        date = data[0].split(" ")
        print(f"{date[0]}_{data[1]}")
        dri = f"C:\\Users\\p-user\\Desktop\\pythonProject1\\data2\\{date[0]}_{data[1]}.pdf"
        with open(f"{dri}", "wb") as f:
            f.write(res.content)
