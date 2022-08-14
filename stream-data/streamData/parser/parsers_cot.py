import requests
from bs4 import BeautifulSoup

URL = 'https://www.cftc.gov/dea/futures/financial_lf.htm'
HEADERS = {'accept':
           'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,\
        image/webp,image/apng,*/*;q=0.8,application/signed-exchange;\
        v=b3;q=0.9', 'user-agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, \
        like Gecko) Chrome/86.0.4240.111 Safari/537.36'}


def get_html(url, params=None):
    return requests.get(url, headers=HEADERS, params=params)


def get_data_from_html(URL):
    html = get_html(URL)
    soup = BeautifulSoup(html.text, 'html.parser')
    items = soup.find('pre')
    return items.text.split(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------"
    )


def parsing_data(data):
    bitcoin_data = [i for i in data if i.find('BITCOIN') != -1]
    bitcoin_data = bitcoin_data[0].split('\r')
    bitcoin_data = [i.split('\n') for i in bitcoin_data]
    bitcoin_data = [[s for s in i if s not in ["", ' ']]
                    for i in bitcoin_data]
    bitcoin_data = list(filter(None, bitcoin_data))
    bitcoin_data = [[" ".join(s.split()) for s in i]for i in bitcoin_data]
    bitcoin_data = bitcoin_data[1:]
    return bitcoin_data


def processing_data(bitcoin_data):
    positions_name_columns = ['Dealer long', 'Dealer short', 'Dealer spreading', 'Institutional long', 'Institutional short', 'Institutional spreading',
                              'Leveraged long', 'Leveraged short', 'Leveraged spreading', 'Other long', 'Other short', 'Other spreading', 'Nonreportable long', 'Nonreportable short']
    data = []
    for i in range(len(bitcoin_data)):
        if bitcoin_data[i][0].startswith('Positions'):
            temp = {
                'name': 'Positions',
                'open interest':  bitcoin_data[i-1][0].split()[-1],
            }
            temp.update(
                dict(zip(positions_name_columns[:-2], bitcoin_data[i+1][0].split())))
            data.append(temp)

        if bitcoin_data[i][0].startswith('Changes'):
            temp = {
                'name': 'Changes',
                'open interest':  bitcoin_data[i][0].split()[-1],
            }
            temp.update(
                dict(zip(positions_name_columns, bitcoin_data[i+1][0].split())))
            data.append(temp)

        if bitcoin_data[i][0].startswith('Percent of Open Interest'):
            temp = {
                'name': 'Open Interest',
                'open interest':  bitcoin_data[i][0].split()[-1],
            }
            temp.update(
                dict(zip(positions_name_columns, bitcoin_data[i+1][0].split())))
            data.append(temp)

        if bitcoin_data[i][0].startswith('Number of Traders'):
            temp = {
                'name': 'Traders',
                'open interest':  bitcoin_data[i][0].split()[-1],
            }
            temp.update(
                dict(zip(positions_name_columns[:-2], bitcoin_data[i+1][0].split())))
            data.append(temp)
    return data


def get_cot_data():
    data = get_data_from_html(URL)
    data = parsing_data(data)
    return processing_data(data)


if __name__ == '__main__':
    get_cot_data()
