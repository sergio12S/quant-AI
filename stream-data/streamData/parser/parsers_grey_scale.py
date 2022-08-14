import requests
from bs4 import BeautifulSoup


URL = 'https://www.bybt.com/Grayscale'
HEADERS = {'accept':
           'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,\
        image/webp,image/apng,*/*;q=0.8,application/signed-exchange;\
        v=b3;q=0.9', 'user-agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, \
        like Gecko) Chrome/86.0.4240.111 Safari/537.36'}


def processing_data(items):
    try:
        headers = [i.text.strip() for i in items.find_all('th')]
        headers[1] = "Total Holdings"
        body = items.find_all('tbody')[0].find_all('tr')
        rows = [i.text.split() for i in [b for b in body]]

        data = [mutation_indexes(i) for i in rows]
        return [{header: b[i::len(headers)][0]
                 for i, header in enumerate(headers)} for b in data]
    except Exception:
        return None


def mutation_indexes(data):
    data = data[:]
    if len(data) == 14:
        del data[1]
        data[11] = data[11] + f'-{data[12]}'
        del data[12]
    elif len(data) == 13:
        data[11] = data[11] + f'-{data[12]}'
        del data[12]
    data[1] = data[1] + f' {data[2]}'
    del data[2]
    return data


def get_html(url, params=None):
    return requests.get(url, headers=HEADERS, params=params)


def pase_grey_scale():
    html = get_html(URL)
    soup = BeautifulSoup(html.text, 'html.parser')
    items = soup.find('div', class_='ivu-table ivu-table-default')
    return processing_data(items)


if __name__ == '__main__':
    pase_grey_scale()
