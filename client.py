import requests


if __name__ == '__main__':
    url = f'http://127.0.0.1:8018/tableRec'

    filename = r'test/1/1.jpg'

    file = {"file": open(filename, 'rb')}
    headers = {'Content-Type': 'multipart/form-data'}
    r = requests.post(url, files=file)
    result = r.json()
    print(result)