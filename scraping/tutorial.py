from pprint import pprint
import requests
from bs4 import BeautifulSoup

URL = 'https://briefly.ru/authors/'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

print(soup.find(class_='alphabetic-index').text)


