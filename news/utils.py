from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
from PIL import Image
import io
import nltk
nltk.download('punkt')

def fetch_rss(url):
    op = urlopen(url)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    return sp_page.find_all('item')

def fetch_news(topic="stocks"):
    return fetch_rss(f'https://news.google.com/rss/search?q={topic}')

def get_summary(link):
    try:
        news = Article(link)
        news.download()
        news.parse()
        news.nlp()
        return news.title, news.summary, news.top_image
    except:
        return None, "Summary not available", "/static/images/no_image.jpg"