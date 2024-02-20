from bs4 import BeautifulSoup
import requests
from .base_parser import BaseParser


class HTMLParser(BaseParser):

    def parse(self, url: str) -> None:
        response = requests.get(url)
        htmlContent = response.text
        soup = BeautifulSoup(htmlContent, 'lxml')
        text = soup.get_text()
        documents = text.split('\n\n')
        for doc in documents:
            self._save_doc(doc)

parser = HTMLParser()
parser.parse("https://lti.cs.cmu.edu/learn")
