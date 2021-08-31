import random
import requests
from bs4 import BeautifulSoup
from tools.utils import normalize, remove_soup_content
import time

class VNExpressCrawler:

    HOMEPAGE = "https://vnexpress.net"
    CATEGORIES = [
        "thoi-su",
        "the-gioi",
        "kinh-doanh",
        "khoa-hoc",
        "giai-tri",
        "the-thao",
        "phap-luat",
        "giao-duc",
        "suc-khoe",
        "doi-song",
        "du-lich",
        "so-hoa",
        "oto-xe-may",
        "y-kien",
        "hai"
    ]

    def __init__(self, sess: requests.Session = None, category: str = None):
        self.sess = sess if sess else requests.Session()
        self.category = category
    
    @classmethod
    def from_category(cls, category, max_page_numb=512):
        print(f"Crawling {category} ...")
        t0 = time.time()
        sess = requests.Session()
        res = sess.get(f"{cls.HOMEPAGE}/{category}")
        assert res.status_code == 200, f"The category {category} does not exist."
        data = cls(sess, category).scan(max_page_numb)
        t1 = time.time()
        print(f"    - Category {category} is collected in {t1-t0:.3f}s")
        return data
    
    def read_content(self, url: str):
        res = self.sess.get(url, timeout=2.0)
        if res.status_code != 200:
            return
        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.find(class_="fck_detail")
        remove_soup_content(content, class_="author")
        remove_soup_content(content, class_="Image")
        remove_soup_content(content, class_="hidden")
        remove_soup_content(content, class_="list-news")
        remove_soup_content(content, class_="item_slide_show")
        remove_soup_content(content, class_="wrap_video")
        remove_soup_content(content, "strong")
        remove_soup_content(content, "em")
        remove_soup_content(content, "figure")
        if content is None:
            return
        for p in content.find_all("p"):
            p.replace_with(p.text)
        text = "\n".join(content.find_all(text=True))
        content = normalize(text)
        return content
    
    def read_page(self, page_numb):
        results = {}
        try:
            res = self.sess.get(f"{self.HOMEPAGE}/{self.category}-p{page_numb}")
            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.find_all("article", class_="item-news")
            assert articles is not None
        except:
            return
        for article in articles:
            try:
                if random.random() > 0.4:
                    continue
                url = article.find(class_="title-news").find("a")["href"]
                title = normalize(article.find(class_="title-news").find("a").text)
                description = article.find(class_="description").find("a")
                remove_soup_content(description, "span")
                description = normalize(description.text)
                results[url] = (self.category, f"p{page_numb}", title, description)
            except:
                pass
        return results

    def scan(self, max_articles, max_page_numb=1024, max_skip=10):
        articles = {}
        n_skip = 0
        len_articles = 0
        while len_articles < max_articles:
            page_numb = random.randint(1, max_page_numb)
            page_data = self.read_page(page_numb)
            if page_data is not None and len(page_data) > 0:
                articles.update(page_data)
                if len(articles) > len_articles:
                    n_skip = 0
                    len_articles = len(articles)
            else:
                n_skip += 1
                if n_skip > max_skip:
                    break
                print(f"Skip crawling {self.category}-p{page_numb} ... {n_skip}/{max_skip}")
        return articles