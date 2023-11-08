# -- coding: utf-8 --
# @Time : 2023/11/9 11:55
# @Author : 王川远
# @Email : 3030764269@qq.com
# @File : translate.py
# @Software: PyCharm
import random
import requests
from hashlib import md5
import os
import wget
import paddle
from PIL import Image
import HTML
import ftfy
import html

# 百度翻译服务访问
class Translator:
    def __init__(self):
        self.appid = '20210313000725566'
        self.appkey = 'anWY5DNo2Ab57bgmXnqR'
        self.url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.payload = {
            'appid': '20210313000725566',
            'from': 'zh',
            'to': 'en',
        }

    @staticmethod
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    def translate(self, query):
        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + query + str(salt) + self.appkey)

        self.payload['salt'] = salt
        self.payload['sign'] = sign
        self.payload['q'] = query
        r = requests.post(self.url, params=self.payload, headers=self.headers)
        result = r.json()['trans_result'][0]['dst']

        return result





# 检索引擎
class IMSP:
    def __init__(self, db_file=None):
        self.model, self.transforms = clip.load('ViT_B_32', pretrained=True)
        if db_file is None:
            db_file = 'image_db'
            db_url = 'https://bj.bcebos.com/v1/ai-studio-online/775e9601019646b2a09f717789a4602f069a26302f8643418ec7c2370b895da9?responseContentDisposition=attachment%3B%20filename%3Dimage_db'
            if not os.path.isfile(db_file):
                wget.download(db_url)
        self.image_features, self.photo_ids = self.load_db(db_file)
        self.translator = Translator()

    @staticmethod
    def load_db(db_file):
        image_db = paddle.load(db_file)

        image_features = image_db['image_features'].astype('float32')
        image_features = paddle.to_tensor(image_features)

        photo_ids = image_db['photo_ids']

        return image_features, photo_ids

    @staticmethod
    def get_urls(photo_ids):
        urls = []
        for photo_id in photo_ids:
            url = f"https://unsplash.com/photos/{photo_id}"
            urls.append(url)
        return urls

    @staticmethod
    def is_chinese(texts):
        return any('\u4e00' <= char <= '\u9fff' for char in texts)

    # 搜索图像，topk表示检索结果个数
    def im_search(self, texts, topk=5, return_urls=True):
        if self.is_chinese(texts):
            texts = self.translator.translate(texts)

        texts = tokenize(texts)
        with paddle.no_grad():
            text_features = self.model.encode_text(texts)

        logit_scale = self.model.logit_scale.exp()
        logits_per_text = logit_scale * text_features @ self.image_features.t()

        indexs = logits_per_text.topk(topk)[1][0]
        photo_ids = [self.photo_ids[index] for index in indexs]

        if return_urls:
            return self.get_urls(photo_ids)
        else:
            return photo_ids




def display_photo(photo_urls):
    for photo_url in photo_urls:
        photo_preview_url = photo_url+"/download?w=224"
        display(Image(url=photo_preview_url))
        display(HTML(f'原图请点击：<a target="_blank" href="{photo_url}">Unsplash Link</a>'))





# 实例化检索引擎
imsp_engine = IMSP()

photo_urls = imsp_engine.im_search('公交车', topk=5)
print(photo_urls)
display_photo(photo_urls)

photo_urls = imsp_engine.im_search('blue sky with cloud', topk=5)
print(photo_urls)

#显示结果结果
display_photo(photo_urls)