# -- coding: utf-8 --
# @Time : 2023/11/10 9:22
# @Author : 王川远
# @Email : 3030764269@qq.com
# @File : get_des_from_txt.py
# @Software: PyCharm
import os

import yaml
import json
from operator import itemgetter

from typing import Dict

fd = open('./des_from_txt.yaml', 'r')
content_yaml = yaml.load(fd, Loader=yaml.FullLoader)
fd.close()

def get_des() -> Dict[str, str]:
    """get pic_no:pic_des dictionary from des_from_txt.yaml"""
    fd = open(content_yaml['dir'], encoding='utf-8', mode='r')
    content_txt = fd.read()
    fd.close()
    print(type(content_txt))
    content_txt = content_txt.replace('\n', '')
    content_json = json.loads(content_txt)
    # print(type(list(content_json.values())))
    # content_list = list(content_json.values())
    content_list = map(itemgetter(0), content_json.values())
    for key in content_json.keys():
        content_json[key] = next(content_list)
    print(content_json)
    return content_json


def get_img():
    dir=content_yaml['dir_img']
    img_list=[]
    for file in os.listdir(dir):
        print(file)
        if file.endswith('.png'):
            img_list.append(os.path.join(dir,file))
    print(img_list)
    return img_list
if __name__ == "__main__":
    #get_des()
    get_img()
