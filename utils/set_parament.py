# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：set_parament.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/28 20:18 
"""
import yaml
from collections import namedtuple


def get_args(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        para_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        ps = namedtuple('parser', list(para_dict.keys()))
        args = ps(**para_dict)
        f.close()

    return args


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
