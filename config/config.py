# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：YDS
@Date    ：2022/7/27 12:13 
"""

# Options: customize，demo, small, large
data_type = 'demo'  # 所采用的数据集类型

demo_data_path = '../data/raw_data/demo/'  # demo数据集路径 请勿更改
small_data_path = '../data/raw_data/small/'  # 请将下载并解压后的小型数据集所在路径复制到此
large_data_path = '../data/raw_data/large/'  # 请将下载并解压后的全量数据集所在路径复制到此
customize_data_path = '../data/raw_data/customize/'  # 自定义数据集路径 请勿更改

recall_save_path = '../data/new_data/recall/'  # 召回阶段涉及的数据文件保存路径
rank_save_path = '../data/new_data/rank/'  # 排序阶段涉及的数据文件保存路径

data_path = '../data/'  # 数据文件保存路径
log_path = '../data/user_log/'  # 每个用户的日志文件路径


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
