# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：run_x_deep_fm.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/27 21:39 
"""
import time

from rank.xDeepFM.x_deep_fm import run_train


def train_deepctr():
    run_train()


# def predict_deepctr():
#     run_predict_deepctr()


def main():
    train_deepctr()  # xDeepFM排序模型训练
    # predict_deepctr()  # xDeepFM排序预测


if __name__ == '__main__':
    t = time.time()
    main()
    print('耗时: ', time.time() - t)
