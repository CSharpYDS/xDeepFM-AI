# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：create_feat.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/27 21:06 
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from config.config import data_type
from rank.xDeepFM.feature_colum import SparseFeat, DenseFeat


def create_ctr_data(path, args, use_dict=True):
    """
    构造排序模型的输入数据和特征
    :param path:
    :return:
    """
    with open(os.path.join(path, data_type + '_rank_data.pkl'), 'rb') as f:
        all_data, feature_info = pickle.load(f)
        f.close()

    # 训练数据和测试数据
    all_data = shuffle(all_data)
    train_df = all_data[all_data['whether_to_click'] != -1]
    test_df = all_data[all_data['whether_to_click'] == -1]
    # 测试数据的标签
    test_labels = pd.read_pickle(os.path.join(path, data_type + '_rank_test_label.pkl'))
    test_labels = pd.merge(test_df[['index']], test_labels, how='left', on=['index'])

    all_features = feature_info['dense_features'] + feature_info['sparse_features']
    if use_dict:
        train_inputs = {name: np.array(train_df[name].tolist()) for name in all_features}
        train_labels = train_df['whether_to_click'].values
        test_inputs = {name: np.array(test_df[name].tolist()) for name in all_features}
        test_labels = test_labels['whether_to_click'].values
    else:
        train_inputs = [np.array(train_df[name]) for name in all_features]
        train_labels = train_df['whether_to_click'].values
        test_inputs = [np.array(test_df[name]) for name in all_features]
        test_labels = test_labels['whether_to_click'].values

    features_columns = [DenseFeat(name=feat,
                                  dimension=1,
                                  dtype='float32', )
                        for feat in feature_info['dense_features']]

    features_columns += [SparseFeat(name=feat,
                                    embed_name=feat,
                                    embed_dim=args.embed_dim,
                                    vocab_size=all_data[feat].max() + 1,
                                    dtype='int32', )
                         for feat in feature_info['sparse_features']]

    return (train_inputs, train_labels), (test_inputs, test_labels), features_columns


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
