# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：data_process_rank.py
@IDE     ：PyCharm 
@Author  ：YDS
@Date    ：2022/7/27 20:50 
"""

import os
import gc
import pickle
import time

import swifter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config.config import rank_save_path, data_type, large_data_path, small_data_path, demo_data_path, \
    customize_data_path
from utils.data_compression import reduce_mem
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prob2val(feat_info):
    # 判断是否为空
    if feat_info == feat_info:
        prob_list = [values.split(':') for values in feat_info.split(',')]
        prob_list = sorted(prob_list, key=lambda x: float(x[1]))
        return prob_list[-1][0]
    else:
        return np.NaN


def get_second_title(x):
    if x['secondary'] == x['secondary']:
        second_titles = x['secondary'].split('/')
        for title in second_titles:
            # 跳过异常数据
            if title == 'A_0_24:0.447656,A_25_29:0.243809,A_30_39:0.076268,A_40+:0.232267':
                continue
            # 优先返回不等于primary的secondary
            if title != x['primary']:
                return title

    return x['primary']


def get_key_word(feat_info):
    if feat_info == feat_info and isinstance(feat_info, str):
        key_word_list = [values.split(':') for values in feat_info.replace('^', '').split(',')]

        new_list = []
        last_elem = ''
        for idx, values in enumerate(key_word_list):
            if len(values) == 1:
                last_elem = values[0] if last_elem == '' else ','.join([last_elem, values[0]])
                continue
            if len(values) > 2:
                # 将类似于‘你好，李焕英’这种key_words重新进行拼接
                # 这类key_words由于存在逗号，在获取key_word_list时被误分开了
                values[0] = ':'.join(values[:-1])

            values[0] = values[0] if last_elem == '' else ','.join([last_elem, values[0]])
            new_list.append(values)
            last_elem = ''

        return new_list[-1][0]
    else:
        return np.NaN


def get_statistical_features(all_data, past_day=7):
    """
    统计特征
    :param all_data:
    :param past_day:
    :return:
    """
    # 统计新闻从发文到曝光的日期差
    temp = all_data['exposure_date'] - all_data['date_of_publication']
    all_data['date_dif_from_pu_to_exposure'] = temp.dt.days
    all_data.loc[all_data['date_dif_from_pu_to_exposure'] < 0, 'date_dif_from_pu_to_exposure'] = 0
    all_data.fillna(value={'date_dif_from_pu_to_exposure': 0}, inplace=True)
    statis_dense_columns = ['date_dif_from_pu_to_exposure']
    dates = all_data['exposure_date'].unique()
    dates.sort()
    date_num = len(dates)
    date_map = dict(zip(dates, range(date_num)))
    all_data['exposure_date_idx'] = all_data['exposure_date'].map(date_map)
    train_data = all_data[all_data['whether_to_click'] != -1]

    for feat in tqdm([['user_id'], ['item_id'], ['primary'], ['secondary'],
                      ['user_id', 'primary'], ['user_id', 'secondary']]):
        res_arr = []
        name = f'{"".join(feat)}_exposure_total'
        statis_dense_columns.append(name)

        for day in range(0, date_num):
            train_data_temp = train_data[
                (train_data['exposure_date_idx'] >= day - past_day) & (train_data['exposure_date_idx'] < day)]
            train_data_temp = train_data_temp.groupby(feat)['item_id'].agg([(name, 'count')]).reset_index()
            train_data_temp['exposure_date_idx'] = day
            res_arr.append(train_data_temp)
        stat_all_data = pd.concat(res_arr)
        all_data = all_data.merge(stat_all_data, how='left', on=feat + ['exposure_date_idx'])

    target = 'whether_to_click'
    for feat in tqdm([['user_id'], ['item_id'], ['primary'], ['secondary'],
                      ['user_id', 'primary'], ['user_id', 'secondary']]):
        res_arr = []
        name_mean = f'{"".join(feat)}_ctr_mean'
        name_sum = f'{"".join(feat)}_ctr_total_sum'

        statis_dense_columns.append(name_mean)
        statis_dense_columns.append(name_sum)

        for day in range(0, date_num):
            train_data_temp = train_data[
                (train_data['exposure_date_idx'] >= day - past_day) & (train_data['exposure_date_idx'] < day)]
            train_data_temp = train_data_temp.groupby(feat)[target].agg(
                [(name_mean, 'mean'), (name_sum, 'sum')]).reset_index()
            train_data_temp['exposure_date_idx'] = day
            res_arr.append(train_data_temp)
        stat_all_data = pd.concat(res_arr)
        all_data = all_data.merge(stat_all_data, how='left', on=feat + ['exposure_date_idx'])

    target = 'consumption_time'
    for feat in tqdm([['user_id'], ['item_id'], ['primary'], ['secondary'],
                      ['user_id', 'primary'], ['user_id', 'secondary']]):
        res_arr = []
        name_mean = f'{"".join(feat)}_consumption_time_mean'
        name_std = f'{"".join(feat)}_consumption_time_std'
        name_sum = f'{"".join(feat)}_consumption_time_sum'
        statis_dense_columns.append(name_mean)
        statis_dense_columns.append(name_std)
        statis_dense_columns.append(name_sum)

        for day in range(0, date_num):
            train_data_temp = train_data[
                (train_data['exposure_date_idx'] >= day - past_day) & (train_data['exposure_date_idx'] < day)]
            train_data_temp = train_data_temp.groupby(feat)[target].agg(
                [(name_mean, 'mean'), (name_std, 'std'), (name_sum, 'sum')]).reset_index()
            train_data_temp['exposure_date_idx'] = day
            res_arr.append(train_data_temp)

        stat_all_data = pd.concat(res_arr)
        all_data = all_data.merge(stat_all_data, how='left', on=feat + ['exposure_date_idx'])

    return all_data, statis_dense_columns


def merge_data(train_data_path, test_data_path, user_path, doc_path, file_save_path):
    """
    合并数据（用于排序模型训练）
    :param train_data_path:
    :param test_data_path:
    :param user_path:
    :param doc_path:
    :param file_save_path:
    :return:
    """
    train_data = pd.read_pickle(train_data_path)
    test_data = pd.read_pickle(test_data_path)
    doc_info = pd.read_pickle(doc_path)
    user_info = pd.read_pickle(user_path)

    test_data['whether_to_click'] = -1
    all_data = pd.concat([train_data, test_data])

    # 1. 合并用户特征
    all_data = all_data.merge(
        user_info[['user_id', 'device_name', 'operating_system', 'province', 'city', 'age', 'gender']],
        how='left', on=['user_id']
    )
    del user_info
    gc.collect()

    # 2. 合并文档特征
    all_data = all_data.merge(
        doc_info[['item_id', 'primary', 'secondary', 'key_words', 'number_of_pictures', 'posting_time',
                  'date_of_publication']],
        how='left', on='item_id'
    )
    del doc_info
    gc.collect()

    # 3. 获取统计特征
    all_data, statis_dense_columns = get_statistical_features(all_data)

    # 4. 连续特征处理
    base_dense_columns = ['number_of_refreshes', 'number_of_pictures']
    dense_columns = base_dense_columns + statis_dense_columns

    all_data.fillna(value={feat: 0 for feat in dense_columns}, inplace=True)
    # sc = StandardScaler()
    # all_data[dense_columns] = sc.fit_transform(all_data[dense_columns])
    for feat in dense_columns:
        all_data[feat] = np.log(1 + all_data[feat])

    # 5. 离散特征处理
    sparse_columns = ['user_id', 'item_id', 'internet_environment', 'device_name', 'operating_system',
                      'exposure_position', 'province', 'city', 'age', 'gender', 'primary', 'secondary', 'key_words']
    for feat in sparse_columns:
        lb = LabelEncoder()
        all_data[feat] = lb.fit_transform(all_data[feat].astype(str))

    all_data = reduce_mem(all_data)
    feature_info = {'dense_features': dense_columns,
                    'sparse_features': sparse_columns}
    file = [all_data, feature_info]

    # all_data.to_csv(os.path.join(file_save_path, data_type + '_rank_data.txt'))  # 方便查看，使用全量数据集时建议注释

    file_save_path = os.path.join(file_save_path, data_type + '_rank_data.pkl')
    with open(file_save_path, 'wb') as f:
        pickle.dump(file, f)
        f.close()


def deal_item_rank(data_item_path, file_save_path):
    """
    处理用于排序模型训练的item
    :param data_item_path:
    :param file_save_path:
    :return:
    """
    doc_info = pd.read_table(data_item_path, sep='\t', dtype={2: np.str}, low_memory=False, header=None)
    doc_info.columns = ['item_id', 'title', 'posting_time', 'number_of_pictures', 'primary', 'secondary', 'key_words']

    # 处理异常的posting_time数据
    condition_row = (doc_info['posting_time'].isnull()) | (doc_info['posting_time'] == 'Android')
    time_fill_value = doc_info.loc[~condition_row, 'posting_time'].swifter.apply(lambda x: int(x[:10])).astype(
        'int').min()
    doc_info.loc[condition_row, 'posting_time'] = str(time_fill_value)

    doc_info['posting_time'] = pd.to_datetime(
        doc_info.loc[:, 'posting_time'], utc=True, unit='ms').dt.tz_convert('Asia/Shanghai')
    doc_info['date_of_publication'] = doc_info['posting_time'].dt.date

    doc_info['number_of_pictures'] = doc_info.loc[:, 'number_of_pictures'].swifter.apply(
        lambda x: 0 if (x in ['上海', '云南', '山东'] or x != x) else int(x))

    doc_info['secondary'] = doc_info.loc[:, ['primary', 'secondary']].swifter.apply(get_second_title, axis=1)
    doc_info['key_words'] = [get_key_word(words) for words in tqdm(doc_info['key_words'])]

    # doc_info.to_csv(os.path.join(file_save_path, data_type + '_rank_doc.txt'))  # 方便查看，使用全量数据集时建议注释
    doc_info.to_pickle(os.path.join(file_save_path, data_type + '_rank_doc.pkl'))


def deal_user_rank(data_user_path, file_save_path):
    """
    处理用于排序模型训练的user
    :param data_user_path:
    :param file_save_path:
    :return:
    """
    # 1. 处理用户文件
    user_info = pd.read_table(data_user_path, sep='\t', index_col=False, encoding='utf-8', header=None)
    user_info.columns = ['user_id', 'device_name', 'operating_system', 'province', 'city', 'age', 'gender']

    user_info['age'] = [prob2val(age_info) for age_info in tqdm(user_info['age'])]
    user_info['gender'] = [prob2val(sex_info) for sex_info in tqdm(user_info['gender'])]

    # user_info.to_csv(os.path.join(file_save_path, data_type + '_rank_user.txt'))  # 方便查看，使用全量数据集时建议注释
    user_info.to_pickle(os.path.join(file_save_path, data_type + '_rank_user.pkl'))


def deal_log_rank(data_log_path, file_save_path):
    """
    处理用于排序模型训练的log
    :param data_log_path:
    :param file_save_path:
    :return:
    """
    # 1. 数据读取
    all_data = pd.read_table(data_log_path, sep='\t', index_col=False, header=None)

    all_data.columns = ['user_id', 'item_id', 'exposure_time', 'internet_environment', 'number_of_refreshes',
                        'exposure_position', 'whether_to_click', 'consumption_time']
    print(f'样本总数为：{all_data.shape[0]}')

    # 2. 数据处理
    all_data.loc[all_data['consumption_time'] < 0, 'consumption_time'] = 0
    all_data['exposure_time'] = pd.to_datetime(all_data.loc[:, 'exposure_time'], utc=True, unit='ms').dt.tz_convert(
        'Asia/Shanghai')
    all_data['exposure_date'] = all_data['exposure_time'].dt.date
    all_data['index'] = range(all_data.shape[0])

    dates = all_data['exposure_date'].unique()
    dates.sort()

    # 3. 训练、测试数据集划分
    train_data = all_data[all_data['exposure_date'] != dates[-1]]
    test_data = all_data[all_data['exposure_date'] == dates[-1]]
    test_label = test_data[['index', 'whether_to_click']]

    # 4. 测试集处理
    test_data = test_data.drop(columns=['consumption_time', 'exposure_position', 'whether_to_click'])

    # train_data.to_csv(os.path.join(file_save_path, data_type + '_rank_train_data.txt'))  # 方便查看，使用全量数据集时建议注释
    # test_data.to_csv(os.path.join(file_save_path, data_type + '_rank_test_data.txt'))  # 方便查看，使用全量数据集时建议注释
    # test_label.to_csv(os.path.join(file_save_path, data_type + '_rank_test_label.txt'))  # 方便查看，使用全量数据集时建议注释

    train_data.to_pickle(os.path.join(file_save_path, data_type + '_rank_train_data.pkl'))
    test_data.to_pickle(os.path.join(file_save_path, data_type + '_rank_test_data.pkl'))
    test_label.to_pickle(os.path.join(file_save_path, data_type + '_rank_test_label.pkl'))


def main():
    if data_type == 'large':
        raw_data_path = large_data_path
    elif data_type == 'small':
        raw_data_path = small_data_path
    elif data_type == 'demo':
        raw_data_path = demo_data_path
    else:
        raw_data_path = customize_data_path

    file_save_path = rank_save_path
    data_merge_path = rank_save_path

    # error_user_id_path = os.path.join(data_path, 'user_error_id.txt')
    # error_item_id_path = os.path.join(data_path, 'doc_error_id.txt')

    raw_path_item = os.path.join(raw_data_path, 'doc_info.txt')
    raw_path_user = os.path.join(raw_data_path, 'user_info.txt')
    raw_path_log = os.path.join(raw_data_path, 'train_data.txt')

    deal_item_rank(raw_path_item, file_save_path)
    deal_user_rank(raw_path_user, file_save_path)
    deal_log_rank(raw_path_log, file_save_path)

    train_data_path = os.path.join(data_merge_path, data_type + '_rank_train_data.pkl')
    test_data_path = os.path.join(data_merge_path, data_type + '_rank_test_data.pkl')
    user_path = os.path.join(data_merge_path, data_type + '_rank_user.pkl')
    doc_path = os.path.join(data_merge_path, data_type + '_rank_doc.pkl')

    merge_data(train_data_path, test_data_path, user_path, doc_path, file_save_path)


if __name__ == '__main__':
    t = time.time()
    main()
    print('总耗时: ', time.time() - t)
