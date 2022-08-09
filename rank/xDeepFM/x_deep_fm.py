# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：x_deep_fm.py
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/30 20:15 
"""
import os
import pickle
import pandas as pd
from core import DNN
import tensorflow as tf
from feature_colum import *
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
from deepctr.feature_column import get_feature_names

# 构建输入层
# 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
from config.config import rank_save_path, data_type


def build_input_layers(feature_columns):
    """构建Input层字典，并以dense和sparse两类字典的形式返回"""
    feat_inputs = OrderedDict()
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_inputs = Input(shape=(1,), name=fc.name, dtype=fc.dtype)
            sparse_input_dict[fc.name] = sparse_inputs
            feat_inputs[fc.name] = sparse_inputs
        elif isinstance(fc, DenseFeat):
            dense_inputs = Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
            dense_input_dict[fc.name] = dense_inputs
            feat_inputs[fc.name] = dense_inputs
    return dense_input_dict, sparse_input_dict, feat_inputs


# 构建embedding层
def build_embedding_layers(feature_columns, input_layer_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()

    # 将特征中的sparse特征筛选出来
    sparse_features_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    # 如果是用于线性部分的embedding层，其维度是1，否则维度是自己定义的embedding维度
    if is_linear:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)

    return embedding_layers_dict


# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # 将sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name]  # 获取输入层
        _embed = embedding_layer_dict[fc.name]  # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input)  # B x dim  将input层输入到embedding层中

        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = Flatten()(embed)

        embedding_list.append(embed)

    return embedding_list


class CIN(Layer):
    def __init__(self, cin_size, l2_reg=1e-4):
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.field_nums = input_shape[1]

        # CIN 的每一层大小，这里加入第0层，也就是输入层H_0
        self.field_nums = [self.field_nums] + self.cin_size

        # 过滤器
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),  # 这个大小要理解
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums) - 1)
        }

        super(CIN, self).build(input_shape)

    def call(self, inputs):
        embed_dim = inputs.shape[-1]
        hidden_layers_results = [inputs]

        # 从embedding的维度把张量一个个的切开,这个为了后面逐通道进行卷积，算起来好算
        # 这个结果是个list， list长度是embed_dim, 每个元素维度是[None, field_nums[0], 1]  field_nums[0]即输入的特征个数
        # 即把输入的[None, field_num, embed_dim]，切成了embed_dim个[None, field_nums[0], 1]的张量
        split_X_0 = tf.split(hidden_layers_results[0], embed_dim, 2)

        for idx, size in enumerate(self.cin_size):
            # 这个操作和上面是同理的，也是为了逐通道卷积的时候更加方便，分割的是当一层的输入Xk-1
            split_X_K = tf.split(hidden_layers_results[-1], embed_dim,
                                 2)  # embed_dim个[None, field_nums[i], 1] feild_nums[i] 当前隐藏层单元数量

            # 外积的运算
            out_product_res_m = tf.matmul(split_X_0, split_X_K,
                                          transpose_b=True)  # [embed_dim, None, field_nums[0], field_nums[i]]
            out_product_res_o = tf.reshape(out_product_res_m,
                                           shape=[embed_dim, -1, self.field_nums[0] * self.field_nums[idx]])  # 后两维合并起来
            out_product_res = tf.transpose(out_product_res_o,
                                           perm=[1, 0, 2])  # [None, dim, field_nums[0]*field_nums[i]]

            # 卷积运算
            # 这个理解的时候每个样本相当于1张通道为1的照片 dim为宽度， field_nums[0]*field_nums[i]为长度
            # 这时候的卷积核大小是field_nums[0]*field_nums[i]的, 这样一个卷积核的卷积操作相当于在dim上进行滑动，每一次滑动会得到一个数
            # 这样一个卷积核之后，会得到dim个数，即得到了[None, dim, 1]的张量， 这个即当前层某个神经元的输出
            # 当前层一共有field_nums[i+1]个神经元， 也就是field_nums[i+1]个卷积核，最终的这个输出维度[None, dim, field_nums[i+1]]
            cur_layer_out = tf.nn.conv1d(input=out_product_res, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                         padding='VALID')

            cur_layer_out = tf.transpose(cur_layer_out, perm=[0, 2, 1])  # [None, field_num[i+1], dim]

            hidden_layers_results.append(cur_layer_out)

        # 最后CIN的结果，要取每个中间层的输出，这里不要第0层的了
        final_result = hidden_layers_results[1:]  # 这个的维度T个[None, field_num[i], dim]  T 是CIN的网络层数

        # 接下来在第一维度上拼起来
        result = tf.concat(final_result, axis=1)  # [None, H1+H2+...HT, dim]
        # 接下来， dim维度上加和，并把第三个维度1干掉
        result = tf.reduce_sum(result, axis=-1, keepdims=False)  # [None, H1+H2+..HT]

        return result


def xDeepFM(feature_columns,
            linear_feature_columns,
            dnn_feature_columns,
            dnn_hidden_units=[64, 128, 64],
            embed_l2_reg=1e-5,
            linear_l2_reg=1e-5,
            linear_use_bias=True,
            cin_size=[128, 128],
            dnn_l2_reg=1e-5,
            dnn_drop_rate=.0,
            dnn_use_bn=False,
            dnn_activation='relu',
            seed=48):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict, feat_inputs = build_input_layers(linear_feature_columns + dnn_feature_columns)
    # ------------------------------------

    inputs_list = list(feat_inputs.values())

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    sparse_embed_dict = create_embed_dict(sparse_feature_columns, embed_l2_reg)
    sparse_embed_list = embedding_lookup(sparse_embed_dict, feat_inputs, sparse_feature_columns, to_list=True)

    dense_inputs = get_dense_inputs(feat_inputs, feature_columns, concat_flag=True)

    linear_logit = get_linear_logit(feat_inputs=feat_inputs,
                                    feature_columns=feature_columns,
                                    linear_l2_reg=linear_l2_reg,
                                    embed_l2_reg=embed_l2_reg,
                                    use_bias=linear_use_bias,
                                    seed=seed)

    sparse_embed_inputs = Flatten()(Concatenate(axis=-1)(sparse_embed_list))
    dnn_inputs = tf.concat([dense_inputs, sparse_embed_inputs], axis=-1)
    dnn_logit = DNN(hidden_units=dnn_hidden_units,
                    activation=dnn_activation,
                    l2_reg=dnn_l2_reg,
                    dropout_rate=dnn_drop_rate,
                    use_bn=dnn_use_bn
                    )(dnn_inputs)
    dnn_logit = Dense(units=1,
                      use_bias=False,
                      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)
                      )(dnn_logit)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # 线性层和dnn层统一的embedding层
    embedding_layer_dict = build_embedding_layers(linear_feature_columns + dnn_feature_columns, sparse_input_dict,
                                                  is_linear=False)

    # CIN侧的计算逻辑， 这里使用的DNN feature里面的sparse部分,这里不要flatten
    exFM_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict,
                                                 flatten=False)
    exFM_input = Concatenate(axis=1)(exFM_sparse_kd_embed)
    exFM_out = CIN(cin_size=cin_size)(exFM_input)
    exFM_logits = Dense(1)(exFM_out)

    final_outputs = tf.nn.sigmoid(linear_logit + dnn_logit + exFM_logits)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=final_outputs)

    return model


def create_ctr_data_new(path):
    """
    构造排序模型的输入数据和特征
    :param path:
    :return:
    """
    with open(os.path.join(path, data_type + '_rank_data.pkl'), 'rb') as f:
        data, feature_info = pickle.load(f)
        f.close()

    # sparse_features = ['user_id', 'item_id', 'exposure_location']
    sparse_features = feature_info['sparse_features']
    dense_features = feature_info['dense_features']

    # 计算每个稀疏字段的#unique特征并为序列特征生成特征配置
    sparse_features_columns = [SparseFeat(feat, data[feat].max() + 1, embed_dim=8) for feat in sparse_features]
    dense_features_columns = [DenseFeat(feat, dimension=1, dtype='float32') for feat in dense_features]
    linear_feature_columns = sparse_features_columns + dense_features_columns
    dnn_feature_columns = sparse_features_columns + dense_features_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 为模型生成输入数据
    data = shuffle(data)
    train_data = data[data['whether_to_click'] != -1]
    test_data = data[data['whether_to_click'] == -1]

    test_labels = pd.read_pickle(os.path.join(path, data_type + '_rank_test_label.pkl'))
    test_labels = pd.merge(test_data[['index']], test_labels, how='left', on=['index'])

    train_inputs = {name: train_data[name] for name in feature_names}
    train_labels = train_data['whether_to_click'].values
    test_inputs = {name: test_data[name] for name in feature_names}
    test_labels = test_labels['whether_to_click'].values

    return (train_inputs, train_labels), (test_inputs, test_labels), linear_feature_columns, dnn_feature_columns


def run_train():
    train_data, test_data, linear_feature_columns, dnn_feature_columns = create_ctr_data_new(rank_save_path)

    # 构建xDeepFM模型
    model = xDeepFM(linear_feature_columns + dnn_feature_columns, linear_feature_columns, dnn_feature_columns)
    model.summary()
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 模型训练
    history = model.fit(train_data[0], train_data[1], batch_size=64, epochs=5, validation_split=0.2, )

    # 训练集准确率
    plt.plot(history.history['auc'], label='training auc')
    # 测试集准确率
    plt.plot(history.history['val_auc'], label='val auc')
    plt.title('auc')
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.legend()
    plt.savefig(os.path.join(rank_save_path, "每一轮曲线下面积auc图片_x_deep_fm.png"))

    plt.clf()
    # 训练集损失值
    plt.plot(history.history['loss'], label='training loss')
    # 测试集损失值
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(rank_save_path, "每一轮损失值loss图片_x_deep_fm.png"))


def main():
    pass


if __name__ == '__main__':
    main()
