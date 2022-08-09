# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：embedding.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/27 21:16 
"""

from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2


def create_embed_dict(sparse_feature_columns, embed_l2_reg):
    sparse_embed_dict = {}
    for feat in sparse_feature_columns:
        feat_embed_name = feat.embed_name
        if feat_embed_name not in sparse_embed_dict.keys():
            embed_layer = Embedding(
                input_dim=feat.vocab_size,
                input_length=1,
                output_dim=feat.embed_dim,
                embeddings_initializer=feat.embed_init,
                embeddings_regularizer=l2(embed_l2_reg)
            )
            embed_layer.trainable = True
            sparse_embed_dict[feat_embed_name] = embed_layer

    return sparse_embed_dict


def embedding_lookup(sparse_embed_dict, feat_inputs, sparse_feature_columns, query_features=(), to_list=False):
    feat_embed_outputs = {}
    for feat in sparse_feature_columns:
        feat_name = feat.name
        if len(query_features) == 0 or feat_name in query_features:
            feat_input = feat_inputs[feat_name]
            feat_embed_outputs[feat_name] = sparse_embed_dict[feat.embed_name](feat_input)

    if to_list:
        return list(feat_embed_outputs.values())

    return feat_embed_outputs


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
