# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：activation.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/27 21:15 
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
# from tensorflow.keras.initializers import Zeros

unicode = str


def activation_layer(activation):
    if isinstance(activation, (str, unicode)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "无效激活找到 %s，应该使用str或激活层类。" % activation)
    return act_layer


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
