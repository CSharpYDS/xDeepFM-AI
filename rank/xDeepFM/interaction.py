# -*- coding: UTF-8 -*-
"""
@Project ：recommender 
@File    ：interaction.py 
@IDE     ：PyCharm 
@Author  ：YDS 
@Date    ：2022/7/27 21:17 
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class FMCross(Layer):
    def __init__(self, **kwargs):
        super(FMCross, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("意外的输入维度%d，预计为3个维度" % (len(input_shape)))

        super(FMCross, self).build(input_shape)

    def call(self, inputs, **kwargs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))      # None, 1, dim
        sum_of_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)        # None, 1, dim

        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)         # None, 1

        return cross_term


def main():
    print("Hello World")


if __name__ == '__main__':
    main()
