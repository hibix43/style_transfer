# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K
import train_network


def contents_feature(train_net):
    input_data = Input(shape=input_shape, name='input_contents')
    # 学習ネットワークインスタンス化
    train_net = train_network.TrainNet()
    # コンテンツ画像から特徴量を抽出するモデル構築
    contents_model = train_net.rebuild_vgg16(input_data, False, True)

    return contents_model


# コンテンツ特徴量の損失関数
def contents_feature_loss(y_contents, contents_pred):
    norm = K.prod(K.cast(K.shape(y_contents)[1:], 'float32'))
    # 二乗誤差
    return K.sum(K.square(contents_pred - y_contents), axis=(1, 2, 3)) / norm
