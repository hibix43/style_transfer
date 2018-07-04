# -*- coding: utf-8 -*-

import train_network
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K


def contents_feature(input_shape):
    # 学習ネットワーク
    train_net = train_network.TrainNet()
    # 入力層
    contents_inputs = Input(shape=input_shape, name='input_contents')
    # コンテンツ画像を入力に、中間層の出力を取得
    hidden_model = train_net.rebuild_vgg16(contents_inputs, False, True)
    # コンテンツ画像から特徴量を抽出するモデル構築
    contents_model = Model(
        inputs=contents_inputs,
        outputs=hidden_model.output
    )

    return contents_model


# コンテンツ特徴量の損失関数
def contents_feature_loss(y_contents, contents_pred):
    # 二乗誤差
    return K.sum(K.square(contents_pred - y_contents), axis=(1, 2, 3)) / 2.0
