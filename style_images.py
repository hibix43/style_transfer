#!/usr/bin/python
# # -*- Coding: utf-8 -*-

import numpy as np
import train_network
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import (
 load_img, img_to_array, array_to_img)

STYLE_IMAGE_PATH = './img/style/style.jpg'


def load_image(image_shape):
    # スタイル画像の読み込み
    style_image = load_img(STYLE_IMAGE_PATH, target_size=image_shape[:2])
    # numpy配列に変換
    np_style_image = np.expand_dims(img_to_array(style_image), axis=0)

    return np_style_image


def style_feature(input_shape):
    # 学習ネットワーク
    train_net = train_network.TrainNet()
    # 入力層
    style_input = Input(shape=input_shape, name='input_style')
    # スタイル画像を入力に、中間層の出力を取得
    hidden_model = train_net.rebuild_vgg16(style_input, True, False)
    # スタイルから特徴量を抽出するモデル構築
    style_model = Model(
        inputs=style_input,
        outputs=hidden_model.output
    )

    return style_model


# スタイル特徴量の損失関数
def style_feature_loss(y_style, style_pred):
    # 二乗誤差
    return K.sum(K.square(
        gram_matrix(style_pred) - gram_matrix(y_style)), axis=(1, 2)) / 2.0


# グラム行列　=> スタイルの近さを計測
def gram_matrix(X):
    # 軸の入れ替え => batch, channel, height, width
    axis_replaced_X = K.permute_dimensions(X, (0, 3, 2, 1))
    replaced_shape = K.shape(axis_replaced_X)
    # 特徴マップ（高さと幅を1つの軸に展開）の内積をとるためのshape
    dot_shape = (replaced_shape[0], replaced_shape[1],
                 replaced_shape[2]*replaced_shape[3])
    # 実際に内積を計算する行列
    dot_X = K.reshape(axis_replaced_X, dot_shape)
    # 転置行列
    dot_X_t = K.permute_dimensions(dot_X, (0, 2, 1))
    # 行列の内積
    dot = K.batch_dot(dot_X, dot_X_t)

    return dot
