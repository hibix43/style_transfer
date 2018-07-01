#!/usr/bin/python
# # -*- Coding: utf-8 -*-

import numpy as np
import train_network
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.model import Model
from tensorflow.python.keras.preprocessing.image import load_img,
img_to_array, array_to_img

STYLE_IMAGE_PATH = './img/style/style.png'


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
    _ = train_net.rebuild_vgg16(style_input, True, False)
    # スタイルから特徴量を抽出するモデル構築
    style_model = Model(
        inputs=style_input,
        outputs=train_net.outputs
    )

    return style_model
