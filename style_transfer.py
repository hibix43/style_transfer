#!/usr/bin/python
# # -*- Coding: utf-8 -*-

import convert_network
import train_network
import style_images


def build():
    input_shape = (224, 224, 3)
    # 変換ネットワーク
    c_model = convert_network.build_network()
    # 学習ネットワーク
    train_net = train_net.TrainNet(c_model)
    # ネットワーク構築
    t_model = train_net.rebuild_vgg16(c_model.outputs, True, True)
    # スタイル画像
    style_img = style_images.load_image(input_shape)
    # スタイル特徴量抽出モデル
    style_t_model = style_images.style_feature(style_img, input_shape)
    # スタイル特徴量を抽出する
    y_style = style_t_model.predict(style_img)
