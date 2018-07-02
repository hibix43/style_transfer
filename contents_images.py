#!/usr/bin/python
# -*- Coding: utf-8 -*-

import train_network


def contents_feature(input_shape):
    # 学習ネットワーク
    train_net = train_network.TrainNet()
    # 入力層
    contents_inputs = Input(shape=input_shape, name='input_contents')
    # コンテンツ画像を入力に、中間層の出力を取得
    _ = train_net.rebuild_vgg16(contents_inputs, False, True)
    # コンテンツ画像から特徴量を抽出するモデル構築
    contents_model = Model(
        inputs=contents_inputs,
        outputs=train_net.outputs
    )

    return contents_model
