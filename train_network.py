#!/usr/bin/python
# # -*- Coding: utf-8 -*-

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda


def preprocess_inputs(inputs):
    """ VGG16に合うよう入力値を前処理する
        BGR変換と近似的中心化 """
    return (inputs[:, :, :, ::-1] - 120) / 255.0


class TrainNet():
    def __init__(self, model):
        # 特徴量を抽出する層
        self.style_layer_names = (
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3'
        )
        self.content_layer_names = ('block3_conv3', )
        # 中間層の出力
        self.style_outputs = []
        self.contents_outputs = []
        # 変換ネットワーク
        # self.convert_model = model
        # 変換ネットワークの出力を入力とする
        # self.inputs_data = model.output_data
        # VGG16呼び出し
        self.vgg16 = VGG16()
        # 学習させない設定をする
        layer.trainable = False for layer in self.vgg16.layers

    def rebuild_vgg16(inputs_data, style_layer=True, contents_layer=True):
        # 正則化
        l = Lambda(preprocess_inputs)(self.inputs_data)
        # VGG16を再構築
        for layer in self.vgg16.layers:
            l = layer(l)
            # 特徴量を抽出する中間層の出力を取得
            if style_layer and layer.name in self.style_layer_names:
                self.style_outputs.append(l)
            if contents_layer and layer.name in self.content_layer_names:
                self.contents_outputs.append(l)
        # モデル構築
        train_model = Model(
            inputs=inputs_data,
            output_data=(self.style_outputs if style_layer +
                         self.contents_outputs if contents_layer)
        )
        return train_model
