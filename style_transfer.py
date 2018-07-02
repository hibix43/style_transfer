#!/usr/bin/python
# # -*- Coding: utf-8 -*-

import convert_network
import train_network
import style_images
import contents_images
import numpy as np
from math import ceil


def build():
    input_shape = (224, 224, 3)
    # 変換ネットワーク
    convert_model = convert_network.build_network()
    # 学習ネットワーク
    train_net = train_net.TrainNet(convert_model)
    # ネットワーク構築
    train_model = train_net.rebuild_vgg16(convert_model.outputs, True, True)
    # スタイル画像
    style_image = style_images.load_image(input_shape)
    # スタイル特徴量抽出モデル
    y_style_model = style_images.style_feature(style_image, input_shape)
    # スタイル特徴量を抽出する
    y_style = y_style_model.predict(style_img)


# 画像パスリストから画像データ配列を得る
def get_images_array_from_path_list(img_path_list, image_size=(224, 224)):
    # img_array   : (N2, N1)
    # expand_dims : (N2, N1) => (1, N2, N1)
    # concatenate : (1, N2, N1) + (1, N2, N1) + ... => (1, N3, N2, N1)
    img_list = [np.expand_dims(
                img_to_array(load_img(img_path, image_size=image_size)),
                axis=0) for img_path in img_path_list]
    return np.concatenate(img_list, axis=0)


# 1エポックあたりの訓練データジェネレータ
def train_generator_per_epoch(img_path_list, batch_size, y_style,
                              y_contents_model, shuffle=True, epoches=None):
    # 訓練データ数
    train_img_size = len(img_path_list)
    # 1エポックにおけるバッチ処理回数（切り上げ）
    batch_steps_per_epoch = ceil(train_img_size / batch_size)
    # numpy配列化
    if not isinstance(img_path_list, np.array):
        img_path_list = np.array(img_path_list)
    # エポック数
    epoch_counter = 0
    # ジェネレータ
    while True:
        epoch_counter += 1
        # シャッフル
        np.random.shuffle(img_path_list) if shuffle
        # バッチ単位
        for step in range(batch_steps_per_epoch):
            # インデックス確保
            start, end = batch_size * step, batch_size * (step + 1)
            # バッチ単位入力画像
            batch_input_images = get_images_array_from_path_list(
                                                   img_path_list[start:end])
            # バッチ単位に拡張
            y_styles = np.array([y_style] * batch_input_images.shape[0])
            # コンテンツ特徴量の抽出
            y_contents = y_contents_model.predict(batch_input_images)
            # ジェネレータとして値を出力
            yield batch_input_images, y_styles + [y_contents]

            # エポック数が指定されていて、上限に達した場合
            if epoches is not None and epoch_counter >= epoches:
                # ジェネレータ停止
                raise StopIteration
