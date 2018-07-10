# -*- coding: utf-8 -*-

import glob
import numpy as np
import convert_network
import train_network
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.preprocessing.image import (
    load_img, img_to_array, array_to_img)


# テスト
def test(covert_model, images_list, input_shape=(224, 224, 3)):
    # 変換
    predict = covert_model.predict(images_list)
    for i in range(predict.shape[0]):
        # 保存できる画像に変換
        predict_image = array_to_img(predict[i])
        # 保存
        file_name = f'./img/test/predicted_images/test{i}.jpg'
        predict_image.save(file_name)
        print(f'>> predict! test{i}.jpg')


# コンテンツ入力画像のパスをすべて取得
def get_img_path_list(path):
    img_path_list = glob.glob(path)
    return img_path_list


# 画像パスリストから画像データ配列を得る
def get_images_array_from_path_list(img_path_list, image_size=(224, 224)):
    img_list = [np.expand_dims(
                img_to_array(load_img(img_path, target_size=image_size)),
                axis=0) for img_path in img_path_list]
    return np.concatenate(img_list, axis=0)


if __name__ == '__main__':
    weight_loss_name = './model/step48150_loss73.20453643798828.h5'
    test_images_path = 'img/test/test_images/*.jpg'

    # 変換ネットワーク
    convert_model = convert_network.build_network()
    # モデル構築
    network = train_network.TrainNet()
    model = network.rebuild_vgg16(convert_model.output,
                                  True, True, convert_model.input)
    # 重み読み込み
    model.load_weights(weight_loss_name)
    print('>> load weights.')
    # 画像読み込み
    images_list = get_images_array_from_path_list(
               get_img_path_list(test_images_path))
    print('>> get image path.')
    # 変換
    print('>> test start.')
    test(convert_model, images_list)
    print('>> test finish.')
