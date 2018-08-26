# -*- coding: utf-8 -*-

import glob
import numpy as np
import convert_network
import train_network
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.preprocessing.image import (
    load_img, img_to_array, array_to_img)
from tensorflow.python.keras.applications.vgg16 import VGG16


# テスト
def test(covert_model, images_list, input_shape=(224, 224, 3)):
    # 変換
    predict = covert_model.predict(images_list)
    for i in range(predict.shape[0]):
        # 保存できる画像に変換
        predict_image = array_to_img(predict[i])
        # 保存
        file_name = f'{test_files}/predicted_images/test{i}.jpg'
        predict_image.save(file_name)
        print(f'>> predict! test{i}.jpg')


# コンテンツ入力画像のパスをすべて取得
def get_img_path_list(path):
    img_path_list = glob.glob(path)
    return img_path_list


# ワイルドカードからファイルを取得する
def get_path_using_glob(path):
    return glob.glob(path)


# 画像パスリストから画像データ配列を得る
def get_images_array_from_path_list(img_path_list, image_size=(224, 224)):
    img_list = [np.expand_dims(
                img_to_array(load_img(img_path, target_size=image_size)),
                axis=0) for img_path in img_path_list]
    return np.concatenate(img_list, axis=0)


if __name__ == '__main__':
    test_files = 'production/style_transfer/'
    # model_json_path = test_files + '*.json'
    weight_loss_path = test_files + '*.h5'
    test_images_path = test_files + 'test_images/*.jpg'

    # model_json_path = get_path_using_glob(model_json_path)
    weight_loss_path = get_path_using_glob(weight_loss_path)
    test_images_path = get_path_using_glob(test_images_path)

    # 変換ネットワーク
    input_shape = (224, 224, 3)
    convert_model = convert_network.build_network(input_shape)
    # モデル構築
    network = train_network.TrainNet(input_shape)
    model = network.rebuild_vgg16(convert_model.output,
                                  True, True, convert_model.input)

    # モデル読み込み
    # model_json_path = open(model_json_path[0]).read()
    # input_data = Input(shape=(1, 224, 224, 3))
    # model = model_from_json(model_json_path, get_vgg16_layer_dict())
    # model = model_from_json(model_json_path)

    print('>> load model.')
    # 重み読み込み
    model.load_weights(weight_loss_path[0])
    print('>> load weights.')
    # 画像読み込み
    images_list = get_images_array_from_path_list(
        test_images_path, input_shape[:2])
    print('>> get image path.')
    # 変換
    print('>> test start.')
    test(convert_model, images_list, input_shape)
    print('>> test finish.')
