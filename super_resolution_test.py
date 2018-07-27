# -*- coding: utf-8 -*-

import glob
import numpy as np
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.preprocessing.image import (
    load_img, img_to_array, array_to_img)


# テスト
def test(model, images_list, input_shape=(224, 224, 3)):
    predict = model.predict(images_list)
    for i in range(predict.shape[0]):
        # 保存できる画像に変換
        predict_image = array_to_img(predict[i])
        # 保存
        file_name = f'{test_files}predicted_images/test{i}.jpg'
        predict_image.save(file_name)
        print(f'>> predict! test{i}.jpg')


# ワイルドカードからファイルを取得する
def get_path_using_glob(path):
    return glob.glob(path)


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
    test_files = 'production/super_resolution/'
    model_json_path = test_files + '*.json'
    weight_loss_path = test_files + '*.h5'
    test_images_path = test_files + 'test_images/*.jpg'

    model_json_path = get_path_using_glob(model_json_path)
    weight_loss_path = get_path_using_glob(weight_loss_path)
    test_images_path = get_path_using_glob(test_images_path)

    # モデル読み込み
    model_json_path = open(model_json_path[0]).read()
    model = model_from_json(model_json_path)

    print('>> load model.')
    # 重み読み込み
    model.load_weights(weight_loss_path[0])
    print('>> load weights.')
    # 画像読み込み
    images_list = get_images_array_from_path_list(test_images_path)
    print('>> get image path.')
    # 変換
    print('>> test start.')
    test(model, images_list)
    print('>> test finish.')
