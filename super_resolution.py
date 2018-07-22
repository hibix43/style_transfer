# -*- coding: utf-8 -*-

import numpy as np
import glob
from os import path, makedirs
from shutil import move
from random import shuffle
from tensorflow.python.keras.preprocessing.image import (
 load_img, img_to_array, array_to_img, ImageDataGenerator)


DATA_IMAGES_DIR_PATH = 'img/contents/'
TRAIN_DIR = 'super_resolution_train/'
TEST_DIR = 'super_resolution_test/'
DATA_NUM = 10700
BATCH_SIZE = 20


# 低解像画像の作成
def create_low_resolution_image(image, low_scale=2.0):
    width, height = image.shape[:2]
    low_image_size = int(width / low_scale), int(height / low_scale)
    # 配列から画像へ
    if isinstance(image, np.ndarray):
        image = array_to_img(image)
    # 強引にサイズを小さくする
    low_resolution_image = image.resize(low_image_size, 3)
    # 大きくして低解像化
    low_image_size = low_resolution_image.resize(image.size, 3)
    # 配列に戻して返す
    return img_to_array(low_resolution_image)


# ジェネレータ
def low_resolution_image_generator(data_dir, mode, scale,
                                   load_size=(224, 224, 3),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True):
    # ジェネレータの設定
    gene_dict = {
        'directory': data_dir,
        'classes': [mode],
        'class_mode': None,
        'color_mode': 'rgb',
        'target_size': load_size,
        'batch_size': batch_size,
        'shuffle': shuffle
    }
    # ジェネレータ
    img_gene = ImageDataGenerator().flow_from_directory(gene_dict)
    # ジェネレータ動作
    for images in img_gene:
        # 入力画像
        x = np.array([
            create_low_resolution_image(image, scale)
            for image in images])
        # 均一化して返す
        yield x/255.0, images/255.0


# 入力画像のパスをすべて取得
def get_img_path_list():
    img_path = path.join(DATA_IMAGES_DIR_PATH, '*.jpg')
    img_path_list = glob.glob(img_path)
    # print(img_path_list)
    return img_path_list


# 訓練データとテストデータを別フォルダに分ける
def split_datas_folder():
    # 振り分け先フォルダ
    train_dir = DATA_IMAGES_DIR_PATH + TRAIN_DIR
    test_dir = DATA_IMAGES_DIR_PATH + TEST_DIR
    # 振り分けるフォルダの作成
    makedirs(train_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)
    # 入力画像パスをすべて取得
    img_paths = get_img_path_list()
    # パスを8:2でランダムにわける
    TRAIN_DATA_PERCENTAGE = 0.8
    index_max = int(len(img_paths)*TRAIN_DATA_PERCENTAGE)
    # パス配列をシャッフル
    shuffle(img_paths)
    # 振り分ける
    train_img_paths = img_paths[:index_max]
    test_img_paths = img_paths[index_max:]
    # 画像をフォルダに移す
    for img_path in train_img_paths:
        print(f'>> Move {img_path} to {train_dir}')
        move(img_path, train_dir)
    for img_path in test_img_paths:
        print(f'>> Move {img_path} to {test_dir}')
        move(img_path, test_dir)


# 振り分けた画像を戻す
def reset_datas_folder():
    pass


# ジェネレータの生成
def create_generator():
    split_datas_folder()
    train_data_generator = low_resolution_image_generator(
        DATA_IMAGES_DIR_PATH, 'train')


if __name__ == '__main__':
    split_datas_folder()
