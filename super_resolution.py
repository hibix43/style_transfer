# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.python.keras.preprocessing.image import (
 load_img, img_to_array, array_to_img, ImageDataGenerator)


DATA_IMAGES_DIR_PATH = 'img/contents/'
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
                                   batch_size=20, shuffle=True):
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
