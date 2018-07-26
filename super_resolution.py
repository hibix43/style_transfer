# -*- coding: utf-8 -*-

import numpy as np
import glob
from os import path, makedirs
from shutil import move
from random import shuffle
from tensorflow.python.keras.preprocessing.image import (
 load_img, img_to_array, array_to_img, ImageDataGenerator)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input, Add, Conv2D, Conv2DTranspose, Activation)

# @googlecolaboratory:drive/
DATA_IMAGES_DIR_PATH = 'img/contents/'
TRAIN_DIR = 'super_resolution_train/'
TEST_DIR = 'super_resolution_test/'
DATA_NUM = 1000
BATCH_SIZE = 20


# 入力画像のパスをまとめて取得
def get_img_path_list(path):
    img_path = ''.join([path, '*.jpg'])
    img_path_list = glob.glob(img_path)
    return img_path_list[:DATA_NUM]


# 訓練データとテストデータを別フォルダに分ける
def split_datas_folder():
    # 振り分け先フォルダ
    train_dir = DATA_IMAGES_DIR_PATH + TRAIN_DIR
    test_dir = DATA_IMAGES_DIR_PATH + TEST_DIR
    # 振り分けるフォルダの作成
    makedirs(train_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)
    # 入力画像パスをすべて取得
    img_paths = get_img_path_list(DATA_IMAGES_DIR_PATH)
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
    return len(train_img_paths), len(test_img_paths)


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
    low_resolution_image = low_resolution_image.resize(image.size, 3)
    # 配列に戻して返す
    return img_to_array(low_resolution_image)


# ジェネレータ
def low_resolution_image_generator(data_dir, mode, scale,
                                   load_size=(448, 448),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True):
    # ジェネレータ
    img_gene = ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode='rgb',
        target_size=load_size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    # ジェネレータ動作
    for images in img_gene:
        # 入力画像
        x = np.array([
            create_low_resolution_image(image, scale)
            for image in images])
        # 均一化して返す
        yield x/255.0, images/255.0


# 振り分けた画像を戻す
def reset_datas_folder():
    pass


# ピーク信号対雑音比（データの劣化具合を採点）
def psnr(y_true, y_pred):
    return -10*K.log(K.mean(K.flatten(y_true - y_pred)**2))/np.log(10)


# モデル構築
def create_model():
    # 入力は任意のサイズで、3チャンネルの画像
    inputs = Input((None, None, 3), dtype='float')
    # Endoder
    conv1 = Conv2D(64, 3, padding='same')(inputs)
    conv1 = Conv2D(64, 3, padding='same')(conv1)
    conv2 = Conv2D(64, 3, strides=2, padding='same')(conv1)
    conv2 = Conv2D(64, 3, padding='same')(conv2)
    conv3 = Conv2D(64, 3, strides=2, padding='same')(conv2)
    conv3 = Conv2D(64, 3, padding='same')(conv3)
    # Decoder
    deconv3 = Conv2DTranspose(64, 3, padding='same')(conv3)
    deconv3 = Conv2DTranspose(64, 3, strides=2, padding='same')(deconv3)
    # Add()レイヤーを使ってスキップコネクションを表現
    merge2 = Add()([deconv3, conv2])
    deconv2 = Conv2DTranspose(64, 3, padding='same')(merge2)
    deconv2 = Conv2DTranspose(64, 3, strides=2, padding='same')(deconv2)
    merge1 = Add()([deconv2, conv1])
    deconv1 = Conv2DTranspose(64, 3, padding='same')(merge1)
    deconv1 = Conv2DTranspose(3, 3, padding='same')(deconv1)
    output = Add()([deconv1, inputs])

    model = Model(inputs, output)
    return model


# モデルコンパイル
def model_compile(model):
    return model.compile(
        loss='mean_squared_error',
        optimizar='adam',
        metrics=[psnr]
    )


# モデルにジェネレータを割り当てる
def model_fit_generator(model, generator, x, y, train_size):
    return model.fit_generator(
        generator,
        validation_data=(x, y),
        step_per_epoch=train_size//BATCH_SIZE,
        epochs=50
    )


if __name__ == '__main__':
    train_size, test_size = split_datas_folder()
    train_data_generator = low_resolution_image_generator(
        DATA_IMAGES_DIR_PATH, TRAIN_DIR[:-1], 2)
    test_x, test_y = next(low_resolution_image_generator(
        DATA_IMAGES_DIR_PATH, TEST_DIR[:-1], 2,
        batch_size=test_size, shuffle=False
    ))
    # モデル
    model = create_model()
    # JSONにモデル構造を保存
    json_name = 'super_resolution/model_struct.json'
    open(json_name, 'w').write(train_model.to_json())
    # コンパイル
    model = model_compile(model)
    # ジェネレータを利用
    model = model_fit_generator(
        model, train_data_generator, test_x, test_y, train_size)
    # 推論
    pred = model.predict(test_x)
    # 重みと損失の保存
    model.save('super_resolution/weights_loss.h5')
