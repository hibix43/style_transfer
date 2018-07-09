# -*- coding: utf-8 -*-

import convert_network
import train_network
import style_images
import contents_images
import glob
import numpy as np
from os import path, makedirs
from math import ceil
from datetime import datetime
from pickle import dump
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.image import (
    load_img, img_to_array, array_to_img)
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.utils import plot_model


CONTENTS_IMAGES_PATH = 'img/contents/*.jpg'
BATCH_SIZE = 2
EPOCH_SIZE = 10
TEST_IMAGE = './img/test/test.jpg'


def build():
    input_shape = (224, 224, 3)
    # 変換ネットワーク
    convert_model = convert_network.build_network()
    print('>> build convert model')
    # 学習ネットワーククラス
    train_net = train_network.TrainNet()
    # 学習ネットワーク
    train_model = train_net.rebuild_vgg16(convert_model.output,
                                          True, True, convert_model.input)
    print('>> build train model')
    # スタイル画像
    style_image = style_images.load_image(input_shape)
    print('>> load style image')
    # スタイル特徴量抽出モデル
    y_style_model = style_images.style_feature(input_shape)
    print('>> build y_style model')
    # スタイル特徴量を抽出する
    y_style_pred = y_style_model.predict(style_image)
    print('>> get y_style_pred')
    # コンテンツ特徴量抽出モデル
    contents_model = contents_images.contents_feature(input_shape)
    print('>> build contents model')
    # ジェネレータ生成
    generator = create_generator(y_style_pred, contents_model)
    print('>> create generator')
    # コンパイル
    train_model = compile_model(train_model)
    print('>> compile train model')
    # 学習
    print('>> train start')
    train(generator, train_model, convert_model)
    print('>> train finish')
    # テスト
    # print('>> test start')
    # test(input_shape, convert_model)
    # print('>> test finish')


# モデルコンパイル
def compile_model(train_model):
    train_model.compile(
        optimizer=Adadelta(),
        loss=[
            style_images.style_feature_loss,
            style_images.style_feature_loss,
            style_images.style_feature_loss,
            style_images.style_feature_loss,
            contents_images.contents_feature_loss
        ],
        loss_weights=[1.0, 1.0, 1.0, 1.0, 3.0]
    )

    return train_model


def make_train_result_datas_dir():
    # 現在の時刻を取得
    now = datetime.now()
    # ディレクトリ名
    log_dir = 'model/{}/log'.format(now)
    weight_loss_dir = 'model/{}/weight_loss'.format(
                       now.strftime('%Y-%m-%d_%H-%M-%S'))
    transfer_dir = 'model/{}/transfer'.format(now)
    # ディレクトリ生成
    makedirs(log_dir, exist_ok=True)
    makedirs(weight_loss_dir, exist_ok=True)
    makedirs(transfer_dir, exist_ok=True)


def train(generator, train_model, convert_model):
    # plot_model(train_model, to_file='tran_model.png')

    # フォルダ作成
    make_train_result_datas_dir()

    # 訓練データ数
    train_imgs = len(get_img_path_list())
    # 1エポックにおけるバッチ処理回数（切り上げ）
    batch_step_per_epoch = ceil(train_imgs / BATCH_SIZE)
    # 学習結果出力周期
    train_output_period = 100
    # 変換テスト出力周期
    test_output_priod = 1000
    # モデル保存周期
    save_model_step = batch_step_per_epoch

    # 進捗バー
    # with tqdm(total=EPOCH_SIZE*batch_step_per_epoch) as pbar:
    # 学習ループ
    for step, (x_train, y_train) in enumerate(generator):
        # 学習
        loss = train_model.train_on_batch(x_train, y_train)
        # 経過出力
        if step % train_output_period == 0:
            print('>> step={} , loss={}'.format(step, loss[0]))
        # 変換テスト
        if step % test_output_priod == 0:
            print('>> Test!! step={} , loss={}'.format(step, loss[0]))
            test(convert_model, step)
        # 保存
        if step % save_model_step == 0:
            # モデル保存
            train_model.save(path.join(weight_dir,
                             'step{}_loss{}.h5'.format(step, loss[0])))
    #        pbar.update(1)


# テスト
def test(covert_model, step, input_shape=(224, 224, 3)):
    # テスト画像読み込み
    test_image = load_img(TEST_IMAGE, target_size=input_shape[:2])
    # 入力用に変換
    test_image = np.expand_dims(img_to_array(test_image), axis=0)
    # 変換
    predict = covert_model.predict(test_image)
    # 保存できる画像に変換
    predict_image = array_to_img(predict[0])
    # 保存
    predict_image.save('./img/test/predicted_test_step{}.png'.format(step))


# ジェネレータの生成
def create_generator(y_style_pred, y_contents_model):
    return train_generator_per_epoch(
                    get_img_path_list(), BATCH_SIZE,
                    y_style_pred, y_contents_model, True, EPOCH_SIZE)


# コンテンツ入力画像のパスをすべて取得
def get_img_path_list():
    img_path = path.join(CONTENTS_IMAGES_PATH)
    img_path_list = glob.glob(img_path)
    return img_path_list


# 1エポックあたりの訓練データジェネレータ
def train_generator_per_epoch(img_path_list, batch_size, y_style_pred,
                              contents_model, shuffle=True, epoches=None):
    # 訓練データ数
    train_imgs = len(img_path_list)
    # 1エポックにおけるバッチ処理回数（切り上げ）
    batch_step_per_epoch = ceil(train_imgs / batch_size)
    # numpy配列化
    if not isinstance(img_path_list, np.ndarray):
        img_path_list = np.array(img_path_list)
    # エポック数
    epoch_counter = 0
    # ジェネレータ
    while True:
        epoch_counter += 1
        # シャッフル
        if shuffle:
            np.random.shuffle(img_path_list)
        # バッチ単位
        for step in range(batch_step_per_epoch):
            # インデックス確保
            start, end = batch_size * step, batch_size * (step + 1)
            # バッチ単位入力画像
            batch_input_images = get_images_array_from_path_list(
                                                   img_path_list[start:end])
            # バッチ単位に拡張
            y_styles_pred = [np.repeat(feature, batch_input_images.shape[0],
                             axis=0) for feature in y_style_pred]
            # コンテンツ特徴量の抽出
            contents_pred = contents_model.predict(batch_input_images)
            # ジェネレータとして値を出力
            yield batch_input_images, y_styles_pred + [contents_pred]

        # エポック数が指定されていて、上限に達した場合
        if epoches is not None and epoch_counter >= epoches:
            # ジェネレータ停止
            raise StopIteration


# 画像パスリストから画像データ配列を得る
def get_images_array_from_path_list(img_path_list, image_size=(224, 224)):
    # img_array   : (N2, N1)
    # expand_dims : (N2, N1) => (1, N2, N1)
    # concatenate : (1, N2, N1) + (1, N2, N1) + ... => (1, N3, N2, N1)
    img_list = [np.expand_dims(
                img_to_array(load_img(img_path, target_size=image_size)),
                axis=0) for img_path in img_path_list]
    return np.concatenate(img_list, axis=0)


if __name__ == '__main__':
    build()
