# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
 Conv2D, BatchNormalization, Add, Activation, Input, Lambda, Conv2DTranspose)


def build_residual_block(input_data):
    x = Conv2D(128, (3, 3), strides=1, padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same')(input_data)
    x = BatchNormalization()(x)
    # Skip Connection
    return Add()([x, input_data])


def build_encoder(input_data):
    # [0, 1]に正則化
    x = Lambda(lambda x: x / 255.0)(input_data)

    # 構築
    x = Conv2D(32, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def build_decoder(input_data):
    x = input_data
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # [0, 255]にスケール変換
    output_data = Lambda(lambda xx: (xx + 1) * 127.5)(x)

    return output_data


def build_network(input_shape=(224, 224, 3)):
    # 入力データ
    input_data = Input(shape=input_shape, name='input')
    # エンコーダ
    x = build_encoder(input_data)
    # ブロック作成
    for i in range(5):
        x = build_residual_block(x)
    # デコーダ
    output_data = build_decoder(x)
    # モデル構築
    convert_model = Model(
        inputs=[input_data],
        outputs=[output_data]
    )
    return convert_model
