# -*- coding: utf-8 -*-

import convert_network
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import (
    load_img, img_to_array, array_to_img)


TEST_IMAGE = './img/test/test.jpg'


# テスト
def test(input_shape, model):
    # テスト画像読み込み
    test_image = load_img(TEST_IMAGE, target_size=input_shape[:2])
    # 入力用に変換
    test_image = np.expand_dims(img_to_array(test_image), axis=0)
    # 変換
    predict = model.predict(test_image)
    print('>> predict result shape={}'.format(predict[0].shape))
    # 保存できる画像に変換
    predict_image = array_to_img(predict[0])
    # 保存
    predict_image.save('./img/test/test_predict.png')
    print('>> Test OK !!')

if __name__ == '__main__':
    model_name = './model/step26750_loss1373626302464.0.h5'
    # テスト
    print('>> test start')
    # 入力
    input_shape = (224, 224, 3)
    # 変換ネットワーク
    conver_model = convert_network.build_network(input_shape=input_shape)
    # ネットワーク
    model = load_model(model_name,
                       custom_objects={'input_1': conver_model.output})
    # 変換
    test(input_shape, convert_model)
