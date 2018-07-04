# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import load_model

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
    predict_image.save('./img/test/test_predict_save.png')
    print('>> Test OK !!')

if __name__ == '__main__':
    model_name = './model/step700_loss26312014036992.0.h5'
    # テスト
    print('>> test start')
    # 入力
    input_shape = (224, 224, 3)
    # 変換ネットワーク
    convert_model = load_model(model_name)
    # 変換
    test(input_shape, convert_model)
