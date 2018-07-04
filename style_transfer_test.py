#!/usr/bin/python
# -*- Coding: utf-8 -*-



# テスト
def test(input_shape, model):
    # テスト画像読み込み
    test_image = load_img(TEST_IMAGE, target_size=input_shape[:2])
    # 入力用に変換
    test_image = np.expand_dims(img_to_array(test_image), axis=0)
    # 変換
    predict = model.predict(test_image)
    # 保存できる画像に変換
    # predict_image = array_to_img(predict[0][1:3])
    # Image.fromarray(predict_image).save('./img/test/test_predict_save.png')
    print('>> Test OK !!')