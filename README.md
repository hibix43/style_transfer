## 画風変換ネットワーク(style transfer network)

特定の画像（スタイル画像）の特徴をつかみ、入力画像に画風を付与する。

画風変換ネットワーク原論文：
(https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)



* style_transfer.py
    * 学習フェーズのメインとなるスクリプト
* convert_network.py
    * 変換ネットワークのスクリプト
* train_network.py
    * 学習ネットワークのスクリプト
    * 損失計算にも利用
* style_images.py
    * スタイル画像に関するスクリプト
    * スタイル画像の特徴量の取得
    * スタイル画像用の損失関数
* contents_images.py
    * コンテンツ画像に関するスクリプト
    * コンテンツ画像の特徴量の取得
    * コンテンツ画像用の損失関数
* style_transfer_test.py
    * 推論フェーズのメインスクリプト
    * 指定フォルダ内の画像をまとめて画風変換することができる