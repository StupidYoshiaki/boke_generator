# boke_generater
大喜利のお題から回答を生成する

## 概要
100万件ほどの大喜利のお題と回答を集めたデータを用いてGPT-2モデルを学習させる。

推論時には、与えられたお題に最も類似した学習データにあるお題を参考に回答を出力させる。  
方針としてはRAGに近い。

## 流れ
大喜利お題回答データベースにあるお題をsentence-transformersモデル（intfloat/multilingual-e5-large）を用いてベクトル化。  
ベクトル検索の高速化を図るため、faissを用いてそれら文ベクトルのインデックスを作成。

データベースを用いてGPT-2モデル（rinna/japanese-gpt2-medium）を学習。

推論時に与えられたお題から類似したものを、先ほどのインデックスから検索。  
類似したお題と評価の高いボケを与えられたお題に組み込むことで、より適当な回答を出力するように設定。