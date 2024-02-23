import torch
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import faiss

import pandas as pd
import numpy as np

import random
import os
import time


# 乱数シードの固定
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
  

def odai_to_vector(odai, id, model):
    if id % 100 == 0:
        print(f"Processing {id}th odai")
    # 文ベクトルへ変換
    embeddings = model.encode(odai)
    # 小数点4桁以下は切り捨て
    embeddings = ((embeddings * 10**4).astype(int) / 10**4)
    return embeddings


def create_vecs(odais, model):
    vecs = model.encode(odais)
    vecs = np.array(vecs, dtype=np.float32)
    return vecs


if __name__ == '__main__':
    # 乱数シードの固定
    seed_everything()

    # 学習データの読み込み
    df = pd.read_csv(f'./data/data.csv', on_bad_lines='skip', engine='python')
    
    # odaiカラムの重複を削除
    df = df.drop_duplicates(subset="odai")
    
    # 10件でテスト
    # df = df[:100]
    
    # sentense bert
    MODEL_NAME = "intfloat/multilingual-e5-large"
    transformers.BertTokenizer = transformers.BertJapaneseTokenizer
    bert = models.Transformer(MODEL_NAME)
    pooling = models.Pooling(
            bert.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
    )
    model = SentenceTransformer(modules=[bert, pooling])
    
    # odaiとidを紐付けたcsvを作成
    # df_id = df["id", "odai"]
    # df_id.to_csv("./data/id_odai.csv", index=False)
    
    # 文脈ベクトルを作る
    odais = df["odai"].to_list()
    vecs = create_vecs(odais, model)
    
    # データセットのサイズと次元
    n_data, d = len(vecs), len(vecs[0])

    # FAISSインデックスの作成（ここではL2距離を使用）
    index = faiss.IndexFlatL2(d)

    # ベクトルをインデックスに追加
    index.add(vecs)
    
    # インデックスを保存
    faiss.write_index(index, "./result/index/odai_index.faiss")
