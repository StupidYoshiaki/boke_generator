from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import T5Tokenizer,AutoModelForCausalLM
import torch

import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

import numpy as np
import pandas as pd
import os
import sys

import faiss



def getarate_sentences(seed_sentence, tokenizer, model):
    x = tokenizer.encode(seed_sentence, return_tensors="pt", 
    add_special_tokens=False)  # 入力
    x = x.cuda()  # GPU対応
    y = model.generate(x, #入力
                        min_length=5,  # 文章の最小長
                        max_length=100,  # 文章の最大長
                        do_sample=True,   # 次の単語を確率で選ぶ
                        top_k=50, # Top-Kサンプリング
                        top_p=0.95,  # Top-pサンプリング
                        temperature=1.0,  # 確率分布の調整
                        num_return_sequences=3,  # 生成する文章の数
                        pad_token_id=tokenizer.pad_token_id,  # パディングのトークンID
                        bos_token_id=tokenizer.bos_token_id,  # テキスト先頭のトークンID
                        eos_token_id=tokenizer.eos_token_id,  # テキスト終端のトークンID
                        # bad_word_ids=[[tokenizer.unk_token_id]]  # 生成が許可されないトークンID
                        )  
    generated_sentences = tokenizer.batch_decode(y, skip_special_tokens=True)  # 特殊トークンをスキップして文章に変換
    return generated_sentences


# 類似したお題をベクトル検索
def search_odai(texts, model, index):
    # ベクトル化
    vecs =  model.encode(texts)
    # 最も近いk個のベクトルを検索
    k = 1
    _, I = index.search(vecs, k)  # Dは距離、Iはそのベクトルのインデックス（ID）
    return I
    
    


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # resultディレクトリのディレクトリ数を取得する
    v_num = len(os.listdir("./result/gpt_boke"))
    
    # sentense bert
    MODEL_NAME = "intfloat/multilingual-e5-large"
    transformers.BertTokenizer = transformers.BertJapaneseTokenizer
    bert = models.Transformer(MODEL_NAME)
    pooling = models.Pooling(
            bert.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
    )
    model = SentenceTransformer(modules=[bert, pooling])
    
    # インデックスを読み込む
    index_loaded = faiss.read_index("./result/index/odai_index.faiss")
    
    df = pd.read_csv(f'./data/data_for_index.csv', on_bad_lines='skip', engine='python')
    
    fn = "ippon"
    
    test = []
    with open(f"./data/{fn}.txt", "r") as f:
        texts = f.read().split("\n")
        ids = search_odai(texts, model, index_loaded)
        ids = ids.flatten()
        for id, text in zip(ids, texts):
            # 最もindexが小さいものを選択
            odai = df["odai"][id]
            boke = df["boke"][id]
            # テストテキストを作成
            # text =  "お題:" + odai + " ボケ:" + boke + " お題:" + text + " ボケ:"
            text = " お題:" + text + " ボケ:"
            test.append(text)
        
        # for id, text in zip(ids, texts):
        #     # id
        #     df_tmp = df[df["id"] == id[0]]
        #     print(id)
        #     print(df_tmp)
            # # 最もindexが小さいものを選択
            # odai = df_tmp["odai"]
            # boke = df_tmp["boke"]
            # # テストテキストを作成
            # text =  "お題:" + odai + " ボケ:" + boke + " お題:" + text + " ボケ:"
            # test.append(text)
        
    # sys.exit()    
    
    # gptの読み込み
    model_name = f"./result/gpt_boke/v{v_num}/checkpoint-36730"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # testの重複を削除
    test = list(set(test))

    # あいうえお順にソート
    test.sort()

    with open(f"./result/gpt_boke/v{v_num}/boke_generated_{fn}.txt", "w") as f:
        for odai in test:
            seed_sentence = odai
            generated_sentences = getarate_sentences(seed_sentence, tokenizer, model)
            for sentence in generated_sentences:
                f.write(sentence + "\n")
          
            
if __name__ == "__main__":
    main()