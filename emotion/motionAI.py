#------------感情分析---------------

import librosa
import numpy as np
#音声ファイルのパスを取得
path = "emotion\\audiofailer\\tsuchiya_angry\\tsuchiya_angry_006.wav"
feature_list = [] #音響的な特徴を格納するリスト
y, sr = librosa.load(path, sr=16000) #音声ファイルの読み込み
mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13) #MFCCを取得
feature_list.append(np.mean(mfcc, axis=1))#各次元の平均を取得

#モデルの読み込み
import pickle
with open('model.pickle', mode='rb') as f:
    clf = pickle.load(f)
#モデルを用いた予測
ans = clf.predict(feature_list)
# 結果
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 事前学習済みの日本語感情分析モデルとそのトークナイザをロード
model = AutoModelForSequenceClassification.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment')
tokenizer = AutoTokenizer.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment', model_max_lentgh=512)

# 感情分析のためのパイプラインを設定
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

# 分析対象となるテキストのリスト
texts = ['私は嬉しい']

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
outputs = model(**inputs)
logits = outputs.logits

    # ロジットを確率に変換
probabilities = torch.softmax(logits, dim=1)[0]

    # 最も高い確率の感情ラベルを取得
sentiment_label = model.config.id2label[torch.argmax(probabilities).item()]
    

print(ans)
print('テキスト：{}'.format(texts))
print('感情：{}'.format(sentiment_label))

if sentiment_label == "positive" :
    #楽しい
    if ans == [2]:
        color = "黄"
        print(color)
    #楽しいと思っているけどまだ韻律が出ていない
    if ans == [1]:
        color = "オレンジ"
        print(color)
    #楽しいと思っているけど韻律があっていない
    if ans == [0]:
        color = "緑"
        print(color)

if sentiment_label == "negative" :
    #怒り
    if ans == [2]:
        color = "赤"
        print(color)
    #嫌悪
    if ans == [1]:
        color = "紫"
        print(color)
    #悲しみ
    if ans == [0]:
        color = "青"
        print(color)

#-----------------終わり-----------------


# if label_name == "normal":
#         label = 0
#     elif label_name == "happy":
#         label = 1
#     elif label_name == "angry":
#         label = 2
#     else: #想定外の値用
#         label = -1
#     return label
