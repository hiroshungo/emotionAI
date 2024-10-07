import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 事前学習済みの日本語感情分析モデルとそのトークナイザをロード
model = AutoModelForSequenceClassification.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment')
tokenizer = AutoTokenizer.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment', model_max_lentgh=512)

# 感情分析のためのパイプラインを設定
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

# 分析対象となるテキストのリスト
texts = ['あなたを愛してます', '殺す', 'あなたはつまらない']

# 各テキストに対して感情分析を実行
for text in texts:
    print('*'*50)
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits

    # ロジットを確率に変換
    probabilities = torch.softmax(logits, dim=1)[0]

    # 最も高い確率の感情ラベルを取得
    sentiment_label = model.config.id2label[torch.argmax(probabilities).item()]

    print('テキスト：{}'.format(text))
    print('感情：{}'.format(sentiment_label))


    # positiveまたはnegativeの場合はその確率を表示、neutralの場合はpositiveとnegativeの最大値を表示
    if ((sentiment_label == 'positive') or (sentiment_label == 'negative')):  
        print('感情スコア：{}'.format(max(probabilities)))
    else:
        print('感情スコア：{}'.format(max(probabilities[0], probabilities[2])))