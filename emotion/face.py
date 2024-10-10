# pythonライブラリのFERを用いる
from fer import FER
import matplotlib.pyplot as plt
import pandas

# 感情分析に使う画像を読み込む
test_image_one = plt.imread("emotion\\face.jpg")

emo_detector = FER(mtcnn=True)
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
plt.imshow(test_image_one)
print(dominant_emotion, emotion_score)

