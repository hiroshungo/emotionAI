import numpy as np
import glob
import librosa
import pandas

def get_label(path): #感情ラベルを数値に変換
    label_name = path.split("/")[-1].split("_")[2] #ファイル名から感情を取得
    if label_name == "normal":
        label = 0
    elif label_name == "happy":
        label = 1
    elif label_name == "angry":
        label = 2
    else: #想定外の値用
        label = -1
    return label

paths = glob.glob("emotion\\audiofailer/**/*.wav") #音声ファイルのパスを取得
feature_list = [] #音響的な特徴を格納するリスト
label_list = [] #正解データを格納するリスト
for path in paths:
    y, sr = librosa.load(path, sr=16000) #音声ファイルの読み込み
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13) #MFCCを取得
    feature_list.append(np.mean(mfcc, axis=1))#各次元の平均を取得
    label = get_label(path)
    label_list.append(label)

from sklearn.model_selection import train_test_split
Y_train, Y_test, X_train, X_test = train_test_split(label_list, feature_list, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train_std, Y_train)
import pickle
# モデルを保存する
filename = 'emotion\\model.sav'
pickle.dump(model, open(filename, 'wb'))
from sklearn.metrics import accuracy_score

loaded_model = pickle.load(open(filename, 'rb'))
Y_pred_train = loaded_model.predict(X_train_std)
Y_pred_test = loaded_model.predict(X_test_std)

train_accuracy = accuracy_score(Y_train, Y_pred_train)
test_accuracy = accuracy_score(Y_test, Y_pred_test)
print("Train accuracy: {}%, Test accuracy: {}%".format(train_accuracy, test_accuracy))




#追加
# path1 = glob.glob("emotion\\audiofailer\\fujitou_angry\\fujitou_angry_001.wav") #音声ファイルのパスを取得
# feature_list = [] #音響的な特徴を格納するリスト
# label_list = [] #正解データを格納するリスト
# for path2 in path1:
#     y, sr = librosa.load(path2, sr=16000) #音声ファイルの読み込み
#     mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13) #MFCCを取得
#     feature_list.append(np.mean(mfcc, axis=1))#各次元の平均を取得
#     label = get_label(path2)
#     label_list.append(label)
