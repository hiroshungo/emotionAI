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
svm = LinearSVC()
svm.fit(X_train_std, Y_train)

from sklearn.metrics import accuracy_score

Y_pred_train = svm.predict(X_train_std)
Y_pred_test = svm.predict(X_test_std)
train_accuracy = accuracy_score(Y_train, Y_pred_train)
test_accuracy = accuracy_score(Y_test, Y_pred_test)
print("Train accuracy: {}%, Test accuracy: {}%".format(train_accuracy, test_accuracy))
