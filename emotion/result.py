import os
import time
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
# from faster_whisper import WhisperModel
import speech_recognition as sr
import wave
import pyaudio
import concurrent.futures

#---------------感情分析---------------------
import librosa
import numpy as np
import pickle
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#---------------感情分析モデル読み込み---------------------

# 韻律感情分析モデルの読み込み
with open('model.pickle', mode='rb') as f:
    clf = pickle.load(f)
# 事前学習済みの日本語感情分析モデルとそのトークナイザをロード
model = AutoModelForSequenceClassification.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment')
tokenizer = AutoTokenizer.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment', model_max_lentgh=512)
# 感情分析のためのパイプラインを設定
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)

#---------------終わり---------------------

HALLUCINATION_TEXTS = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございました。",
    "ありがとうございました", "ありがとうございました。",
    "どうもありがとうございました", "どうもありがとうございました。",
    "どうも、ありがとうございました", "どうも、ありがとうございました。",
    "おやすみなさい", "おやすみなさい。",
    "Thanks for watching!",
    "終わり", "おわり",
    "お疲れ様でした", "お疲れ様でした。",
]

# モデルのロード
# MODEL_SIZE = "large-v3"
# model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
recognizer = sr.Recognizer()

# パラメータ設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 50 # 無音判定のしきい値
SILENCE_DURATION = 0.2  # 無音判定する持続時間 (秒)
OUT_DURATION = 1.0 # 強制的に途中出力する時間(秒)
MIN_AUDIO_LENGTH = 0.1  # 最小音声長 (秒)

# PyAudioインスタンス作成
audio = pyaudio.PyAudio()

# ストリームの設定
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def record_audio(audio_directory):
    frames = []
    recording = False
    silent_chunks = 0
    speak_chunks = 0
    speak_cnt = 1

    audio_directory.mkdir(parents=True, exist_ok=True)

    def is_silent(data):
        # 無音かどうかを判定する関数
        rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
        return rms < SILENCE_THRESHOLD

    def save_wave_file(filename, frames):
        # 録音データをWAVファイルとして保存する関数
        with wave.open(str(filename), 'wb') as wf:  # Pathオブジェクトを文字列に変換
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    while True:
        data = stream.read(CHUNK)
        silent = is_silent(data)

        if silent:
            silent_chunks += 1
            speak_chunks = 0
        else:
            silent_chunks = 0
            speak_chunks += 1
        
        # print("silent_chunks:", silent_chunks, "speak_chunks:", speak_chunks, end="\r")

        if silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
            if recording:
                if len(frames) * CHUNK / RATE < MIN_AUDIO_LENGTH:
                    return
                else:
                    # 無音状態が続いたら録音を停止してファイルを保存
                    file_path = audio_directory / f"recorded_audio_{file_name}_latest.wav"
                    executor.submit(save_wave_file, Path(file_path), frames)
                frames = []
                recording = False
                speak_cnt = 1
        else:
            if not recording:
                file_name = f"{int(time.time())}"
                recording = True

            if speak_chunks > (OUT_DURATION * RATE / CHUNK):
                print("out of duration")
                file_path = audio_directory / f"recorded_audio_{file_name}_{speak_cnt}.wav"
                executor.submit(save_wave_file, Path(file_path), frames)
                speak_cnt += 1
                speak_chunks = 0

            frames.append(data)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_name, file_ext = os.path.splitext(event.src_path)
        # print(f"Created: {event.src_path}")
        if not file_name.endswith("_latest"):
            base_name = file_name.rsplit('_', 1)[0]
            suffix = int(file_name.rsplit('_', 1)[-1])
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{base_name}_latest.wav")):
                # 最終ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return
            if os.path.exists(os.path.join(os.path.dirname(event.src_path), f"{base_name}_{suffix + 1}.wav")):
                # 次ファイルがあるので処理不要、ファイル削除してスキップ
                os.remove(event.src_path)
                return

        # 文字起こしして、ファイルを削除
        self.process_file(event.src_path)
        os.remove(event.src_path)

    def process_file(self, file_path):
        # 文字起こし
        transcription = self.transcribe(file_path)

        # ハルシネーションで出力された可能性のある場合は処理しない
        if transcription in HALLUCINATION_TEXTS:
            return

        if transcription:
            if "latest" in str(file_path):
                # 最終ファイルの場合、そのまま出力
                print(transcription)


            #------------感情分析の予測を行う---------------

            #音声ファイルのパスを取得
                path = file_path
                feature_list = [] #音響的な特徴を格納するリスト
                y, sr = librosa.load(path, sr=16000) #音声ファイルの読み込み
                mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13) #MFCCを取得
                feature_list.append(np.mean(mfcc, axis=1))#各次元の平均を取得

            #モデルを用いた予測
                ans = clf.predict(feature_list)
            # -----------------------結果-------------------------------

            # 分析対象となるテキストのリスト
                texts = transcription
                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                outputs = model(**inputs)
                logits = outputs.logits
            # ロジットを確率に変換
                probabilities = torch.softmax(logits, dim=1)[0]
            # 最も高い確率の感情ラベルを取得
                sentiment_label = model.config.id2label[torch.argmax(probabilities).item()]
                # print(ans)
                # print('テキスト：{}'.format(texts))
                # print('感情：{}'.format(sentiment_label))

            # -----------------------感情によって色判別------------------------------------
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

            # -----------------------終わり------------------------------------

            else:
                # 喋っている途中の文字起こしは《》で囲う
                print("《"+transcription+"》")

    def transcribe(self, file_path):
        try:
            with  sr.AudioFile(file_path) as audio_file:
                audio_data = recognizer.record(audio_file)
                transcription = recognizer.recognize_google(audio_data, language="ja")
            return transcription
        except sr.UnknownValueError:
            # print("音声を認識できませんでした")
            return ""
        except sr.RequestError as e:
            print(f"Google Speech Recognition service にアクセスできませんでした; {e}")
            return ""
        except Exception as e:
            print(f"Error in transcribe: {e}")
            return ""

def start_monitoring(watch_path):
    # 録音処理(スレッドを立てる)
    # wavファイルを生成し続ける処理
    record_thread = Thread(target=record_audio, args=(watch_path,))
    record_thread.daemon = True
    record_thread.start()

    # フォルダを監視してwavファイルが生成された場合
    # 文字起こし処理を行う
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring(Path.cwd() / "tmp")
