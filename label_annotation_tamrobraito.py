import sys
import argparse
from PIL import Image

def detect_video(video_path="/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/研究用コード/動画自動切り出し/data/temp/2018 04 09 10 21 33 タムロプライト-4.mp4", output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    count = 1
    n = 0
    n_2 = 0

    #エクセルファイルを読みこむ
    import pandas as pd

    #スノフラワー0シート
    data1 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/タムロブライト移動距離など.xlsx", sheet_name="タムロブライト０")
    #エクセルの使用する部分をスライシング
    data1=data1.ix[13:,:]
    #シーン数とラベルを取り出し
    df1 = data1.ix[:,[1,8]]
    #カラム名を変更
    df1 = df1.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

    #最後の最小値などのセルを削除
    df1 = df1.head(len(df1)-5)
    #インデックス番号をリセット
    df1.reset_index()
    #シーン番号の最初うちと最大値
    min_1 = df1.iloc[0, [0]].iat[0]
    max_1 = df1.iloc[len(df1)-1, [0]].iat[0]

    #行動ラベルをリスト化
    df1_list = df1['label'].values.tolist()

    #スノフラワー1シート
    data2 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/タムロブライト移動距離など.xlsx", sheet_name="タムロブライト－１")
    #エクセルの使用する部分をスライシング
    data2=data2.ix[13:,:]
    #シーン数とラベルを取り出し
    df2 = data2.ix[:,[1,8]]
    #カラム名を変更
    df2 = df2.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

    #最後の最小値などのセルを削除
    #今回のケースのみ-6 他の場合は-5でOK!
    df2 = df2.head(len(df2)-6)
    #インデックス番号をリセット
    df2.reset_index()
    #シーン番号の最初うちと最大値
    min_2 = df2.iloc[0, [0]].iat[0]
    max_2 = df2.iloc[len(df2)-2, [0]].iat[0]
    print(type(min_2))
    print(type(max_2))

    #行動ラベルをリスト化
    df2_list = df2['label'].values.tolist()

    data3 = pd.read_excel("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/諏訪牧場/タムロブライト移動距離など.xlsx", sheet_name="タムロブライト－２")
    #エクセルの使用する部分をスライシング
    data3=data3.ix[13:,:]
    #シーン数とラベルを取り出し
    df3 = data3.ix[:,[1,8]]
    #カラム名を変更
    df3 = df3.rename(columns={'Unnamed: 1': 'scene', 'Unnamed: 8': 'label'})

    #最後の最小値などのセルを削除
    df3 = df3.head(len(df3)-5)

    #インデックス番号をリセット
    df3.reset_index()

    min_3 = df3.iloc[0, [0]].iat[0]
    max_3 = df3.iloc[len(df3)-1, [0]].iat[0]
    #行動ラベルをリスト化
    df3_list = df3['label'].values.tolist()
    print(str(max_3-min_3+1) + "==" + str(len(df3_list)))
    print(max_3-min_3)

    import os

    if not os.path.exists("annotation"):
        os.mkdir("annotation")

    n = 1

    while True:
        return_value, frame = vid.read()
        #まずはシート2から画像をラベル別に保存
        #>=から>に変更してみた
        #あと考えられる原因はwhile文の中だとうまくいかないこと
        if count >= min_3 and count <=max_3 and (count+3) % 5 == 0:

            if type(frame) == type(None):
                break
            image = Image.fromarray(frame)
            #行動ラベルの名前のフォルダに画像を保存する
            #もしそのフォルダが存在しなければ作成
            #下記ではエラー発生。while文の中でcount(変数)を使わない方がいいみたい
            #label = str(df3[df3['scene']==count]['label'].iat[0])
            #変数を使わなければいける説（できた）
            #おそらくインデックス1から始まってる
            #label = str(df3_list[count-min_3])
            label = str(df3_list.pop(0))

            if not os.path.exists("annotation/"+ label):
                os.mkdir("annotation/" + label)
            image.save("annotation/" + label + "/" + str(count) + ".jpg")

        if count >= min_2 and count <=max_2 and (count+3) % 5 == 0:
            if type(frame) == type(None):
                break
            image = Image.fromarray(frame)
            #行動ラベルの名前のフォルダに画像を保存する
            #もしそのフォルダが存在しなければ作成
            #下記ではエラー発生。while文の中でcount(変数)を使わない方がいいみたい
            #label = str(df3[df3['scene']==count]['label'].iat[0])
            #変数を使わなければいける説（できた）
            #label = str(df2_list[count-min_2])
            label = str(df2_list.pop(0))

            if not os.path.exists("annotation/"+ label):
                os.mkdir("annotation/" + label)
            image.save("annotation/" + label + "/" + str(count) + ".jpg")

        if count >= min_1 and count <=max_1 and (count+3) % 5 == 0:
            if type(frame) == type(None):
                break
            image = Image.fromarray(frame)
            #行動ラベルの名前のフォルダに画像を保存する
            #もしそのフォルダが存在しなければ作成
            #下記ではエラー発生。while文の中でcount(変数)を使わない方がいいみたい
            #label = str(df3[df3['scene']==count]['label'].iat[0])
            #変数を使わなければいける説（できた）
            #label = str(df1_list[count-min_1])
            label = str(df1_list.pop(0))

            if not os.path.exists("annotation/"+ label):
                os.mkdir("annotation/" + label)
            image.save("annotation/" + label + "/" + str(count) + ".jpg")

        count += 1
        if count % 10000==0:
            print(count)

        if count > (max_1+10):
            break
detect_video()
