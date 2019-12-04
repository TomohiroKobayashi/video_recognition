import sys
import argparse
from PIL import Image

def detect_video(video_path="/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/研究用コード/動画自動切り出し/data/temp/2018_04_18_14_52_27_crop.mp4", output_path=""):
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
    while True:
        return_value, frame = vid.read()
        if count >= 1819047 and n <= 100:
            if (count+3) % 5 == 0:
                if type(frame) == type(None):
                    break
                image = Image.fromarray(frame)
                image.save("その場足踏み/" + str(count) + ".jpg")
                n += 1

        if count >= 2477127:
            if (count+3) % 5 == 0:
                if type(frame) == type(None):
                    break
                image = Image.fromarray(frame)
                image.save("首振り/" + str(count) + ".jpg")
                n_2 += 1
                if n_2 >= 100:
                    break

        count += 1
        if count % 10000==0:
            print(count)
detect_video()
