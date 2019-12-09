import sys
import argparse
from PIL import Image
import os
import glob

#yoloを呼び出すためにパスを追加
sys.path.append("/Users/kobayashitomohiro/Document/大学/研究室/馬の分娩予測研究/研究用コード/keras-yolo3")
from yolo import YOLO, detect_video


def detect_img(yolo,img):
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        #continueはwhile文で無限ループしてる時だけ
        #continue
    else:
        r_image = yolo.detect_image(image)

        im_name = os.path.splitext(img)
        im_name = im_name[0][24:]
        #画像を保存
        print(type(r_image))
        print(im_name)
        r_image.save("31_done/" + im_name +'_done.png')
        #r_image.show()
    #yolo.close_session()


FLAGS = None
if __name__ == '__main__':
    folder = "annotation_snowflower/31"
    files = glob.glob(folder + "/*.jpg")
    for file in files:
        img = file
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        parser.add_argument(
            '--image', default="False", action="store_true",
            help='Image detection mode, will ignore all positional arguments'
        )
        FLAGS = parser.parse_args()
        output = detect_img(YOLO(**vars(FLAGS)),img)
        #detect_img(YOLO({"image":img}))
        """
        try:
            output.save("17_done"+file)
        except:
            continue
        """
