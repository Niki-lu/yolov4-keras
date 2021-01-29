#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from keras.layers import Input
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from yolo_orgin import YOLO

yolo = YOLO()
#-------------------------------------#
#   调用摄像头
capture=cv2.VideoCapture("./img/2.avi")
#-------------------------------------#
#   capture=cv2.VideoCapture(0)

fps = 0.0
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame,out_classes,out_boxes =yolo.detect_image(frame)

    thickness = max((frame.size[0] + frame.size[1]) // 600, 1)
    font = ImageFont.truetype(font='font/simhei.ttf',
                              size=np.floor(3e-2 * frame.size[1] + 0.5).astype('int32'))
    for i, c in list(enumerate(out_classes)):
        predicted_class = yolo.class_names[c]
        box = out_boxes[i]
        #   score = out_scores[i]

        top, left, bottom, right = box
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

        # 画框框
        # label = '{} {:.2f}'.format(predicted_class, score)
        label = '{}'.format(predicted_class)
        draw = ImageDraw.Draw(frame)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=yolo.colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=yolo.colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    frame = np.array(frame)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.namedWindow("video",0)
    cv2.resizeWindow("video", 1000, 1000)
    cv2.imshow("video",frame)
    c= cv2.waitKey(1) & 0xff
    if c==27:
        capture.release()
        break

yolo.close_session()
    
