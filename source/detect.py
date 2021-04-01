#!/usr/bin/python
# -*- coding: utf8 -*-
import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import os
import threading
import NLP as nlp
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from playsound import playsound


path = os.getcwd()
path = path[:-7]

model = load_model(f'{path}/models_img/model_10.h5')
bgModel = None

ROI_top = 100
ROI_bottom = 350
ROI_right = 350
ROI_left = 600

threshold = 127  
blurValue = 21
bgSubThreshold = 50
learningRate = 0



gesture_names = {0:"'",
                1: 'A',
                2: 'B',
                3: 'C',
                4: 'D',
                5: 'Đ',
                6: 'E',
                7: 'G',
                8: 'H',
                9: 'I',
                10: 'K',
                11: 'L',
                12: 'M',
                13: 'N',
                14: 'O',
                15: 'P',
                16: 'Q',
                17: 'R',
                18: 'S',
                19: 'T',
                20: 'U',
                21: 'V',
                22: 'X',
                23: 'Y',
                24: '^',
                25: 'hoi',
                26: 'huyen',
                27: 'nang',
                28: 'nga',
                29: 'sac'}


vowel = {"A": {"'": u"Ă",
               "^": u"Â",
               "sac": u"Á",
               "huyen": u"À",
               "hoi": u"Ả",
               "nga": u"Ã",
               "nang": u"Ạ"},
        "Ă": {"sac": u"Ắ",
              "huyen": u"Ằ",
              "hoi": u"Ẳ",
              "nga": u"Ẵ",
              "nang": u"Ặ"},
        "Â": {"sac": u"Ấ",
              "huyen": u"Ầ",
              "hoi": u"Ẩ",
              "nga": u"Ẫ",
              "nang": u"Ậ"},
        "E": {"^": u"Ê",
              "sac": u"É",
              "huyen": u"È",
              "hoi": u"Ẻ",
              "nga": u"Ẽ",
              "nang": u"Ẹ"},
        "Ê": {"sac": u"Ế",
              "huyen": u"Ề",
              "hoi": u"Ể",
              "nga": u"Ễ",
              "nang": u"Ệ"},
        "O": {"'": u"Ơ",
              "^": u"Ô",
              "sac": u"Ó",
              "huyen": u"Ò",
              "hoi": u"Ỏ",
              "nga": u"Õ",
              "nang": u"Ọ"},
        "Ơ": {"sac": u"Ớ",
              "huyen": u"Ờ",
              "hoi": u"Ở",
              "nga": u"Ỡ",
              "nang": u"Ợ"},
        "Ô": {"sac": u"Ố",
              "huyen": u"Ồ",
              "hoi": u"Ổ",
              "nga": u"Ỗ",
              "nang": u"Ộ"},
        "U": {"'": u"Ư",
              "sac": u"Ú",
              "huyen": u"Ù",
              "hoi": u"Ủ",
              "nga": u"Ũ"},
        "Ư": {"sac": u"Ứ",
              "huyen": u"Ừ",
              "hoi": u"Ử",
              "nga": u"Ữ"},
        }

accent = ['hoi','huyen','nang', 'nga','sac']
special = ["'",'^']

def predict_rgb_image_vgg(image):

    
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score


def remove_background(frame):
    global bgModel
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera


String = u""
Temp = []

speak = False

def camera_detect():
    global String, Temp, speak
    global path, bgModel, ROI_top, ROI_bottom, ROI_right, ROI_left
    global threshold, blurValue, bgSubThreshold
    global gesture_names, vowel, accent, special
    global thread2

    isBgCaptured = 0
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    camera.set(15,-4)

    font = ImageFont.truetype(f"{path}/font.ttf", 32)
    while camera.isOpened():
        ret, frame = camera.read()
        
        
        frame = cv2.flip(frame, 1) 



        if isBgCaptured == 1 and not speak:
            t = time.time()
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
            roi = cv2.bilateralFilter(roi, 5, 50, 100) 
            img = remove_background(roi)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            stencil = np.zeros(test_image.shape).astype(test_image.dtype)
            cv2.fillPoly(stencil, contours, [255,255,255])
            result = cv2.bitwise_and(test_image, stencil)
            cv2.imshow('ori',result)



            length = len(contours)
            if length > 0:
                target = np.stack((result,) * 3, axis=-1)
                target = cv2.resize(target, (250, 250))
                target = target.reshape(1, 250, 250, 3)
                prediction, score = predict_rgb_image_vgg(target)

                # Neu probality > nguong du doan thi hien thi
                if (score>=95):
                    Temp.append(prediction)
                    cv2.putText(frame, "Sign:" + prediction, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    dic = dict((i, Temp.count(i)) for i in Temp)
                    if len(Temp) > 20:
                        Temp = []
                    if not any(x in accent for x in dic):
                        max_key = max(dic, key=dic.get)
                        if max_key not in accent:
                            if dic[max_key] >=10:
                                if String and max_key != String[-1]:
                                    if max_key in special and String[-1] in vowel and max_key in vowel[String[-1]]:
                                        String = String[:-1] + vowel[String[-1]][max_key]
                                    elif max_key not in special:
                                        String +=max_key
                                elif not String:
                                    String +=max_key
                                Temp = []
                    else:
                        dic = dict((i, Temp.count(i)) for i in accent)
                        max_key = max(dic, key=dic.get)
                        if dic[max_key] >=5:
                            if String and String[-1] in vowel:
                                    String = String[:-1] + vowel[String[-1]][max_key]
                            Temp = []
                stop = time.time()
            else:
                if time.time()-stop >= 1.5:
                    if String:
                        if String[-1] != " ":
                            String += " "
                    stop = time.time()
                Temp = []
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((50, 100),  u"Text:" + String,font = font, fill = (0,255,0,0))
            frame = np.array(img_pil)


        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit all windows at any time
            break
        elif k == 8:
            if String:
              if len(String) > 1:
                  String = String[:-1]
              else:
                  String = ""

        elif k == ord('b'):  # press 'b' to capture the background
            num_frames = 0
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print('Background captured')
        elif k == 13 and String and not speak:
            speak = True
            thread2.start()
        if not speak:    
            cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Waiting...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3, lineType=cv2.LINE_AA)
        cv2.imshow('original', frame)

def Speak():
  global speak,String,Temp
  global thread2
  if speak:
      result = nlp.edit(String)
      String = ""
      Temp =[]
      tts = gTTS(text =result,lang='vi')
      tts.save("speech.mp3")
      playsound("speech.mp3")
      os.remove("speech.mp3")
      speak = False
      thread2 = threading.Thread(name='2', target=Speak)


try:
  thread1 = threading.Thread(name='1', target=camera_detect)
  thread2 = threading.Thread(name='2', target=Speak)
 
  thread1.start()
  
except:
  print ("error")