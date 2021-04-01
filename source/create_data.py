import copy
import cv2
import numpy as np
import time
import os

path = os.getcwd()
path = f"{path[:-7]}/data_img/"
## bam b de chup lai background trc khi chup du lieu, de tay vao doi mot lat neu nhay so la dang chup, bam b lan nua de reset lai background neu bi nhieu


# phan chinh sua trc khi tao du lieu
########################################################################
name_gesture = "ho" #ten cua tap du lieu
number_pic = 100  #so luong du lieu

#do sang
exposure = -3 #am la toi, duong la sang them
#do tre moi lan chup
delay = 5 # tinh theo khung hinh cu 20 khung hinh chup lai 1 lan

if not os.path.isdir(f"{path}{name_gesture}"):
    os.mkdir(f"{path}{name_gesture}") #tao folder neu chua tao

#khung hinh vuong de tay vao
ROI_top = 100
ROI_bottom = 350
ROI_right = 350
ROI_left = 600
#######################################################################


threshold = 127  
blurValue = 5 
bgSubThreshold = 50
learningRate = 0

isBgCaptured = 0

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(15,exposure)

num_frames = 0
num_imgs_taken = 0


while camera.isOpened():
    ret, frame = camera.read()
    
    
    frame = cv2.flip(frame, 1) 



    if isBgCaptured == 1:
        begin = time.time()
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
        
        cv2.imshow('ori', result)
        length = len(contours)
        if length > 0:
            if num_frames != 0 and num_frames % delay ==0:
                  if num_imgs_taken < number_pic: #luu file
                      cv2.imwrite(f'{path}{name_gesture}/{name_gesture}_{num_imgs_taken}.jpg', result)
        
                  else:
                      break
                  '''elif num_imgs_taken < number_pic+50:
                       cv2.putText(frame_copy, "Dua tay trai vao o", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                  elif num_imgs_taken < number_pic*2+50:
                      cv2.imwrite(f'data/{name_gesture}/{name_gesture}_{num_imgs_taken-50}.jpg', thresh_img)
                      cv2.putText(frame_copy, f'Da chup {num_imgs_taken+1-50} anh', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)'''
                  num_imgs_taken +=1
            num_frames +=1
        cv2.putText(frame, f'Da chup {num_imgs_taken} anh', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        print(time.time() - begin)

    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break

    elif k == ord('b'):  # press 'b' to capture the background
        num_frames = 0
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('Background captured')

    cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 0, 0), 2)
    
    cv2.imshow('original', frame)