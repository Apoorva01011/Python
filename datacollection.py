import cv2
import math
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
detector=HandDetector(maxHands=1)
cap=cv2.VideoCapture(0)
offset=20
imgsize=300
counter=0
folder="data/B"
while(cap.isOpened()):
 res,img=cap.read()
 if(res==True):
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape=imgcrop.shape
        aspectratio=h/w
        if aspectratio>1:
            k=imgsize/h #constant
            wcal=math.ceil(k*w)
            imgresize=cv2.resize(imgcrop,(wcal,imgsize))
            imgresizeshape=imgresize.shape            
            imgwhite[0:imgresizeshape[0],0:imgresizeshape[1]]=imgresize       
        else:
            k=imgsize/w #constant
            hcal=math.ceil(k*h)
            imgresize=cv2.resize(imgcrop,(imgsize,hcal))
            imgresizeshape=imgresize.shape          
            imgwhite[0:imgresizeshape[0],0:imgresizeshape[1]]=imgresize 
        cv2.imshow('imggg',imgcrop)
        cv2.imshow('img',imgwhite)
    cv2.imshow('img1',img) 
    if cv2.waitKey(1)==ord('s'):
        counter =counter+1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
cap.release()
cv2.destroyAllWindows()   
