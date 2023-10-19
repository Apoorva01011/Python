import cv2
import math
import numpy as np
import time
import tensorflow 
import keras
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
detector=HandDetector(maxHands=1)
classifier=Classifier("converted_keras/keras_model.h5","converted_keras/labels.txt")
cap=cv2.VideoCapture(0)
offset=20
imgsize=300
counter=0
labels=["A","B","C"]
folder="data/B"
while(cap.isOpened()):
 res,img=cap.read()
 if(res==True):
    hands,img=detector.findHands(img)
    imgout=img.copy
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
            prediction,index=classifier.getPrediction(imgwhite)   
            print(prediction,index)  
              
        else:
            k=imgsize/w #constant
            hcal=math.ceil(k*h)
            imgresize=cv2.resize(imgcrop,(imgsize,hcal))
            imgresizeshape=imgresize.shape          
            imgwhite[0:imgresizeshape[0],0:imgresizeshape[1]]=imgresize 
            prediction,index=classifier.getPrediction(imgwhite)   
            print(prediction,index)
        cv2.putText(img,labels[index],(x,y),2,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)    
        cv2.imshow('imggg',imgcrop)
        cv2.imshow('img',imgwhite)
    cv2.imshow('img1',img) 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
cap.release()
cv2.destroyAllWindows()   
