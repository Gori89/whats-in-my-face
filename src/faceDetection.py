import cv2
import pandas as pd
import numpy as np
import imutils


def fdetect(image):
    facePos=[]
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    if len(faces)==1 and faces[0,2]>60 and faces[0,3]>90:
        #for face in faces:
        #    (x, y, w, h) = face
        #    facePos.append([x, y, x+w, y+h])
        #return (True,facePos)
        
        (x, y, w, h) = faces[0]
        return (True,[x, y, x+w, y+h])
    else:
        return (False,[])
    