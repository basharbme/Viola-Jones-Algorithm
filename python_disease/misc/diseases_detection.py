import cv2
import numpy as numpy
import glob
from skimage.io import imread

diseases_cascade = cv2.CascadeClassifier('All-diseases-cascade-500-2-Stages.xml')

cap = cv2.VideoCapture(0)

for file in glob.iglob("/home/ebraiz/Documents/python_disease/opencv_workspace/neg/*.jpg"):
    img = gray = imread(file)

    diseases = diseases_cascade.detectMultiScale(gray, 30, 30)
    
    for (x,y,w,h) in diseases:
    	font = cv2.FONT_HERSHEY_SIMPLEX
    	cv2.putText(img, 'Disease', (x+w, y+h), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(500) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()             
