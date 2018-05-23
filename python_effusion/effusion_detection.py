import cv2
import numpy as numpy
import glob
from skimage.io import imread
import os
from skimage.transform import rescale, resize, downscale_local_mean
import random

effusion_cascade = cv2.CascadeClassifier('data/effusion-cascade.xml')

count = 1
files = []
res = {"neg-true": 0, "neg-false": 0, "pos-true": 0, "pos-false": 0}
for file in glob.iglob("/home/ebraiz/Documents/python_effusion/opencv_workspace/pos/*.jpg"):
    gray = imread(file)

    flag = False
    effusion = effusion_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in effusion:
        flag = True
        
    s = "pos" if "pos" in file else "neg"
    if flag:
        s = s+"-true"
        res[s] += 1
    else:
        s = s+"-false"
        res[s] += 1

    if s in ("pos-true", "neg-false"):
        files.append(file)

true = res['neg-false'] + res['pos-true']
false = res['neg-true'] + res['pos-false']
total = true+false
print(res)
print("True: ", true)
print("False: ", false)
print("Total: ", total)
print ((true/float(total))*100.0)
random.shuffle(files)

print ("Correct files: ", len(files))
for file in files:
    gray = imread(file)

    effusion = effusion_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in effusion:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Effusion', (20, 20), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    cv2.imshow('img', gray)
    k = cv2.waitKey(500) & 0xff