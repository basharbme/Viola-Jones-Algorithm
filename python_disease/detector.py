import cv2
import numpy as numpy
import glob
from skimage.io import imread
import os
from skimage.transform import rescale, resize, downscale_local_mean
import random

effusion_cascade = cv2.CascadeClassifier('data/effusion-cascade.xml')
atelectasis_cascade = cv2.CascadeClassifier('classifier/atelectasis-cascade.xml')
cardiomegaly_cascade = cv2.CascadeClassifier('classifier/cardiomegaly-cascade.xml')
effusion_cascade = cv2.CascadeClassifier('classifier/effusion-cascade.xml')
emphysema_cascade = cv2.CascadeClassifier('classifier/emphysema-cascade.xml')
hernia_cascade = cv2.CascadeClassifier('classifier/hernia-cascade.xml')
infiltration_cascade = cv2.CascadeClassifier('classifier/infiltration-cascade.xml')
mass_cascade = cv2.CascadeClassifier('classifier/mass-cascade.xml')
nodule_cascade = cv2.CascadeClassifier('classifier/nodule-cascade.xml')
pneumonia_cascade = cv2.CascadeClassifier('classifier/pneumonia-cascade.xml')
pneumothorax_cascade = cv2.CascadeClassifier('classifier/pneumothorax-cascade.xml')
pleural_thickening_cascade = cv2.CascadeClassifier('classifier/pleural_thickening-cascade.xml')

count = 1
files = []
res = {"neg-true": 0, "neg-false": 0, "pos-true": 0, "pos-false": 0}
for file in glob.iglob("/home/ebraiz/Documents/python_disease/opencv_workspace/*/*.jpg"):
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

    hernia = hernia_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in hernia:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Hernia', (20, 20), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    pneumonia = pneumonia_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in pneumonia:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Pneumonia', (19, 19), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    atelectasis = atelectasis_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in atelectasis:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Atelectasis', (18, 18), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    cardiomegaly = cardiomegaly_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in cardiomegaly:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Cardiomegaly', (17, 17), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    effusion = effusion_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in effusion:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Effusion', (16, 16), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)        

    emphysema = emphysema_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in emphysema:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Emphysema', (15, 15), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    infiltration = infiltration_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in infiltration:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Infiltration', (14, 14), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    mass = mass_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in mass:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Mass', (13, 13), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    nodule = nodule_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in nodule:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Nodule', (12, 12), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    pleural_thickening = pleural_thickening_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in pleural_thickening:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Pleural Thickening', (11, 11), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    pneumothorax = pneumothorax_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=1, minSize=(1,1))
    for (x,y,w,h) in pneumothorax:
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, 'Pneumothorax', (10, 10), font, 0.5, (255,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255,255,0), 2)

    cv2.imshow('img', gray)
    k = cv2.waitKey(500) & 0xff