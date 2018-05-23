import cv2
import numpy as numpy

cascade = cv2.CascadeClassifier('cascade.xml')

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
	ret, img = cap.read()

	if ret:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cas = cascade.detectMultiScale(gray, 30, 30)
		
		for (x,y,w,h) in cas:
			cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)

	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()				
