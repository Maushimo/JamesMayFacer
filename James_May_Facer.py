import numpy as np
import cv2

#load in the cascade xml files
FaceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#load images
InputImage = cv2.imread('taher.jpeg', cv2.IMREAD_COLOR)
MayImage = cv2.imread('may.jpg', cv2.IMREAD_COLOR)

InputGrey = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
MayGrey = cv2.cvtColor(MayImage, cv2.COLOR_BGR2GRAY)

#find InputFaces in the picture using the cascade we loaded
InputFaces = FaceCascade.detectMultiScale(InputGrey, 1.1, 5)
MayFace = FaceCascade.detectMultiScale(MayGrey, 1.1, 5)
for (x,y,w,h) in InputFaces:
	#draw rectangles over InputFaces detected
	cv2.rectangle(InputImage, (x,y), (x+w, y+h), (255, 0, 0), 2)
	#cv2.rectangle(MayImage, (mx,my), (mx+mw, my+mh), (255, 0, 0), 2)
	InputRoiGrey = InputGrey[y:y+h, x:x+w]
	InputRoiColour = InputImage[y:y+h, x:x+w]
	
	#find eyes
	eyes = EyeCascade.detectMultiScale(InputRoiGrey)
	for (ex, ey, ew, eh) in eyes:
		#draw rectangles over eyes
		cv2.rectangle(InputRoiColour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


#show image in a window
cv2.imshow('James May Facer', InputImage)
#wait for any key to be pressed
cv2.waitKey(0)
cv2.destroyAllWindows()