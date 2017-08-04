import numpy as np
import cv2

#load in the cascade xml files
FaceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#load images
InputImage = cv2.imread('taher.jpg', cv2.IMREAD_COLOR)
MayImage = cv2.imread('may.jpg', cv2.IMREAD_COLOR)

InputGrey = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
MayGrey = cv2.cvtColor(MayImage, cv2.COLOR_BGR2GRAY)

#find InputFaces in the picture using the cascade we loaded
InputFaces = FaceCascade.detectMultiScale(InputGrey, 1.1, 5)
MayFace = FaceCascade.detectMultiScale(MayGrey, 1.1, 5)
for (x,y,w,h) in InputFaces:
	for (mx, my, mw, mh) in MayFace:
		#draw rectangles over faces detected
		#cv2.rectangle(InputImage, (x,y), (x+w, y+h), (255, 0, 0), 2)
		#cv2.rectangle(MayImage, (mx,my), (mx+mw, my+mh), (255, 0, 0), 2)

		#describe "regions of images" for the faces
		InputRoiGrey = InputGrey[y:y+h, x:x+w]
		InputRoiColour = InputImage[y:y+h, x:x+w]
		MayRoiGrey = MayGrey[my:my+mh, mx:mx+mw]
		MayRoiColour = MayImage[my:my+mh, mx:mx+mw]

		#InputRoiColour[y:y+h, x:x+w] = MayRoiColour[y:y+h, x:x+w]
		#resize and blend image
		face = cv2.resize(MayImage, (h, w), interpolation = cv2.INTER_CUBIC)
		face = cv2.addWeighted(InputImage[y:y+h, x:x+w], 0.1, face, 0.5, 1)
		#swap faces
		InputImage[y:y+h, x:x+w] = face
		
		#find eyes
		# InputEyes = EyeCascade.detectMultiScale(InputRoiGrey)
		# MayEyes = EyeCascade.detectMultiScale(MayRoiGrey)
		# for (ex, ey, ew, eh) in InputEyes:
		# 	for (mex, mey, mew, meh) in MayEyes:
		# 		#draw rectangles over eyes
		# 		cv2.rectangle(InputRoiColour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
		# 		cv2.rectangle(MayRoiColour, (mex, mey), (mex+mew, mey+meh), (0, 255, 0), 2)

#show image in a window
cv2.imshow('James May Facer', InputImage)
#wait for any key to be pressed
cv2.waitKey(0)
cv2.destroyAllWindows()