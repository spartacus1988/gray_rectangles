import cv2
import utils
import os

import numpy as np



# def rgb2gray(img):
# 	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



if __name__ == "__main__":


	path_file = "indexc.jpeg"

	#img = cv2.imread(path_file)
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	#cv2.imwrite('indexc_gray.jpeg', img)


	# cvt_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# cv2.imshow('cvtColor image', cvt_image)
	# cv2.waitKey(0)
	# input('')
	# cv2.destroyAllWindows()


	# image = cv2.imread(path_file)
	# rows = image.shape[0]
	# cols = image.shape[1] 

	# for row in range(rows): 
	# 	for col in range(cols):
	# 		average = np.mean(image[row,col,:])
	# 		image[row, col, :] = average
	# cv2.imshow('average image', image)
	# cv2.waitKey(0)
	# input('')
	# cv2.destroyAllWindows()


	# image = cv2.imread(path_file)
	# rows = image.shape[0]
	# cols = image.shape[1] 

	# for row in range(rows): 
	# 	for col in range(cols):
	# 		gray = 0.11*image[row,col,0]+0.59*image[row,col,1]+0.3*image[row,col,2]
	# 		image[row, col, :] = gray
	# cv2.imshow('formula image', image)
	# cv2.waitKey()
	# cv2.destroyAllWindows()



	photopath = 'indexc.jpeg'
	classifier = os.getcwd()+'/haarcascade_frontalface_default.xml'


	image = cv2.imread(photopath)


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	face_casacade = cv2.CascadeClassifier(classifier)


	faces = face_casacade.detectMultiScale(image)


	color = (0,0,255)
	strokeWeight = 1
	
	windowName = "Object Detection"

	while True: 

		print(len(faces))
		for x, y, width, height in faces:
			cv2.rectangle(image, (x, y), (x + width, y + height), color, strokeWeight)

	
		cv2.imshow(windowName, image)

		
		if cv2.waitKey(20) == 27:
			break

	
	exit()