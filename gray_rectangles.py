import cv2
import utils
import os

import numpy as np



# def rgb2gray(img):
# 	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



def threshold_slow(T, image):
	# grab the image dimensions
	h = image.shape[0]
	w = image.shape[1]
	
	# loop over the image, pixel by pixel
	for y in range(0, h):
		for x in range(0, w):
			# threshold the pixel
			image[y, x] = 255 if image[y, x] >= T else 0
			
	# return the thresholded image
	return image




def threshold_slow_delta(T, image, delta_height, delta_width):
	# grab the image dimensions
	h = image.shape[0]
	w = image.shape[1]

	delta_height_prev = 0
	delta_width_prev = 0

	brightness_sum = 0

	while delta_width_prev < w-2:

		while delta_height_prev < h-2:
		
			# loop over the image, pixel by pixel
			for y in range(delta_height_prev, delta_height_prev + delta_height):
				for x in range(delta_width_prev, delta_width_prev + delta_width):
					# threshold the pixel
					#image[y, x] = 255 if image[y, x] >= T else 0
					brightness_sum = brightness_sum + image[y, x]
			

			brightness_avg = brightness_sum/(delta_height * delta_width)
			brightness_sum = 0
			gray_scale = 255 if brightness_avg >= T else 0

			# loop over the image, pixel by pixel
			for y in range(delta_height_prev, delta_height_prev + delta_height):
				for x in range(delta_width_prev, delta_width_prev + delta_width):
					image[y, x] = gray_scale

			delta_height_prev = delta_height_prev + delta_height

		delta_width_prev = delta_width_prev + delta_width
		delta_height_prev = 0








	# return the thresholded image
	return image






if __name__ == "__main__":


	path_file = "indexc.jpeg"


	img = cv2.imread(path_file)

	cvt_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	#cv2.imwrite('indexc_gray.jpeg', img)


	#cvt_image = threshold_slow(127, cvt_image)

	height, width = cvt_image.shape
	print(height, width)


	delta_height = height/50
	delta_height = round(delta_height, 0)
	delta_height = int(delta_height)
	delta_width = width/50
	delta_width = round(delta_width, 0)
	delta_width = int(delta_width)


	cvt_image= threshold_slow_delta(127, cvt_image, delta_height, delta_width)


	cv2.imshow('image', cvt_image)



	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
		print(height, width)
		print(delta_height, delta_width)
		cv2.imwrite('indexc_gray_chank.jpeg', cvt_image)
		#print(brightness_sum)
		#print(brightness_avg )





	# raw_image = cv2.imread(path_file)
	# cv2.imshow('raw image',raw_image) 

	# #get the size of pic(row & column)
	# rows = raw_image.shape[0]
	# cols = raw_image.shape[1] 

	# #create a null array and be filled in with random number
	# image = np.zeros(shape=(rows,cols,3), dtype=np.uint8) 
	# for r in range(rows): 
	# 	for c in range(cols):
	# 		#print(image[r, c, 0])
	# 		print(image[r, c, 1])
	# 		break
	# #         image[r, c, 0] = np.random.randint(0, 255)
	# #         image[r, c, 1] = np.random.randint(0, 255)
	# #         image[r, c, 2] = np.random.randint(0, 255)

	# # cv2.imshow('random pixel image', image)
	

	# while (True):
	# 	k=cv.waitKey(0)
	# 	input('')
	# 	if k==ord('e'):
	# 		break

	# cv2.destroyAllWindows()

 
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



	# photopath = 'indexc.jpeg'
	# classifier = os.getcwd()+'/haarcascade_frontalface_default.xml'

 

	# image = cv2.imread(photopath)


	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	# face_casacade = cv2.CascadeClassifier(classifier)


	# faces = face_casacade.detectMultiScale(image)


	# color = (0,0,255)
	# strokeWeight = 1
	
	# windowName = "Object Detection"

	# while True: 

	# 	print(len(faces))
	# 	for x, y, width, height in faces:
	# 		cv2.rectangle(image, (x, y), (x + width, y + height), color, strokeWeight)

	
	# 	cv2.imshow(windowName, image)

		
	# 	if cv2.waitKey(20) == 27:
	# 		break

	
	# exit()