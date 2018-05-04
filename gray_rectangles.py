import cv2
import utils

def rgb2gray(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def rgb2hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
def gray2rgb(img):
	return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#def main():


if __name__ == "__main__":



	img = cv2.imread('indexc.jpeg')
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	#cv2.imwrite('indexc_gray.jpeg',cv2.cvtColor(utils.augment(img, 0, 2), cv2.COLOR_RGB2GRAY))

	cv2.imwrite('indexc_gray.jpeg', img)