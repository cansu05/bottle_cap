import cv2
import numpy as np
import os
from skimage.filters import try_all_threshold, threshold_otsu, threshold_isodata, threshold_mean
import matplotlib.pyplot as plt

def rectangle(img):
	cv2.rectangle(img,(290,435),(800,1045),(0,255,0),3)
	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.imshow("img",img)
	# cv2.waitKey(0)
	return img


def crop_image(img, y, height, x, width):
    cropped_part = img[y:y+height, x:x+width]
    cv2.namedWindow("cropped_part", cv2.WINDOW_NORMAL)
    cv2.imshow("cropped_part",cropped_part)
    # cv2.waitKey(0)
    return cropped_part

def adjust_brightness(img, value):
	num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	if value >= 0:
		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] += value
	else:
		value = int(-value)
		lim = 0 + value
		v[v < lim] = 0
		v[v >= lim] -= value

	final_hsv = cv2.merge((h, s, v))

	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img

	return img


def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			highlight = 255
		else:
			shadow = 0
			highlight = 255 + brightness
		alpha_b = (highlight - shadow)/255
		gamma_b = shadow
		
		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
	else:
		buf = input_img.copy()

	if contrast != 0:
		f = 131*(contrast + 127)/(127*(131-contrast))
		alpha_c = f
		gamma_c = 127*(1-f)
		
		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

	return buf


def hough_circle(img, bottle_size):

	image_original = img.copy()
	if bottle_size == "1L": img = cv2.medianBlur(img,9)
	elif bottle_size == "3L": img = cv2.medianBlur(img,9)
	else: img = cv2.medianBlur(img,7)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) #5L7L th50
	# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=120, param2=20, minRadius=100, maxRadius=0)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=120, param2=20, minRadius=100, maxRadius=0)
	


	if circles is not None:
		
	   # convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
       
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(image_original, (x, y), r, (0, 255, 0), 4)
			# print("---x:", x, "y:", y, "r:", r)


			startpoint = (int(x+(r/(2**0.5))),int(y-(r/(2**0.5))))
			endpoint = (int(x-(r/(2**0.5))),int(y+(r/(2**0.5))))
			# print(startpoint, endpoint)

			imageWithInscribingSquare = cv2.rectangle(image_original, startpoint, endpoint, (255, 0, 0) , 2)
			
			crop_image(imageWithInscribingSquare, y=startpoint[1], height=endpoint[1]-startpoint[1], x=endpoint[0], width= startpoint[0]-endpoint[0])
			
			cv2.namedWindow("imageWithInscribingSquare", cv2.WINDOW_NORMAL)
			cv2.imshow('imageWithInscribingSquare',imageWithInscribingSquare)



			#cv2.rectangle(image_original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	else:
		print("Circles Not Found")

	cv2.imshow('detected circles',image_original)
	cv2.imwrite('detected circles.jpg',image_original)
	cv2.waitKey(0)


size = "7L"
test_path = r"top_inset/" + size



test_imgs = [(cv2.imread(os.path.join(test_path,f)),f) for f in os.listdir(test_path)]
for ref_img,f  in test_imgs:
	try:
		# print(f)
		img = crop_image(ref_img, y=435, height=650, x=200, width= 610)
		parlak_img = adjust_brightness(img, 1)
		contrast = apply_brightness_contrast(parlak_img, brightness = 0, contrast = 10)
		hough = hough_circle(img, size)
	except:
		pass
