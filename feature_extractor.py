import cv2
import numpy as np
import imutils



class CbirExtractor:
	def __init__(self,bins):
		self.bins=bins # number of bins 
	def extract(self, img):
		"""
		Extract a deep feature from an input image
		Args:
			img: a BGR from imread
			Returns:
			feature (np.ndarray): deep feature with the shape=(#p,128)
		"""
		image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert it to HSV color space
		features = []
		# now getting the dimensions of the image and its center:
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
		#For our image descriptor, we are going to divide our image into five different regions: 
		#(1) the top-left corner, (2) the top-right corner, (3) the bottom-right corner,
		#(4) the bottom-left corner, and finally (5) the center of the image.
		#first get the top-lift,top-right,bottom-lift,bottom-right
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]
		# then get the center by applying an elliptical mask
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		#-----------------------
		# now we will be getting the histogram for the 5 regions 
		# first get gor the 4 regions on the corners:-
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# extract a color histogram from the image, then update the
			# feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)
		# now for the center region
		hist = self.histogram(image, ellipMask)
		features.extend(hist)	
		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
		# normalize the histogram if we are using OpenCV 2.4

		hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist


	def chi2_distance(self,A, B):
		eps = 1e-10 # add it to prevent dividing by zero
 		# compute the chi-squared distance from https://www.geeksforgeeks.org/chi-square-distance-in-python/
		chi = 0.5 * np.sum([((a - b) ** 2) / (a + b+eps) for (a, b) in zip(A, B)])
		return chi


class SiftExtractor:
	def __init__(self):
		self.model=cv2.xfeatures2d.SIFT_create()
		#self.FLANN_INDEX_KDTREE=1
		self.index_params=dict(algorithm=1,trees=5)
		self.search_params=dict(checks=50)   # or pass empty dictionary
		self.flann=cv2.FlannBasedMatcher(self.index_params,self.search_params)
	def extract(self, img):
		"""
		Extract a deep feature from an input image
		Args:
			img: a gray scale image from cv2.imread
			Returns:
			feature (np.ndarray): deep feature with the shape=(#p,128)
		"""
		_,feature = self.model.detectAndCompute(img,None)  # (1, 4096) -> (4096, )
		return feature
	def matching(self,des1,des2):
		matches=self.flann.knnMatch(des1,des2,k=2)
		N_matches=sum([1 for m,n in matches if m.distance < 0.5*n.distance])
		if N_matches == 0:
			return 1000
		else:
   			return 1.0/N_matches*1000      	
