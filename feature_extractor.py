import cv2
import numpy as np


class FeatureExtractor:
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
		N_matches=sum([1 for m,n in matches if m.distance < 0.7*n.distance])
		if N_matches == 0:
			return 1000
		else:
   			return 1.0/N_matches*1000
        	
