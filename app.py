import numpy as np
from PIL import Image
#from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import cv2
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
app = Flask(__name__)
# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
	features.append(np.load(feature_path))
	img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
#features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&********************^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(str(file.name),type(str(file.name)))
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv2=cv2.cvtColor(opencvImage,cv2.COLOR_BGR2GRAY)

        # Run search
        query = fe.extract(img_cv2)
        dists=[fe.matching(x,query) for x in features ]
        #dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('index.html')

