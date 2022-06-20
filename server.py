import numpy as np
from PIL import Image
from feature_extractor import SiftExtractor,CbirExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import cv2

app = Flask(__name__)

# Read image features
bins=(8, 12, 3)
CE = CbirExtractor(bins)
SE=SiftExtractor()
CE_features = [] # all the CBIR features of the images
SE_features=[]
CE_img_paths = []
SE_img_paths=[]
#looping to get cbir features
for feature_path in Path("./static/CbirFeature").glob("*.npy"):
	CE_features.append(np.load(feature_path))
	CE_img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
#features = np.array(features)
# looping to get Sift features
for feature_path in Path("./static/SiftFeature").glob("*.npy"):
	SE_features.append(np.load(feature_path))
	SE_img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))

# this is for CBIR retrieval
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
        queryImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Run search
        query = CE.extract(queryImage)
        #print('length of query',len(query))
        #print('length of all features',len(features))
        #print('length of one feature',len(features[1]))
        #print('type of features',type(features))
        #print('first_feature',type(features[0]),features[0])
        #print('query feature list',query)
        #print('------------------')
        #print('one feature list',features[1])
        #print('------------------')
        dists=[CE.chi2_distance(list(x),query) for x in CE_features ] # getting the distances between the query image and the other images 
        #dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], CE_img_paths[id]) for id in ids]

        return render_template('index.html',query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('index.html')
# this route for the Sift image retrieval
@app.route('/Sift', methods=['GET', 'POST'])
def indexSift():
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
        query = SE.extract(img_cv2)
        dists=[SE.matching(x,query) for x in SE_features ]
        #dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], SE_img_paths[id]) for id in ids]

        return render_template('indexSift.html',query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('indexSift.html')

if __name__=="__main__":
	app.run()
