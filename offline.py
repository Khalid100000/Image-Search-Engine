from PIL import Image
from feature_extractor import CbirExtractor,SiftExtractor
from pathlib import Path
import numpy as np
import cv2

if __name__ == '__main__':
    print('starting offline.py')
    print('Starting CBIR Extraction')
    bins=(8, 12, 3)
    CE = CbirExtractor(bins)
    #print('done makeing the feature extraction object')
    #print(sorted(Path("./static/img").glob("*.jpg")))
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(str(img_path),type(str(img_path)))  # e.g., ./static/img/xxx.jpg
        feature = CE.extract(img=cv2.imread(str(img_path)))
        print(feature,type(feature))
        feature_path = Path("./static/CbirFeature") / (img_path.stem + ".npy")  # e.g., ./static/CbirFeature/xxx.npy
        np.save(feature_path, feature)
    print('Done CBIR Extraction')
    print('--------------------*******************------------------------')
    print('Now starting Sift Extraction')
    SE = SiftExtractor()
    #print(sorted(Path("./static/img").glob("*.jpg")))
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(str(img_path),type(str(img_path)))  # e.g., ./static/img/xxx.jpg
        feature = SE.extract(img=cv2.imread(str(img_path), 0))
        feature_path = Path("./static/SiftFeature") / (img_path.stem + ".npy")  # e.g., ./static/SiftFeature/xxx.npy
        np.save(feature_path, feature)
    print('Done Making Sift Extraction')