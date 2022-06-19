from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import cv2

if __name__ == '__main__':
    print('starting offline.py')
    fe = FeatureExtractor()
    print('done makeing the feature extraction object')
    #print(sorted(Path("./static/img").glob("*.jpg")))
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(str(img_path),type(str(img_path)))  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=cv2.imread(str(img_path), 0))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
