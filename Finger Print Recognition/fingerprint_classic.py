import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import List, Tuple
import matplotlib.pyplot as plt

def read_img_paths(dir: str) -> List[str]:
    """
    Read image file names from given directory and return as a list of str: "dir/name.tif"
    """
    filename_list: List[str] = []

    for filename in os.listdir(dir):
        if filename.endswith('.tif'):
            img_path = os.path.join(dir,filename)
            if os.path.isfile(img_path):
                filename_list.append(img_path)
            
    filename_list = [filename.replace('\\', '/') for filename in filename_list]

    return filename_list


def img_preprocessing(path: str) -> np.ndarray:
    """
    Read image from file and convert to grayscale
    """
    gray : np.ndarray = None

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return gray


def get_features(gray: np.ndarray, alg_type: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute keypoints locations and their descriptors
    """
    assert alg_type == "sift" or alg_type == "orb"
    kp: List[np.ndarray] = []
    dsc: List[np.ndarray] = []
    
    if alg_type == 'sift':
        detector = cv2.SIFT_create()
    else:
        detector= cv2.ORB_create()
        
    kp = detector.detect(gray,None)
    kp, dsc = detector.compute(gray,kp)
    
    return kp, dsc


def match_fingerprints(
    features2match: np.ndarray, features_base: np.ndarray, alg_type: str, m_coeff=0.75
) -> Tuple[List[str], List[str]]:
    """
    Match features and gather stats
    """
    assert alg_type == "sift" or alg_type == "orb"
    y_pred: List[str] = list()
    y_test: List[str] = list()

    if alg_type == 'sift':
        matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
    
    for img_name in features2match.keys():
        for i in range(len(features2match[img_name][0 if alg_type=="sift" else 1][alg_type])):
            matches = matcher.knnMatch(features2match[img_name][0 if alg_type=="sift" else 1][alg_type][i],features_base[img_name][0 if alg_type=="sift" else 1][alg_type][0 if alg_type=="sift" else 1], k=2)
            good = []
            for m,n in matches:
                if m.distance < m_coeff*n.distance:
                    good.append([m])
            if len(good) > 10:
                y_pred.append(img_name)
                y_test.append(img_name)
            else:
                y_pred.append("none")
                y_test.append(img_name)
    
    return y_pred, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    # Read filepaths
    filepaths = read_img_paths(args.path)

    # Get features
    features_base = dict()
    features2match = dict()
    alg =(dict(),dict())
    

    for path in tqdm(filepaths):
        
        img_name = path.split('/')[2].split('_')[0]
        img_idx = int(path.split('/')[2].split('_')[1].split('.')[0])
        gray = img_preprocessing(path)
        kp_sift, dsc_sift = get_features(gray, "sift")
        kp_orb, dsc_orb = get_features(gray, "orb")

        if img_idx == 1 :
            features_base[img_name]= alg
            alg[0]["sift"] = dsc_sift
            alg[1]["orb"] = dsc_orb
        else:
            if img_name in features2match : 
                features2match[img_name][0]["sift"].append(dsc_sift)
                features2match[img_name][1]["orb"].append(dsc_orb)
            else:
                features2match[img_name]= alg
                alg[0]["sift"] = list()
                alg[1]["orb"] = list()
                alg[0]["sift"].append(dsc_sift)
                alg[1]["orb"].append(dsc_orb)           

       
    #Match features
    preds, gt = match_fingerprints(features2match, features_base, "sift")
    print("--- SIFT ---")
    print(classification_report(gt, preds,zero_division=0))

    preds, gt = match_fingerprints(features2match, features_base, "orb")
    print("--- ORB ---")
    print(classification_report(gt, preds,zero_division=0))





""" 
    test
img_gray = img_preprocessing('./DB1_B/101_1.tif')
feature = get_features(img_gray ,'sift')
img2 = cv2.drawKeypoints(img_gray,feature[0] , None, color=(0,255,0), flags=0)
plt.imshow(img2)
plt.show()

"""