import cv2
import csv
import os
import argparse
import random
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from typing import Dict, List, Tuple

N_FACES_THRESHOLD = 20

IMG_HEIGHT = 100
IMG_WIDTH = 70

X_TOP_LEFT_IDX = 2
Y_TOP_LEFT_IDX = 3
X_BOT_RIGHT_IDX = 6
Y_BOT_RIGHT_IDX = 7

TRAIN_FRAC = 0.75

FRs = {
    "Eigen": cv2.face.EigenFaceRecognizer_create(),
    "Fisher": cv2.face.FisherFaceRecognizer_create(),
    "LBPH": cv2.face.LBPHFaceRecognizer_create(),
}


def read_labels(csv_path: str) -> Dict[int, int]:
    """
    Read labels from Caltech database .csv file and resturn as dict {id: label}
    """
    mydict: Dict[int, int] = {}

    with open(csv_path,'r') as csvfile:
        reader = csv.reader(csvfile)

        for k,row in enumerate(reader):
            mydict[k+1] = int(row[0])

    return mydict


def read_ROIs(path: str) -> np.ndarray:
    """
    Read ROI's array
    """
    rois: np.ndarray = None
    mydict = loadmat(path)
    rois = mydict['SubDir_Data']
    return rois


def read_img_paths(dir: str) -> List[str]:
    """
    Read image file names from given directory and return as a list of str: "dir/name.jpg"
    """
    filename_list: List[str] = []

    for filename in os.listdir(dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dir,filename)
            if os.path.isfile(img_path):
                filename_list.append(img_path)
            
    filename_list = [filename.replace('\\', '/') for filename in filename_list]

    return filename_list


def img_preprocessing(path: str, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess images to use them as inputs to FRs
    """
    out_img: np.ndarray = None
    
    img_RGB = cv2.imread(path)
    img_GRAY = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2GRAY)
    
    out_img = img_GRAY[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])] 
    # cv2.imshow('Crop_image',out_img)
    # cv2.waitKey(0) 
    out_img = cv2.resize(out_img,(IMG_WIDTH,IMG_HEIGHT))
    
    # cv2.imshow('Crop_image',out_img)
    # cv2.waitKey(0)
    return out_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    random.seed(11)

    # Read labels
    csv_path = os.path.join(args.path, "caltech_labels.csv")
    labels = read_labels(csv_path)

    # Read ROIs
    mat_path = os.path.join(args.path, "ImageData.mat")
    rois = read_ROIs(mat_path)

    # Read filenames
    filenames = read_img_paths(args.path)

    # Create train and test data
    train_img: List[np.ndarray] = []
    train_lbl: List[int] = []
    test_img: List[np.ndarray] = []
    test_lbl: List[int] = []
    for img_path in filenames:
        
        img_idx = int(img_path.split('_')[1].split('.')[0])
        img_label = labels[img_idx]
        roi = rois[:,img_idx-1]
        top_left = (roi[X_TOP_LEFT_IDX],roi[Y_TOP_LEFT_IDX])
        bottom_right = (roi[X_BOT_RIGHT_IDX],roi[Y_BOT_RIGHT_IDX])
        
        n_faces = sum([1 for label in labels.values() if label == img_label])
        
        if n_faces >= N_FACES_THRESHOLD:
            img_input = img_preprocessing(img_path,top_left,bottom_right)  
            # print(random.random())
            if random.random() <= TRAIN_FRAC:
                train_img.append(img_input)
                # cv2.imshow('Crop_image',img_input)
                # cv2.waitKey(0)
                train_lbl.append(img_label)
                # print(img_label)
            else:
                test_img.append(img_input)
                # cv2.imshow('Crop_image',img_input)
                # cv2.waitKey(0)
                test_lbl.append(img_label)
                # print(img_label)
    
    # Check all methods
    for method_name, method in FRs.items():
        print(method_name,'is currently working ... ')
        method.train(train_img,np.array(train_lbl))
        correct_n = 0
       
        for i in tqdm(range(len(test_lbl))):
            prediction = method.predict(test_img[i])
            if prediction[0] == test_lbl[i] :
                correct_n += 1
        
        print("{} accuracy = {:.2f}".format(method_name, correct_n / float(len(test_lbl))))
