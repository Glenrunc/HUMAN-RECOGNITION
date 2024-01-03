import cv2
import csv
import os
import argparse
import tqdm
import random
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from facenet_pytorch import MTCNN,InceptionResnetV1
import torch

from typing import Dict, List, Tuple

N_FACES_THRESHOLD = 20

IMG_HEIGHT = 160
IMG_WIDTH = 160

X_TOP_LEFT_IDX = 2
Y_TOP_LEFT_IDX = 3
X_BOT_RIGHT_IDX = 6
Y_BOT_RIGHT_IDX = 7

TRAIN_FRAC = 0.75

# FRs = {
#     "Eigen": cv2.face.EigenFaceRecognizer_create(),
#     "Fisher": cv2.face.FisherFaceRecognizer_create(),
#     "LBPH": cv2.face.LBPHFaceRecognizer_create(),
# }
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    # img_GRAY = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2GRAY)
    img_RGB = img_RGB[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])] 
    out_img = cv2.resize(img_RGB,(IMG_WIDTH,IMG_HEIGHT))
    out_img = out_img/255
    out_img_tensor = torch.from_numpy(out_img).permute(2,0,1).float()
    
    return out_img_tensor


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
    train_img = []
    train_lbl: List[int] = []
    test_img = []
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
    
    train_lbl = np.array(train_lbl)
    # print(train_lbl)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
 
    aligned_train_img = torch.stack(train_img).to('cpu')
    aligned_test_img = torch.stack(test_img).to('cpu')
    
    embeddings_train = resnet(aligned_train_img).detach().cpu().numpy()
    embeddings_test = resnet(aligned_test_img).detach().cpu().numpy()

# create SVM classifier with specified parameters
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm.train(embeddings_train, cv2.ml.ROW_SAMPLE, train_lbl)

label_pred = svm.predict(embeddings_test)
label_pred = label_pred[1]

correct_pred = sum([1 for i in range(len(test_lbl)) if label_pred[i] == test_lbl[i]])

print("{} accuracy = {:.2f}".format('SVM', correct_pred / float(len(test_lbl))))


#We cannot draw conclusion because of the size of the dataset. However the time to compute SVM method takes more time than EigerFace method or FisherFace method for example


