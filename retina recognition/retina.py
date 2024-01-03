import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

data_dir_path = './RIDB'

def main(data_path: str) -> None:
    retina_final_image = []
   
    # Get files from data path
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]

    for filename in tqdm(filename_list):
        img = cv2.imread(os.path.join(data_path, filename))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_floodfill = gray.copy()
        h, w = img_floodfill.shape[:2]
        floodfill_mask = np.zeros((h+2, w+2), np.uint8)
        # Perform floodfill from (0, 0) to remove black border
        cv2.floodFill(img_floodfill, floodfill_mask, (0,0), 255)

        dil_kernel = np.ones((11, 11),np.uint8)
        img_floodfill = cv2.dilate(img_floodfill,dil_kernel)

        img_floodfill = cv2.bitwise_not(img_floodfill)
        mask = cv2.threshold(img_floodfill, 1, 255, cv2.THRESH_BINARY)[1]
        
        
        # enhance contrast with CLAHE algorithm
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # invert enhanced grayscale image and apply CLAHE again
        gray = 255 - gray
        gray = clahe.apply(gray)

        # blur image with Gaussian kernel
        k_size = int((13 - 1) / 2)
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
        img_blur = cv2.GaussianBlur(gray, (13, 13), sigma)

        # adaptive thresholding with gaussian-weighted sum of the neighbourhood values
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0
        )

        # combine threshold output with binary mask
        img_thresh = cv2.bitwise_and(img_thresh, mask)

        # blur image with median filter and invert values
        img_thresh = cv2.medianBlur(img_thresh, 5)
        img_thresh = cv2.bitwise_not(img_thresh)

        # fill the holes with morphological close operation
        closing_kernel = np.ones((4,4),np.uint8)
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, closing_kernel)

        img_thresh = cv2.bitwise_not(img_thresh)
         # get objects stats and remove background object with Connected Component Analysis
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            img_thresh, connectivity=8
        )
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        
        for i in range(0, nb_components):
            if sizes[i] < 500:
                img_thresh[output == i + 1] = 0
        retina_final_image.append(img_thresh) 
        # get skeleton of preserved objects
        skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros(img_thresh.shape, np.uint8)
        
        while True:
            opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, skel_element)
            temp = cv2.subtract(img_thresh, opened)
            eroded = cv2.erode(img_thresh, skel_element)
            skel = cv2.bitwise_or(skel, temp)
            img_thresh = eroded.copy()
            if cv2.countNonZero(img_thresh)==0:
                break
                
        retina_final_image.append(skel)            

    
    return retina_final_image

if __name__ == "__main__":
    image_retina_skel = main(data_dir_path)
    
    for img in image_retina_skel:
        cv2.imshow('final test', img)
        key = cv2.waitKey()
        if key == ord("x"):
            break
        