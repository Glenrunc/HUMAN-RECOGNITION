import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from skimage.filters import gabor_kernel
from skimage.util import crop
from scipy.signal import convolve2d
from scipy.spatial.distance import hamming
from typing import List, Tuple

def remove_glare(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    H = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(H[150:])
    # plt.show()
    idx = np.argmax(H[150:]) + 151
    binary = cv2.threshold(image, idx, 255, cv2.THRESH_BINARY)[1]

    st3 = np.ones((3, 3), dtype="uint8")
    st7 = np.ones((7, 7), dtype="uint8")

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, st3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, st3, iterations=2)

    im_floodfill = binary.copy()

    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, st7, iterations=1)
    _, _, stats, cents = cv2.connectedComponentsWithStats(im_out)
    cx, cy = 0, 0
    for st, cent in zip(stats, cents):
        if 1500 < st[4] < 3000:
            if 0.9 < st[2] / st[3] < 1.1:
                cx, cy = cent.astype(int)
                r = st[2] // 2
                cv2.circle(image, (cx, cy), r, (125, 125, 125), thickness=2)

    image = np.where(im_out, 64, image)
    image = cv2.medianBlur(image, 5)

    return image, cx, cy

def find_pupil_iris(img, center_estimate, step=2, step_radius=10, initial_radius=30, max_radius=200):
    # Initialize variables
    best_seed = None
    best_radius = None
    best_brightness_change = []
    
    # Loop over seed points
    for x_offset in range(-step, step+1, step):
        for y_offset in range(-step, step+1, step):
            seed = (center_estimate[0] + x_offset, center_estimate[1] + y_offset)
            
            # Loop over radius with adaptive step size
            brightness_changes = []
            for r in range(initial_radius, max_radius+1, step_radius):
                if r < 50:
                    angle_step = 1
                else:
                    angle_step = 5
                
                # Loop over angles
                points = []
                brightness_sum = 0
                for angle in range(0, 360, angle_step):
                    x = int(seed[0] + r*np.cos(angle*np.pi/180))
                    y = int(seed[1] + r*np.sin(angle*np.pi/180))
                    points.append((x, y))
                    brightness_sum += img[y,x]

                # print('SUM',brightness_sum)
                mean_brightness = brightness_sum / len(points)
                # print(mean_brightness)
                brightness_changes.append((r, mean_brightness))
            
            # Select the radius with largest brightness change
            largest_gap = 0
            k = 0
            # print(brightness_changes[0][1])
            for i in range(len(brightness_changes) - 1):
                gap = brightness_changes[i+1][1] - brightness_changes[i][1]
                if gap > largest_gap:
                    largest_gap = gap
                    k = i
                    
            best_brightness_change.append((seed, brightness_changes[k]))
            
    # Select the seed with the highest brightness change
    best_feature = max(best_brightness_change, key=lambda x: x[1][1])
    
    # Draw circle on image and return center and radius
    # img_with_circle = cv2.circle(img.copy(), best_feature[0], best_feature[1][0], (255, 0, 0), 1)
   
    best_seed = best_feature[0]
    best_radius = best_feature[1][0]

    return (best_seed,best_radius)

def gabor_filter(img,feature_iris,feature_pupil):
    
  # Create the Gabor filter bank
    kernels = []
    sigma = 2
    for theta in range(8):
        t = theta / 8. * np.pi
        kernel = gabor_kernel(0.15, theta=t, sigma_x=sigma, sigma_y=sigma)
        kernels.append(kernel)

    # Determine the sample locations using the polar coordinate system
    rz = feature_pupil[1]
    rt = feature_iris[1]
    pupil_center_x = feature_pupil[0][0]
    pupil_center_y = feature_pupil[0][1]
    alpha = 0.5
    radius_step = (rt - rz) / 8
    angle_step = np.pi / 8
    locations = []
    for r in range(1,9):
        radius = rz + r * radius_step
        for i, angle in enumerate(np.arange(-np.pi / 4, np.pi / 4, angle_step)):
            x1 = pupil_center_x + radius * np.cos(angle + alpha)
            y1 = pupil_center_y + radius * np.sin(angle + alpha)
            locations.append((int(x1), int(y1)))
        for i, angle in enumerate(np.arange(-np.pi*3 / 4, np.pi*3 / 4, angle_step)):
            x1 = pupil_center_x + radius * np.cos(angle + alpha)
            y1 = pupil_center_y + radius * np.sin(angle + alpha)
            locations.append((int(x1), int(y1)))
    # Extract features at each sample location using the Gabor filter bank
    features = []
    patch_size = 21
    for location in locations:
        x, y = location
        patch = img[y - patch_size // 2: y + patch_size // 2 + 1, x - patch_size // 2: x + patch_size // 2 + 1]
        patch = cv2.resize(patch, (64, 64))
        code = ''
        for kernel in kernels:
            filtered_real = cv2.filter2D(patch, cv2.CV_32F, kernel.real)
            filtered_imag = cv2.filter2D(patch, cv2.CV_32F, kernel.imag)
            if filtered_real.mean() > 0:
                code += '1'
            else:
                code += '0'
            if filtered_imag.mean() > 0:
                code += '1'
            else:
                code += '0'
        features.append(code)

    # Concatenate the feature codes to obtain a 2048-length binary code
    binary_code = ''.join(features)
    
    return binary_code

def compare_binary_code(binary_code_1,binary_code_2):
    
    hamming_value = hamming(binary_code_1,binary_code_2)
    percent_of_proof = (1-hamming_value)*100

    return percent_of_proof

def main(data_path: str) -> None:
    # Get files from data path
    filename_list = [
        f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
    ]
    binary_code_dict = {}
    hamming_train_dataset = {}
    for filename in filename_list:
        # Read image
        img = cv2.imread(os.path.join(data_path, filename))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        feature_pupil = find_pupil_iris(img_no_glare,(x,y), step=2, step_radius=10, initial_radius=30, max_radius=200)
        feature_iris = find_pupil_iris(img_no_glare,(x,y), step=2, step_radius=10, initial_radius=150, max_radius=269)
        
        # img_circles = cv2.circle(img.copy(),feature_pupil[0], feature_pupil[1], (255, 0, 0), 1)
        # cv2.circle(img_circles,feature_iris[0], feature_iris[1], (255, 0, 0), 1)
        
        
        # Gabor filters
        
        img_binary = gabor_filter(img_no_glare,feature_iris,feature_pupil)
        binary_code_dict.setdefault(filename.split('_')[0][4], []).append(img_binary)


        #TODO
        # cv2.imshow('final test',img_binary)
        # key = cv2.waitKey()
        # if key == ord("x"):
        #     break
        # cv2.imshow("Original image", img)
        # cv2.imshow("Gray", gray)
        # cv2.imshow("No glare", img_no_glare)
        

    for eye,binary_code in binary_code_dict.items():
        hamming_train_dataset[eye] = compare_binary_code(list(binary_code[0]),list(binary_code[1]))
        
        #HERE you can find the result for the train_dataset, it works.
        # result for train dataset {letter of the eye , percent of accuracy} --->  {'A': 86.181640625, 'B': 89.892578125, 'C': 89.84375, 'D': 90.72265625, 'E': 90.72265625}
      
    print('result for train dataset {letter of the eye , percent of accuracy} ---> ',hamming_train_dataset)

    #Let test our model 
    data_test_path = './iris_database_test'
    filename_test_list = [
        f for f in os.listdir(data_test_path) if os.path.isfile(os.path.join(data_test_path, f))
    ]
    binary_code_test_dict = {}

    for filename_test in filename_test_list : 
        # Read image
        img = cv2.imread(os.path.join(data_test_path, filename_test))

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove glare
        img_no_glare, x, y = remove_glare(gray)

        # Exploding circle algorithm
        feature_pupil = find_pupil_iris(img_no_glare,(x,y), step=2, step_radius=10, initial_radius=30, max_radius=200)
        feature_iris = find_pupil_iris(img_no_glare,(x,y), step=2, step_radius=10, initial_radius=150, max_radius=269)

        img_binary_test = gabor_filter(img_no_glare,feature_iris,feature_pupil)
        binary_code_test_dict[filename_test] = img_binary_test

    #Now we have two dictionnary with test and train let see if that's work
    result = {}
    
    for filename , binary_code  in  binary_code_test_dict.items():
        percent = 0 
        letter = ''
        for eye,binary_code_eye in binary_code_dict.items():
            for i in range(2):
                ine =compare_binary_code(list(binary_code_eye[i]),list(binary_code))
                if ine > percent : 
                    percent = ine
                    letter = eye
    
        result[filename + ' Result  --> ' + letter] = percent 
        #You can see that the last image doesn't work at all because it isn't in the dataset
        # But the eye which fits better is E    
        # {'irisA_3.png Result  --> A': 92.431640625, 'irisB_3.png Result  --> B': 92.1875, 'irisF.png Result  --> E': 78.955078125}
    print(result)
if __name__ == "__main__":
    data_path = "./iris_database_train"
    main(data_path)
    