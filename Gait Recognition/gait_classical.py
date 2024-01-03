from configparser import Interpolation
from msilib import sequence
from multiprocessing import process
import cv2
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

def img_processing(images_path):
    process_img = []
    for path in images_path:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binarize = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binarize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the largest area (assuming it represents the person)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the person region using the bounding box coordinates
            process_img.append(gray[y:y+h, x:x+w].astype(float))

    max_h = process_img[0].shape[0]
    max_w = process_img[0].shape[1]

    for img in process_img[1:]:
        max_h = max(max_h, img.shape[0])
        max_w = max(max_w, img.shape[1])
            

    new_shape = (max_h, max_w)

    for i, img in enumerate(process_img):
        add_ligne_haut = (new_shape[0] - img.shape[0]) // 2
        add_ligne_bas = new_shape[0] - img.shape[0] - add_ligne_haut
        add_col_gauche = (new_shape[1] - img.shape[1]) // 2
        add_col_droit = new_shape[1] - img.shape[1] - add_col_gauche
    
        process_img[i] = np.pad(img, ((add_ligne_haut, add_ligne_bas), (add_col_gauche, add_col_droit)), mode='constant')

    return process_img
  
   
def compute_gei(images_path):

    sequence_person = img_processing(images_path)
    batch_sequence = np.zeros((len(sequence_person),sequence_person[0].shape[0],sequence_person[0].shape[1]))

    for i,array in enumerate(sequence_person):
        batch_sequence[i] = array

    gei = np.mean(batch_sequence,axis=0)
    
    gei_flat = gei.flatten()

    return gei_flat


if __name__ == "__main__":
    
    angle_of_view = "108"
    walking_training_statut = ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01','nm-01','nm-05','cl-02','bg-02']
    walking_validation_statut = ['nm-02']
    list_person = os.listdir("./GaitDatasetB-silh/")
    
    training_person_gei = {}
    validation_person_gei = {}
    path_sequence = "./GaitDatasetB-silh/"+list_person[0]+"/"+walking_validation_statut[0]+"/"+ angle_of_view+"/"
    path_img = os.listdir(path_sequence)
    path_all = [path_sequence+s for s in path_img ]

    for nb_pr in tqdm(list_person):
        for walking_st in walking_training_statut:
            path_sequence = "./GaitDatasetB-silh/"+nb_pr+"/"+walking_st+"/"+ angle_of_view+"/"
            path_img = os.listdir(path_sequence)
            path_all = [path_sequence+s for s in path_img ]
            gei = compute_gei(path_all)
            training_person_gei[nb_pr + ","+walking_st] = gei

        for walking_val in walking_validation_statut:
            path_sequence = "./GaitDatasetB-silh/"+nb_pr+"/"+walking_val+"/"+ angle_of_view+"/"
            path_img = os.listdir(path_sequence)
            path_all = [path_sequence+s for s in path_img ]
            gei = compute_gei(path_all)
            validation_person_gei[nb_pr + ","+walking_val] = gei
    
    classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2023, verbose=1, max_depth=100, max_features=100)
    #TRAINING PART OF THE CLASSIFIER
    features_training = list(training_person_gei.values())
    labels_training = list(training_person_gei.keys())    
    labels_real_list_tr = []

    max_length = max(len(seq) for seq in features_training)
    #We put the GEI at the same shape for all 
    padded_features_tr = pad_sequences(features_training, maxlen=max_length, padding='post', truncating='post')

    #Here I keep only the ID i.e 001/002,etc
    for label in labels_training:
        labels_real_list_tr.append(label.split(',')[0])

    classifier.fit(padded_features_tr, labels_real_list_tr)
    
    #VALIDATION PART     
    
    features_validation = list(validation_person_gei.values())
    labels_validation = list(validation_person_gei.keys())
    labels_real_list_vl = []

    
    #We put the GEI at the same shape for all 
    padded_features_vl = pad_sequences(features_validation, maxlen=max_length, padding='post', truncating='post')

    #Here I keep only the ID i.e 001/002,etc
    for label in labels_validation:
        labels_real_list_vl.append(label.split(',')[0])
   
    predictions = classifier.predict(padded_features_vl)

    # Print the predictions
    correct_prediction = 0 
    for idx, prediction in enumerate(predictions):
        person_gait = labels_real_list_vl[idx]
        if person_gait == prediction:
            correct_prediction = correct_prediction + 1

        print(f"Prediction for {person_gait}: {prediction}")
        
    print("CORRECT PREDICTION:",correct_prediction)
    #Accuracy
    accuracy = correct_prediction * 100 /124
    print("ACCURACY OF THE MODEL : ", accuracy,"%" )


    
    #RESULT FOR TRAIN ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01']

# Prediction for 001: 050
# Prediction for 002: 002
# Prediction for 003: 003
# Prediction for 004: 014
# Prediction for 005: 064
# Prediction for 006: 006
# Prediction for 007: 097
# Prediction for 008: 008
# Prediction for 009: 009
# Prediction for 010: 010
# Prediction for 011: 117
# Prediction for 012: 118
# Prediction for 013: 013
# Prediction for 014: 014
# Prediction for 015: 015
# Prediction for 016: 102
# Prediction for 017: 002
# Prediction for 018: 098
# Prediction for 019: 017
# Prediction for 020: 020
# Prediction for 021: 021
# Prediction for 022: 031
# Prediction for 023: 023
# Prediction for 024: 011
# Prediction for 025: 099
# Prediction for 026: 026
# Prediction for 027: 027
# Prediction for 028: 072
# Prediction for 029: 113
# Prediction for 030: 026
# Prediction for 031: 031
# Prediction for 032: 032
# Prediction for 033: 033
# Prediction for 034: 102
# Prediction for 035: 035
# Prediction for 036: 058
# Prediction for 037: 111
# Prediction for 038: 097
# Prediction for 039: 080
# Prediction for 040: 040
# Prediction for 041: 121
# Prediction for 042: 015
# Prediction for 043: 043
# Prediction for 044: 102
# Prediction for 045: 016
# Prediction for 046: 046
# Prediction for 047: 060
# Prediction for 048: 065
# Prediction for 049: 025
# Prediction for 050: 020
# Prediction for 051: 112
# Prediction for 052: 121
# Prediction for 053: 067
# Prediction for 054: 048
# Prediction for 055: 048
# Prediction for 056: 107
# Prediction for 057: 057
# Prediction for 058: 058
# Prediction for 059: 039
# Prediction for 060: 060
# Prediction for 061: 014
# Prediction for 062: 115
# Prediction for 063: 063
# Prediction for 064: 064
# Prediction for 065: 065
# Prediction for 066: 105
# Prediction for 067: 067
# Prediction for 068: 068
# Prediction for 069: 069
# Prediction for 070: 070
# Prediction for 071: 081
# Prediction for 072: 097
# Prediction for 073: 038
# Prediction for 074: 050
# Prediction for 075: 094
# Prediction for 076: 076
# Prediction for 077: 077
# Prediction for 078: 057
# Prediction for 079: 079
# Prediction for 080: 069
# Prediction for 081: 027
# Prediction for 082: 118
# Prediction for 083: 072
# Prediction for 084: 084
# Prediction for 085: 091
# Prediction for 086: 025
# Prediction for 087: 087
# Prediction for 088: 046
# Prediction for 089: 105
# Prediction for 090: 050
# Prediction for 091: 105
# Prediction for 092: 092
# Prediction for 093: 041
# Prediction for 094: 094
# Prediction for 095: 018
# Prediction for 096: 096
# Prediction for 097: 097
# Prediction for 098: 105
# Prediction for 099: 113
# Prediction for 100: 038
# Prediction for 101: 065
# Prediction for 102: 102
# Prediction for 103: 103
# Prediction for 104: 104
# Prediction for 105: 105
# Prediction for 106: 106
# Prediction for 107: 009
# Prediction for 108: 021
# Prediction for 109: 041
# Prediction for 110: 110
# Prediction for 111: 111
# Prediction for 112: 083
# Prediction for 113: 113
# Prediction for 114: 114
# Prediction for 115: 080
# Prediction for 116: 081
# Prediction for 117: 086
# Prediction for 118: 034
# Prediction for 119: 057
# Prediction for 120: 120
# Prediction for 121: 121
# Prediction for 122: 122
# Prediction for 123: 124
# Prediction for 124: 124
# CORRECT PREDICTION: 53
# ACCURACY OF THE MODEL :  42.74193548387097 %


#WITH TRAIN ['nm-03', 'bg-01', 'nm-06', 'nm-04', 'cl-01','nm-01','nm-05','cl-02','bg-02']

# Prediction for 001: 050
# Prediction for 002: 002
# Prediction for 003: 003
# Prediction for 004: 004
# Prediction for 005: 005
# Prediction for 006: 006
# Prediction for 007: 097
# Prediction for 008: 008
# Prediction for 009: 009
# Prediction for 010: 010
# Prediction for 011: 117
# Prediction for 012: 047
# Prediction for 013: 013
# Prediction for 014: 014
# Prediction for 015: 015
# Prediction for 016: 016
# Prediction for 017: 063
# Prediction for 018: 018
# Prediction for 019: 017
# Prediction for 020: 020
# Prediction for 021: 021
# Prediction for 022: 022
# Prediction for 023: 023
# Prediction for 024: 030
# Prediction for 025: 025
# Prediction for 026: 026
# Prediction for 027: 068
# Prediction for 028: 028
# Prediction for 029: 029
# Prediction for 030: 030
# Prediction for 031: 031
# Prediction for 032: 032
# Prediction for 033: 033
# Prediction for 034: 034
# Prediction for 035: 035
# Prediction for 036: 058
# Prediction for 037: 111
# Prediction for 038: 043
# Prediction for 039: 039
# Prediction for 040: 040
# Prediction for 041: 121
# Prediction for 042: 015
# Prediction for 043: 043
# Prediction for 044: 102
# Prediction for 045: 016
# Prediction for 046: 046
# Prediction for 047: 110
# Prediction for 048: 065
# Prediction for 049: 049
# Prediction for 050: 020
# Prediction for 051: 051
# Prediction for 052: 121
# Prediction for 053: 053
# Prediction for 054: 048
# Prediction for 055: 055
# Prediction for 056: 063
# Prediction for 057: 057
# Prediction for 058: 058
# Prediction for 059: 059
# Prediction for 060: 060
# Prediction for 061: 014
# Prediction for 062: 080
# Prediction for 063: 063
# Prediction for 064: 064
# Prediction for 065: 065
# Prediction for 066: 105
# Prediction for 067: 067
# Prediction for 068: 068
# Prediction for 069: 069
# Prediction for 070: 070
# Prediction for 071: 032
# Prediction for 072: 097
# Prediction for 073: 007
# Prediction for 074: 050
# Prediction for 075: 075
# Prediction for 076: 076
# Prediction for 077: 077
# Prediction for 078: 078
# Prediction for 079: 079
# Prediction for 080: 069
# Prediction for 081: 081
# Prediction for 082: 082
# Prediction for 083: 051
# Prediction for 084: 084
# Prediction for 085: 085
# Prediction for 086: 023
# Prediction for 087: 087
# Prediction for 088: 088
# Prediction for 089: 095
# Prediction for 090: 050
# Prediction for 091: 105
# Prediction for 092: 092
# Prediction for 093: 093
# Prediction for 094: 094
# Prediction for 095: 064
# Prediction for 096: 096
# Prediction for 097: 097
# Prediction for 098: 018
# Prediction for 099: 090
# Prediction for 100: 031
# Prediction for 101: 048
# Prediction for 102: 102
# Prediction for 103: 103
# Prediction for 104: 104
# Prediction for 105: 105
# Prediction for 106: 106
# Prediction for 107: 025
# Prediction for 108: 108
# Prediction for 109: 109
# Prediction for 110: 110
# Prediction for 111: 111
# Prediction for 112: 083
# Prediction for 113: 113
# Prediction for 114: 114
# Prediction for 115: 010
# Prediction for 116: 081
# Prediction for 117: 086
# Prediction for 118: 034
# Prediction for 119: 057
# Prediction for 120: 120
# Prediction for 121: 121
# Prediction for 122: 122
# Prediction for 123: 123
# Prediction for 124: 124
# CORRECT PREDICTION: 78
# ACCURACY OF THE MODEL :  62.903225806451616 %

#TRAIN With view of 108 degree

# Prediction for 001: 049
# Prediction for 002: 002
# Prediction for 003: 072
# Prediction for 004: 004
# Prediction for 005: 005
# Prediction for 006: 006
# Prediction for 007: 043
# Prediction for 008: 008
# Prediction for 009: 121
# Prediction for 010: 010
# Prediction for 011: 088
# Prediction for 012: 012
# Prediction for 013: 071
# Prediction for 014: 014
# Prediction for 015: 015
# Prediction for 016: 043
# Prediction for 017: 017
# Prediction for 018: 018
# Prediction for 019: 045
# Prediction for 020: 001
# Prediction for 021: 021
# Prediction for 022: 022
# Prediction for 023: 023
# Prediction for 024: 073
# Prediction for 025: 089
# Prediction for 026: 026
# Prediction for 027: 068
# Prediction for 028: 008
# Prediction for 029: 029
# Prediction for 030: 094
# Prediction for 031: 031
# Prediction for 032: 062
# Prediction for 033: 033
# Prediction for 034: 034
# Prediction for 035: 035
# Prediction for 036: 045
# Prediction for 037: 117
# Prediction for 038: 043
# Prediction for 039: 039
# Prediction for 040: 040
# Prediction for 041: 121
# Prediction for 042: 095
# Prediction for 043: 043
# Prediction for 044: 044
# Prediction for 045: 072
# Prediction for 046: 046
# Prediction for 047: 096
# Prediction for 048: 055
# Prediction for 049: 049
# Prediction for 050: 103
# Prediction for 051: 123
# Prediction for 052: 121
# Prediction for 053: 019
# Prediction for 054: 048
# Prediction for 055: 008
# Prediction for 056: 025
# Prediction for 057: 119
# Prediction for 058: 058
# Prediction for 059: 059
# Prediction for 060: 060
# Prediction for 061: 061
# Prediction for 062: 062
# Prediction for 063: 010
# Prediction for 064: 042
# Prediction for 065: 065
# Prediction for 066: 066
# Prediction for 067: 023
# Prediction for 068: 068
# Prediction for 069: 069
# Prediction for 070: 051
# Prediction for 071: 071
# Prediction for 072: 072
# Prediction for 073: 073
# Prediction for 074: 074
# Prediction for 075: 075
# Prediction for 076: 076
# Prediction for 077: 077
# Prediction for 078: 078
# Prediction for 079: 111
# Prediction for 080: 080
# Prediction for 081: 027
# Prediction for 082: 082
# Prediction for 083: 072
# Prediction for 084: 084
# Prediction for 085: 085
# Prediction for 086: 106
# Prediction for 087: 087
# Prediction for 088: 088
# Prediction for 089: 085
# Prediction for 090: 099
# Prediction for 091: 091
# Prediction for 092: 062
# Prediction for 093: 093
# Prediction for 094: 001
# Prediction for 095: 105
# Prediction for 096: 096
# Prediction for 097: 097
# Prediction for 098: 098
# Prediction for 099: 120
# Prediction for 100: 100
# Prediction for 101: 101
# Prediction for 102: 034
# Prediction for 103: 103
# Prediction for 104: 104
# Prediction for 105: 105
# Prediction for 106: 111
# Prediction for 107: 107
# Prediction for 108: 108
# Prediction for 109: 109
# Prediction for 110: 110
# Prediction for 111: 039
# Prediction for 112: 101
# Prediction for 113: 029
# Prediction for 114: 017
# Prediction for 115: 115
# Prediction for 116: 094
# Prediction for 117: 037
# Prediction for 118: 034
# Prediction for 119: 027
# Prediction for 120: 119
# Prediction for 121: 121
# Prediction for 122: 116
# Prediction for 123: 104
# Prediction for 124: 046
# CORRECT PREDICTION: 65
# ACCURACY OF THE MODEL :  52.41935483870968 %