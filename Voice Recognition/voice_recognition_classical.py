import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture


train_dir = "Voice Dataset/train/"
test_dir = "Voice Dataset/test/"

gmm_list = []

for filename in os.listdir(train_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(train_dir, filename)
        print(f"Processing {filepath}")
        
        y, sr = librosa.load(filepath)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36)
        
        gmm = GaussianMixture(n_components=32, random_state=0)
        gmm.fit(mfcc.T)
        
        gmm_list.append(gmm)

for filename in os.listdir(test_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(test_dir, filename)
        print(f"Processing {filepath}")
        
        y, sr = librosa.load(filepath)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=36)
        
        scores = [gmm.score(mfcc.T) for gmm in gmm_list]
        
        max_index = np.argmax(scores)
        
        print(f"Detected speaker: {os.path.splitext(os.path.basename(train_dir))[0]}_{max_index+1}")


#If I change n_mfcc = 30 
# Detected speaker: _9
# Processing Voice Dataset/test/s2.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s3.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s4.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s5.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s6.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s7.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s8.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s9.wav
# Detected speaker: _9

# Obviously that is awfully wrong


#if i change n_components = 10 


# Processing Voice Dataset/test/s1.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s2.wav
# Detected speaker: _2
# Processing Voice Dataset/test/s3.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s4.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s5.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s6.wav
# Detected speaker: _6
# Processing Voice Dataset/test/s7.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s8.wav
# Detected speaker: _9
# Processing Voice Dataset/test/s9.wav
# Detected speaker: _9

# And if I increase that is partially the same result. But with some error

# Processing Voice Dataset/test/s1.wav
# Detected speaker: _8
# Processing Voice Dataset/test/s2.wav
# Detected speaker: _2
# Processing Voice Dataset/test/s3.wav
# Detected speaker: _7
# Processing Voice Dataset/test/s4.wav
# Detected speaker: _4
# Processing Voice Dataset/test/s5.wav
# Detected speaker: _5
# Processing Voice Dataset/test/s6.wav
# Detected speaker: _6
# Processing Voice Dataset/test/s7.wav
# Detected speaker: _7
# Processing Voice Dataset/test/s8.wav
# Detected speaker: _8
# Processing Voice Dataset/test/s9.wav
# Detected speaker: _9