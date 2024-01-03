import os


wav_file = "./D-TDNN-master/data_dir/wav.scp"

trials_file = "./D-TDNN-master/data_dir/trials"

storage_trials_file = []

with open(trials_file) as file:
    for line in file:
        storage_trials_file.append(line.split(' ')[0])
        storage_trials_file.append(line.split(' ')[1])

file =  open(wav_file,"w")

for item in storage_trials_file: 
  file.write("./D-TDNN-master/data_dir/"+item.split('-')[0]+"/"+item.split('-')[1]+'/'+item.split('-')[2]+".wav\n")