import os
from random import shuffle

# data_path = '/home/n10370986/Dreamer/data/segmented/'

# data_files = []

# for subject_t in range(23):
#     print(subject_t)
#     listOfFiles = list()
#     for (dirpath, dirnames, filenames) in os.walk(data_path+"Subject"+str(subject_t)+"/"):
#         listOfFiles += [os.path.join(dirpath, file) for file in filenames]

#     for file_t in listOfFiles:

#         if "IMAGE" not in file_t and 'EEG' in file_t and '222S' in file_t:
#             data_files.append(file_t.strip())

data_files = open('eeg_data_segmented_4444S_.txt','r').readlines()
shuffle(data_files)

print(len(data_files))

per_fold = len(data_files)//10

fold1 = data_files[:per_fold]
fold2 = data_files[per_fold:per_fold*2]
fold3 = data_files[per_fold*2:per_fold*3]
fold4 = data_files[per_fold*3:per_fold*4]
fold5 = data_files[per_fold*4:per_fold*5]
fold6 = data_files[per_fold*5:per_fold*6]
fold7 = data_files[per_fold*6:per_fold*7]
fold8 = data_files[per_fold*7:per_fold*8]
fold9 = data_files[per_fold*8:per_fold*9]
fold10 = data_files[per_fold*9:]

folds = [fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10]

for i in range(10):
    data_list = folds[i]
    print(len(data_list))
    with open("./folds/Fold_"+str(i)+'_EEG_44S_STIM.txt', 'w') as f:
        for item in data_list:
            f.write("%s\n" % item.strip())

