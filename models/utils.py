import os
import numpy as np
import sys
import random
from sklearn.preprocessing import minmax_scale
import torch
import math


def torch_round(tensor,decimals=4):
    return round(tensor.item(),decimals)


def get_LOSO_EEG(data_path,selection_regex):

    modality,subject = selection_regex.split("_")

    train_set = []
    test_set = []
    val_set = []

    
    channel_mean = {}
    channel_var = {}

    if subject != 22:
        val_subject = int(subject)+1
    else:
        val_subject = 0

    subject_dataarray = []

    for subject_t in range(23):
        
        
        filtered_files = open('/home/n10370986/Dreamer/utils/folds/Subject_'+str(subject_t)+'_EEG_STIM.txt',"r").readlines()

        # print("Subject "+str(subject_t))

        if subject == str(subject_t):
            test_set+=filtered_files
            val_set+=filtered_files
        else:
            train_set+=filtered_files

            for file_t in filtered_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def get_LOSO_ECG(data_path,selection_regex):

    modality,subject = selection_regex.split("_")
    
    train_set = []
    test_set = []
    val_set = []

    val_subject = int(subject)+1

    subject_dataarray = []

    for subject_t in range(23):
        
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(data_path+"Subject"+str(subject_t)+"/"):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        filtered_files = []

        # print("Subject "+str(subject_t))

        for file_t in listOfFiles:
            if modality+"_STIM" in file_t and 'IMAGE' not in file_t:
                filtered_files.append(file_t)
                # if subject == str(subject_t):
                #     subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))
        
        # print(len(filtered_files))

        if subject == str(subject_t):
            test_set+=filtered_files
        elif val_subject == subject_t:
            val_set+=filtered_files
        else:
            train_set+=filtered_files

    print(len(train_set),len(val_set),len(test_set))

    return train_set,val_set,test_set,subject_dataarray

def get_10_fold_EEG(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)

    if test_fold == 9:
        val_fold = 0
    else:
        val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()
        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray

def get_10_fold_EEG_NEW10(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)

    if test_fold == 9:
        val_fold = 0
    else:
        val_fold = test_fold

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM_NEW10.txt","r").readlines()
        if i == test_fold:
            test_set+=data_files
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def get_10_fold_EEG_NEW10R(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)

    # if test_fold == 9:
    #     val_fold = 0
    # else:
    #     val_fold = test_fold

    val_fold = test_fold

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM_NEW10R.txt","r").readlines()
        if i == test_fold:
            test_set+=data_files
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def get_10_fold_EEG_Time(selection_regex,time):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)

    if test_fold == 9:
        val_fold = 0
    else:
        val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_"+str(time)+"S_STIM.txt","r").readlines()
        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray

def get_10_fold_EEGM1(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    for i in range(10):
        
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()

        # data_files = [w.replace('EEG', 'ECG') for w in data_files]

        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:
                subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))


    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def get_10_fold_ECG(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()

        data_files = [w.replace('EEG', 'ECG') for w in data_files]

        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(2):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray



def get_10_fold_ECG_Time(selection_regex,time):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    channel_mean = {}
    channel_var = {}

    for i in range(10):
        
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_"+str(time)+"S_STIM.txt","r").readlines()

        data_files = [w.replace('EEG', 'ECG') for w in data_files]

        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                # subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))

                temp_data = np.load(file_t.strip(),allow_pickle=True)

                # print(temp_data.shape)
                for j in range(2):
                    if j not in channel_mean:
                        channel_mean[j] = []
                        channel_var[j] = []
                    
                    channel_mean[j].append(np.mean(temp_data[j]))
                    channel_var[j].append(np.var(temp_data[j]))
    

    for key in channel_mean.keys():
        total_mean = sum(channel_mean[key])/len(channel_mean[key])
        total_var = 0
        for i in range(len(channel_mean[key])):
            total_var+=channel_var[key][i]+(channel_mean[key][i]-total_mean)**2

        total_std = (total_var/len(channel_var[key]))**0.5

        subject_dataarray.append([total_mean,total_std])

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray



def get_10_fold_ECGM1(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    for i in range(10):
        
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()

        data_files = [w.replace('EEG', 'ECG') for w in data_files]


        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            # for file_t in data_files:
            #     subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))


    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def get_10_fold_EEG_IMAGE(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()

        data_files = [w.replace('EEG_STIM', 'EEG_STIM_IMAGE') for w in data_files]

        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            # for file_t in data_files:
            #     subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))


    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray

def get_10_fold_ECG_IMAGE(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)
    val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    subject_dataarray = []

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()

        data_files = [w.replace('EEG_STIM', 'ECG_STIM_IMAGE') for w in data_files]

        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            # for file_t in data_files:
            #     # print(np.load(file_t.strip(),allow_pickle=True).shape)
            
            #     subject_dataarray.append(np.load(file_t.strip(),allow_pickle=True))


    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc



def get_10_fold_EEG_ECG_for_Fusion(selection_regex):

    modality,fold = selection_regex.split("_")

    test_fold = int(fold)

    if test_fold == 9:
        val_fold = 0
    else:
        val_fold = test_fold+1

    train_set, val_set, test_set = [],[],[]

    channel_mean_eeg = {}
    channel_var_eeg = {}

    channel_mean_ecg = {}
    channel_var_ecg = {}

    for i in range(10):
        data_files = open("/home/n10370986/Dreamer/utils/folds/Fold_"+str(i)+"_EEG_STIM.txt","r").readlines()
        if i == test_fold:
            test_set+=data_files
        elif i == val_fold:
            val_set+=data_files
        else:
            train_set+=data_files

            for file_t in data_files:

                temp_data_eeg = np.load(file_t.strip(),allow_pickle=True)
                temp_data_ecg = np.load(file_t.strip().replace('EEG','ECG'),allow_pickle=True)
                # print(temp_data.shape)
                for j in range(14):
                    if j not in channel_mean_eeg:
                        channel_mean_eeg[j] = []
                        channel_var_eeg[j] = []
                    
                    channel_mean_eeg[j].append(np.mean(temp_data_eeg[j]))
                    channel_var_eeg[j].append(np.var(temp_data_eeg[j]))

                for j in range(2):
                    if j not in channel_mean_ecg:
                        channel_mean_ecg[j] = []
                        channel_var_ecg[j] = []
                    
                    channel_mean_ecg[j].append(np.mean(temp_data_ecg[j]))
                    channel_var_ecg[j].append(np.var(temp_data_ecg[j]))

    eeg_var_mean_array = []
    for key in channel_mean_eeg.keys():
        total_mean_eeg = sum(channel_mean_eeg[key])/len(channel_mean_eeg[key])
        total_var_eeg = 0
        for i in range(len(channel_mean_eeg[key])):
            total_var_eeg+=channel_var_eeg[key][i]+(channel_mean_eeg[key][i]-total_mean_eeg)**2

        # print(total_var_eeg,len(channel_var_eeg[key]))
        total_std_eeg = (total_var_eeg/len(channel_var_eeg[key]))**0.5

        eeg_var_mean_array.append([total_mean_eeg,total_std_eeg])

    ecg_var_mean_array = []
    for key in channel_mean_ecg.keys():
        total_mean_ecg = sum(channel_mean_ecg[key])/len(channel_mean_ecg[key])
        total_var_ecg = 0
        for i in range(len(channel_mean_ecg[key])):
            total_var_ecg+=channel_var_ecg[key][i]+(channel_mean_ecg[key][i]-total_mean_ecg)**2

        total_std_ecg = (total_var_ecg/len(channel_var_ecg[key]))**0.5

        ecg_var_mean_array.append([total_mean_ecg,total_std_ecg])


    subject_dataarray = [eeg_var_mean_array,ecg_var_mean_array]

    print(subject_dataarray)

    print(len(train_set),len(test_set),len(val_set))

    return train_set,val_set,test_set,subject_dataarray


def multi_acc_prediction(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc,y_test.cpu().numpy(),y_pred_tags.cpu().numpy()


import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          color="Blues",
                          normalize=True,savename=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'font.weight': 'bold'})

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # if cmap is None:
    cmap = plt.get_cmap(color)


    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names,weight = 'bold')
        plt.yticks(tick_marks, target_names,weight = 'bold')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center", weight='bold',
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",weight='bold',
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True Label',weight = 'bold')
    plt.xlabel('Predicted Label',weight = 'bold')
    plt.title(title)
    plt.savefig(savename,bbox_inches='tight',dpi=3000)