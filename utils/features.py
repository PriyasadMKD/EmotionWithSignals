import logging
import os
from pathlib import Path
import subprocess
from typing import Union
import wave

import numpy as np
import python_speech_features
import scipy.io.wavfile as wav


def extract_energy(rate, sig):
    """ Extracts the energy of frames. """

    mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
    energy_row_vec = mfcc[:, 0]
    energy_col_vec = energy_row_vec[:, np.newaxis]
    return energy_col_vec

def fbank(sig,rate, flat=True):
    """ Currently grabs log Mel filterbank, deltas and double deltas."""

    fbank_feat = python_speech_features.logfbank(sig, rate, nfilt=40)
    energy = extract_energy(rate, sig)
    feat = np.hstack([energy, fbank_feat])
    delta_feat = python_speech_features.delta(feat, 2)
    delta_delta_feat = python_speech_features.delta(delta_feat, 2)
    # print(feat.shape,delta_feat.shape,delta_delta_feat.shape)
    
    feat = np.expand_dims(feat,axis=2)
    delta_feat = np.expand_dims(delta_feat,axis=2)
    delta_delta_feat = np.expand_dims(delta_delta_feat,axis=2)
    
    all_feats = [feat, delta_feat, delta_delta_feat]

    all_feats = np.concatenate(np.array(all_feats), axis=2)
    # Log Mel Filterbank, with delta, and double delta
    # feat_fn = wav_path[:-3] + "fbank.npy"
    # np.save(feat_fn, all_feats)
    return all_feats

count = 0

for subject_t in range(23):
        
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk('/home/n10370986/Dreamer/data/segmented/'+"Subject"+str(subject_t)+"/"):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for file_t in listOfFiles:

        if "STIM" in file_t and 'EEG' in file_t and 'IMAGE' not in file_t:

            count+=1

            data = np.load(file_t.strip(),allow_pickle=True)

            stacked_data = []

            for i in range(14):
                waved = data[i]
                stacked_mfcc = fbank(waved,128)
                stacked_data.append(stacked_mfcc)

            stacked_data= np.stack(stacked_data, axis=0)

            # print(stacked_data.shape)

            save_name = file_t.strip().replace(".npy","_IMAGE.npy")
            # print(save_name,file_t.strip())
            np.save(save_name,stacked_data)
            
            print(count,save_name)

        if "STIM" in file_t and 'ECG' in file_t and 'IMAGE' not in file_t:

            count+=1

            data = np.load(file_t.strip(),allow_pickle=True)

            stacked_data = []

            for i in range(2):
                waved = data[i]
                stacked_mfcc = fbank(waved,256)
                stacked_data.append(stacked_mfcc)

            stacked_data= np.stack(stacked_data, axis=0)

            # print(stacked_data.shape)

            save_name = file_t.strip().replace(".npy","_IMAGE.npy")
            # print(save_name,file_t.strip())
            np.save(save_name,stacked_data)
            
            print(count,save_name)
