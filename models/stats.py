from os import times
import scipy.io as spio
import numpy as np

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_data(file_name='../DREAMER.mat'):
    
    data = loadmat(file_name)['DREAMER']['Data']

    length = []

    for i in range(23):
        # break
        sub_data = data[i]

        EEG_data = sub_data.EEG
        ECG_data = sub_data.ECG

        EEG_stimuli = EEG_data.stimuli
        EEG_baseline= EEG_data.baseline

        ECG_stimuli = ECG_data.stimuli
        ECG_baseline = ECG_data.baseline

        valance = sub_data.ScoreValence
        arousal = sub_data.ScoreArousal
        dominance = sub_data.ScoreDominance

        print(valance)
        print(arousal)
        print(dominance)

        for j in range(18):

            
            stim_eeg_video_i = np.transpose(EEG_stimuli[j],(1,0))
            base_eeg_video_i = np.transpose(EEG_baseline[j],(1,0))
            stim_ecg_video_i = np.transpose(ECG_stimuli[j],(1,0))
            base_ecg_video_i = np.transpose(ECG_baseline[j],(1,0))
            
            v_video_i = valance[j]
            a_video_i = arousal[j]
            d_video_i = dominance[j]

            print(stim_eeg_video_i.shape)
            print(base_eeg_video_i.shape)
            print(stim_ecg_video_i.shape)
            print(base_ecg_video_i.shape)

            print(v_video_i,a_video_i,d_video_i)

            length.append(stim_eeg_video_i.shape[1]/128)

            save_file_eeg_stim = "../data/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_EEG_STIM.npy'
            save_file_eeg_base = "../data/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_EEG_BASE.npy'
            save_file_ecg_stim = "../data/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_ECG_STIM.npy'
            save_file_ecg_base = "../data/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_ECG_BASE.npy'

            # print(save_file_eeg_stim)
            # print(save_file_eeg_base)
            # print(save_file_ecg_stim)
            # print(save_file_ecg_base)

            # np.save(save_file_eeg_stim,stim_eeg_video_i)
            # np.save(save_file_eeg_base,base_eeg_video_i)
            # np.save(save_file_ecg_stim,stim_ecg_video_i)
            # np.save(save_file_ecg_base,base_ecg_video_i)

            # break
        
        # break

    print(sum(length))


def segment_data(file_name='../DREAMER.mat'):
    
    data = loadmat(file_name)['DREAMER']['Data']

    data_list = []

    for i in range(23):
        # break
        sub_data = data[i]

        EEG_data = sub_data.EEG
        ECG_data = sub_data.ECG

        EEG_stimuli = EEG_data.stimuli
        EEG_baseline= EEG_data.baseline

        ECG_stimuli = ECG_data.stimuli
        ECG_baseline = ECG_data.baseline

        valance = sub_data.ScoreValence
        arousal = sub_data.ScoreArousal
        dominance = sub_data.ScoreDominance

        # print(valance)
        # print(arousal)
        # print(dominance)

        for j in range(18):

            
            stim_eeg_video_i = np.transpose(EEG_stimuli[j],(1,0))
            base_eeg_video_i = np.transpose(EEG_baseline[j],(1,0))
            stim_ecg_video_i = np.transpose(ECG_stimuli[j],(1,0))
            base_ecg_video_i = np.transpose(ECG_baseline[j],(1,0))
            
            v_video_i = valance[j]
            a_video_i = arousal[j]
            d_video_i = dominance[j]

            # print(stim_eeg_video_i.shape)
            # print(base_eeg_video_i.shape)
            # print(stim_ecg_video_i.shape)
            # print(base_ecg_video_i.shape)

            # print(i,j,eeg_length,v_video_i,a_video_i,d_video_i)

            # length.append(stim_eeg_video_i.shape[1]/128)

            eeg_length = stim_eeg_video_i.shape[1]
            ecg_length = stim_ecg_video_i.shape[1]

            print(i,j,eeg_length/128,ecg_length/256,v_video_i,a_video_i,d_video_i)

            for k in range(0,eeg_length,128):
                save_file_eeg_stim = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/128)+'_EEG_STIM.npy'
                save_file_eeg_base = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/128)+'_EEG_BASE.npy'

                segment_eeg_stim = stim_eeg_video_i[:,k:k+128]
                segment_eeg_base = base_eeg_video_i[:,k:k+128]
                # print(segment_eeg_stim.shape)

                np.save(save_file_eeg_stim,segment_eeg_stim)
                np.save(save_file_eeg_base,segment_eeg_base)

                data_list.append(save_file_eeg_stim)

            for k in range(0,ecg_length,256):
                save_file_ecg_stim = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/256)+'_ECG_STIM.npy'
                save_file_ecg_base = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/256)+'_ECG_BASE.npy'

                segment_ecg_stim = stim_ecg_video_i[:,k:k+256]
                segment_ecg_base = base_ecg_video_i[:,k:k+256]

                np.save(save_file_ecg_stim,segment_ecg_stim)
                np.save(save_file_ecg_base,segment_ecg_base)

                # print(segment_ecg_stim.shape)


    with open("eeg_data_segmented.txt", "w") as f:
        for s in data_list:
            f.write(str(s) +"\n")
            # print(save_file_eeg_stim)
            # print(save_file_eeg_base)
            # print(save_file_ecg_stim)
            # print(save_file_ecg_base)

            # np.save(save_file_eeg_stim,stim_eeg_video_i)
            # np.save(save_file_eeg_base,base_eeg_video_i)
            # np.save(save_file_ecg_stim,stim_ecg_video_i)
            # np.save(save_file_ecg_base,base_ecg_video_i)
# load_data()





def segment_data_with_time(file_name='../DREAMER.mat',times=4):
    
    data = loadmat(file_name)['DREAMER']['Data']

    data_list = []

    for i in range(23):
        # break
        sub_data = data[i]

        EEG_data = sub_data.EEG
        ECG_data = sub_data.ECG

        EEG_stimuli = EEG_data.stimuli
        EEG_baseline= EEG_data.baseline

        ECG_stimuli = ECG_data.stimuli
        ECG_baseline = ECG_data.baseline

        valance = sub_data.ScoreValence
        arousal = sub_data.ScoreArousal
        dominance = sub_data.ScoreDominance

        # print(valance)
        # print(arousal)
        # print(dominance)

        for j in range(18):

            
            stim_eeg_video_i = np.transpose(EEG_stimuli[j],(1,0))
            # base_eeg_video_i = np.transpose(EEG_baseline[j],(1,0))
            stim_ecg_video_i = np.transpose(ECG_stimuli[j],(1,0))
            # base_ecg_video_i = np.transpose(ECG_baseline[j],(1,0))
            
            v_video_i = valance[j]
            a_video_i = arousal[j]
            d_video_i = dominance[j]

            # print(stim_eeg_video_i.shape)
            # print(base_eeg_video_i.shape)
            # print(stim_ecg_video_i.shape)
            # print(base_ecg_video_i.shape)

            # print(i,j,eeg_length,v_video_i,a_video_i,d_video_i)

            # length.append(stim_eeg_video_i.shape[1]/128)

            eeg_length = stim_eeg_video_i.shape[1]
            ecg_length = stim_ecg_video_i.shape[1]

            print(i,j,(eeg_length-128*times)//128*times+1,(ecg_length-256*times)//256*times+1,v_video_i,a_video_i,d_video_i)

            for k in range(0,eeg_length-128*times,128*times):
                save_file_eeg_stim = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/128)+'_EEG'+str(times)+'2S_STIM.npy'
                save_file_eeg_base = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/128)+'_EEG'+str(times)+'2S_BASE.npy'

                segment_eeg_stim = stim_eeg_video_i[:,k:k+128*times]
                # segment_eeg_base = base_eeg_video_i[:,k:k+512]
                # print(segment_eeg_stim.shape)

                np.save(save_file_eeg_stim,segment_eeg_stim)
                # np.save(save_file_eeg_base,segment_eeg_base)

                data_list.append(save_file_eeg_stim)

            for k in range(0,ecg_length-256*times,256*times):
                save_file_ecg_stim = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/256)+'_ECG'+str(times)+'2S_STIM.npy'
                save_file_ecg_base = "../data/segmented/Subject"+str(i)+"/Subject_"+str(i)+'_Video_'+str(j)+'_'+str(v_video_i)+'_'+str(a_video_i)+'_'+str(d_video_i)+'_Segment'+str(k/256)+'_ECG'+str(times)+'2S_BASE.npy'

                segment_ecg_stim = stim_ecg_video_i[:,k:k+256*times]
                # segment_ecg_base = base_ecg_video_i[:,k:k+1024]

                np.save(save_file_ecg_stim,segment_ecg_stim)
                # np.save(save_file_ecg_base,segment_ecg_base)

                # print(segment_ecg_stim.shape)


    with open("eeg_data_segmented_"+str(times)+"2S_.txt", "w") as f:
        for s in data_list:
            f.write(str(s) +"\n")
            # print(save_file_eeg_stim)
            # print(save_file_eeg_base)
            # print(save_file_ecg_stim)
            # print(save_file_ecg_base)

            # np.save(save_file_eeg_stim,stim_eeg_video_i)
            # np.save(save_file_eeg_base,base_eeg_video_i)
            # np.save(save_file_ecg_stim,stim_ecg_video_i)
            # np.save(save_file_ecg_base,base_ecg_video_i)
# load_data()

segment_data_with_time(time=2)
# segment_data_with_time(time=2)




