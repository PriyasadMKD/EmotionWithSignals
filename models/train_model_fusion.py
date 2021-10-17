import utils
from model_graph_conv import GraphNet
import os
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
from tqdm import trange 
import pickle        
from fusion_model import FusionModel

def create_batches_rnd(data,batch_size,modes,classes,phase,standardize):

    batch_labelsd = np.zeros((batch_size))
    batch_labelsv = np.zeros((batch_size))
    batch_labelsa = np.zeros((batch_size))
    batch_features_array_eeg= np.zeros((batch_size,14,128))
    batch_features_array_ecg= np.zeros((batch_size,2,256))

    batch_adj_matrix = np.zeros((batch_size,14,14))
    # Subject_0_Video_3_4_3_2_Segment138.0_ECG_STIM.npy

    # # node --> AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

    standardize_eeg = standardize[0]
    standardize_ecg = standardize[1]

    adj_matrix = np.array([
                            [1,1,1,1,1,0,0,0,0,0,0,1,0,1],
                            [1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                            [1,1,1,1,1,0,0,0,0,0,0,0,0,1],
                            [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                            [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                            [1,0,0,0,0,0,0,0,0,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,0,1,1,1,1,1],
                            [1,0,1,0,0,0,0,0,0,1,1,1,1,1]
                            ])


    for i in range(batch_size):

        file_name = data[i].strip().split("/")[-1].split("_")

        valence = int(file_name[4])
        dominance = int(file_name[5])
        arousal = int(file_name[6])

        # consider valence only

        if valence <=3 :
            class_label = 0
        else:
            class_label = 1
        
        batch_labelsv[i] = class_label

        if dominance <=3 :
            class_label = 0
        else:
            class_label = 1
        
        batch_labelsd[i] = class_label

        if arousal <=3 :
            class_label = 0
        else:
            class_label = 1
        
        batch_labelsa[i] = class_label

        # batch_features_array[i] = (np.load(data[i].strip(),allow_pickle=True)-standardize[0])/standardize[1]

        temp_data_eeg = np.load(data[i].strip(),allow_pickle=True)
        temp_data_ecg = np.load(data[i].strip().replace('EEG','ECG'),allow_pickle=True)

        for k in range(14):
            batch_features_array_eeg[i][k] = (temp_data_eeg[k]-standardize_eeg[k][0])/standardize_eeg[k][1]
            
        for k in range(2):
            batch_features_array_ecg[i][k] = (temp_data_ecg[k]-standardize_ecg[k][0])/standardize_ecg[k][1]

        # batch_features_array[i] = (np.load(data[i].strip(),allow_pickle=True)-standardize[0])/standardize[1]

        batch_adj_matrix[i] = adj_matrix

    # print(batch_features_array)

    inputs = [Variable(torch.from_numpy(batch_features_array_eeg).float().cuda().contiguous()),Variable(torch.from_numpy(batch_features_array_ecg).float().cuda().contiguous()),Variable(torch.from_numpy(batch_adj_matrix).float().cuda().contiguous())]
    labelsa=Variable(torch.from_numpy(batch_labelsa).long().cuda().contiguous())
    labelsv=Variable(torch.from_numpy(batch_labelsv).long().cuda().contiguous())
    labelsd=Variable(torch.from_numpy(batch_labelsd).long().cuda().contiguous())

    
    return inputs,[labelsa,labelsv,labelsd] 


def generator(model,optimizer,train_data,val_data,test_data,train_batch,val_batch,test_batch,modes,epochs,classes,save,cost,save_name,standardize):

    model.cuda()

    train_batches = len(train_data)//train_batch
    val_batches = len(val_data)//val_batch
    test_batches = len(test_data)//test_batch

    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(val_data)

    best_val_acc = 0
    
    loss_history_tr = []
    accuracy_history_tr = []
    loss_history_te = []
    accuracy_history_te = []
    loss_history_val = []
    accuracy_history_val = []

    # =================== FOR TEST PURPOSES =========================

    # train_batches = 1
    # val_batches = 1
    # test_batches = 1

    for epoch in range(epochs):

        if epoch == 50:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001


        test_flag = 0
        model.train()

        loss_sum_tr=0
        acc_sum_tra=0
        acc_sum_trv=0
        acc_sum_trd=0

        pbar = trange(train_batches,unit="it", position=0, leave=True)

        for i in pbar:
  
            batch_data = train_data[i*train_batch:(i+1)*train_batch]
            
            inputs,labels = create_batches_rnd(batch_data,train_batch,modes,classes,"train",standardize)

            prediction = model(inputs)
            
            arousal,valence,dominance = prediction[0],prediction[1],prediction[2]

            bce_loss_a = cost(arousal, labels[0].long())
            acc_a = utils.multi_acc(arousal, labels[0].long())

            bce_loss_v = cost(valence, labels[1].long())
            acc_v = utils.multi_acc(valence, labels[1].long())

            bce_loss_d = cost(dominance, labels[2].long())
            acc_d = utils.multi_acc(dominance, labels[2].long())

            bce_loss = bce_loss_a + bce_loss_v + bce_loss_d

            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            loss_sum_tr = loss_sum_tr+bce_loss.detach()
            acc_sum_tra = acc_sum_tra+acc_a.detach()
            acc_sum_trv = acc_sum_trv+acc_v.detach()
            acc_sum_trd = acc_sum_trd+acc_d.detach()

            pbar.set_description(f"Epoch {epoch+1}/{epochs} : loss = {utils.torch_round(loss_sum_tr/(i+1),4)}, acc_ar = {utils.torch_round(acc_sum_tra/(i+1),4)}, acc_val = {utils.torch_round(acc_sum_trv/(i+1),4)}, acc_dom = {utils.torch_round(acc_sum_trd/(i+1),4)}")


        pbar.close()

        loss_tot_tr = loss_sum_tr/train_batches
        acc_tot_tra = acc_sum_tra/train_batches
        acc_tot_trv = acc_sum_trv/train_batches
        acc_tot_trd = acc_sum_trd/train_batches

        print("End of epoch %i training : loss_tr=%f acc_ar=%f acc_va=%f acc_do=%f " % (epoch+1, utils.torch_round(loss_tot_tr,4),utils.torch_round(acc_tot_tra,4),utils.torch_round(acc_tot_trv,4),utils.torch_round(acc_tot_trd,4)))
        
        accuracy_history_tr.append([utils.torch_round(acc_tot_tra,4),utils.torch_round(acc_tot_trv,4),utils.torch_round(acc_tot_trd,4)])
        loss_history_tr.append(utils.torch_round(loss_tot_tr,4))

        # ============================= VAL ======================================

        model.eval()

        test_flag = 1
        loss_sum_val = 0
        acc_sum_vala = 0
        acc_sum_valv = 0
        acc_sum_vald = 0
        
        with torch.no_grad():
            
            for i in range(val_batches):
                
                batch_data = val_data[i*val_batch:(i+1)*val_batch]
                
                inputs,labels = create_batches_rnd(batch_data,val_batch,modes,classes,"val",standardize)

                prediction = model(inputs)
            
                arousal,valence,dominance = prediction[0],prediction[1],prediction[2]

                bce_loss_a = cost(arousal, labels[0].long())
                acc_a = utils.multi_acc(arousal, labels[0].long())

                bce_loss_v = cost(valence, labels[1].long())
                acc_v = utils.multi_acc(valence, labels[1].long())

                bce_loss_d = cost(dominance, labels[2].long())
                acc_d = utils.multi_acc(dominance, labels[2].long())

                bce_loss = bce_loss_a + bce_loss_v + bce_loss_d

                loss_sum_val = loss_sum_val+bce_loss.detach()
                acc_sum_vala = acc_sum_vala+acc_a.detach()
                acc_sum_valv = acc_sum_valv+acc_v.detach()
                acc_sum_vald = acc_sum_vald+acc_d.detach()

            loss_tot_val = loss_sum_val/val_batches
            acc_tot_vala = acc_sum_vala/val_batches
            acc_tot_valv = acc_sum_valv/val_batches
            acc_tot_vald = acc_sum_vald/val_batches

            print("Validation Perforomance: loss_val=%f acc_ar=%f acc_va=%f acc_do=%f " % (utils.torch_round(loss_tot_val,4),utils.torch_round(acc_tot_vala,4),utils.torch_round(acc_tot_valv,4),utils.torch_round(acc_tot_vald,4)))
        
        mean_acc = (utils.torch_round(acc_tot_vala) + utils.torch_round(acc_tot_valv) + utils.torch_round(acc_tot_vald))/3
        if mean_acc>best_val_acc:
            best_val_acc = mean_acc
            checkpoint={'model':model.state_dict()}  
            torch.save(checkpoint,save+''+save_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(best_val_acc)+'.pkl')
            print("Weight saved to " + save+''+save_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(best_val_acc)+'.pkl')


        accuracy_history_val.append([utils.torch_round(acc_tot_vala,4),utils.torch_round(acc_tot_valv,4),utils.torch_round(acc_tot_vald,4)])
        loss_history_val.append(utils.torch_round(loss_tot_val,4))

        #    =============================== TEST =====================================

        model.eval()

        test_flag = 1
        loss_sum_te = 0
        acc_sum_tea = 0
        acc_sum_tev = 0
        acc_sum_ted = 0
        
        with torch.no_grad():
            
            for i in range(test_batches):

                batch_data = test_data[i*test_batch:(i+1)*test_batch]
                inputs,labels = create_batches_rnd(batch_data,test_batch,modes,classes,"test",standardize)

                prediction = model(inputs)

                arousal,valence,dominance = prediction[0],prediction[1],prediction[2]

                bce_loss_a = cost(arousal, labels[0].long())
                acc_a = utils.multi_acc(arousal, labels[0].long())

                bce_loss_v = cost(valence, labels[1].long())
                acc_v = utils.multi_acc(valence, labels[1].long())

                bce_loss_d = cost(dominance, labels[2].long())
                acc_d = utils.multi_acc(dominance, labels[2].long())

                bce_loss = bce_loss_a + bce_loss_v + bce_loss_d

                loss_sum_te = loss_sum_te+bce_loss.detach()
                acc_sum_tea = acc_sum_tea+acc_a.detach()
                acc_sum_tev = acc_sum_tev+acc_v.detach()
                acc_sum_ted = acc_sum_ted+acc_d.detach()

            loss_tot_te = loss_sum_te/test_batches
            acc_tot_tea = acc_sum_tea/test_batches
            acc_tot_tev = acc_sum_tev/test_batches
            acc_tot_ted = acc_sum_ted/test_batches

            print("Test Performance : loss_te=%f acc_ar=%f acc_va=%f acc_do=%f " % (utils.torch_round(loss_tot_te,4),utils.torch_round(acc_tot_tea,4),utils.torch_round(acc_tot_tev,4),utils.torch_round(acc_tot_ted,4)))
        
        accuracy_history_te.append([utils.torch_round(acc_tot_tea,4),utils.torch_round(acc_tot_tev,4),utils.torch_round(acc_tot_ted,4)])
        loss_history_te.append(utils.torch_round(loss_tot_te,4))

        print("============================ END OF EPOCH " +str(epoch+1)+ " =====================================")
  
    pickle_dict = {}

    pickle_dict["tr_losss"] = loss_history_tr
    pickle_dict["tr_acc"] = accuracy_history_tr
    
    pickle_dict["val_loss"] = loss_history_val
    pickle_dict["val_acc"] = accuracy_history_val

    pickle_dict["te_loss"] = loss_history_te
    pickle_dict["te_acc"] = accuracy_history_te
    
    with open(save_name, 'wb') as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_model(weight_array):

    # this section needs to be re-written according to the model and the encoders
    weight, weight_eeg,weight_ecg = weight_array

    fusion_model = FusionModel(weight_eeg,weight_ecg)

    if weight !=None:   
        fusion_model.load_state_dict(torch.load(weight)['model'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fusion_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

    return fusion_model,optimizer,criterion


def execute(data_path,selection_regex,modes,train_batch,val_batch,test_batch,epochs,save_name,weight):

    # train_set,val_set,test_set,subject_dataarray = get_LOSO(data_path,selection_regex)

    train_set,val_set,test_set,subject_dataarray = utils.get_10_fold_EEG_ECG_for_Fusion(selection_regex)
    
    # print(len(train_set),len(val_set),len(test_set))

    # dataarray= np.stack(subject_dataarray, axis=0)

    # mean,std = np.mean(dataarray),np.std(dataarray)
    # print(mean,std)

    model,optimizer,criterion = get_model(weight)
    
    classes = 2

    # mean_std_array = [mean,std]
    # generator(model,optimizer,train_set,val_set,test_set,train_batch,val_batch,test_batch,modes,epochs,classes,"../weights/",criterion,save_name,mean_std_array)
    
    generator(model,optimizer,train_set,val_set,test_set,train_batch,val_batch,test_batch,modes,epochs,classes,"../weights/",criterion,save_name,subject_dataarray)

if __name__ == "__main__":

    # Argument parsing starts here

    parser = argparse.ArgumentParser(description='Argument Passing for TUH_Seizure Dataset')

    parser.add_argument('--btr', help='training batch size', default=64)
    parser.add_argument('--btv', help='validation batch size', default=6)
    parser.add_argument('--bte', help='evaluation batch size', default=6)
    parser.add_argument('-m', help='mode limit', default=14)
    parser.add_argument('-e', help='epochs', default=80)
    parser.add_argument('--path', help='path to the fold files', default="/home/n10370986/Dreamer/data/segmented/")
    parser.add_argument('-f', help="unique class for test file", default="EEG_0")
    parser.add_argument('--sn', help="save name to the file", default='/home/n10370986/Dreamer/history/Emotion_Fold_0_Conv_Channel_Fusion.pkl')
    parser.add_argument('-w', help="weight", default=None)
    parser.add_argument('-weeg', help="weight", default='../weights/Emotion_Fold_0_Conv_Channel_53_88.39890000000001.pkl')
    parser.add_argument('-wecg', help="weight", default='../weights/ECG_FOLD_0_Conv_Channel_1_55.58176666666666.pkl')

    args = parser.parse_args()

    opt_dict = vars(parser.parse_args())    

    assert opt_dict["path"] != None, "no path for data is given"
    assert os.path.exists(opt_dict["path"]), "no path exisits"

    data_path = opt_dict["path"]
    selection_regex = opt_dict["f"]
    train_batch = int(opt_dict["btr"])
    val_batch = int(opt_dict["btv"])
    test_batch = int(opt_dict["bte"])
    modes = int(opt_dict["m"])
    epochs = int(opt_dict["e"])
    save_name = opt_dict["sn"]
    weight = opt_dict["w"]
    weighteeg = opt_dict["weeg"]
    weightecg = opt_dict["wecg"]

    for i in range(0,10,1):
        
        save_name = save_name.replace("Fold_0","Fold_"+str(i))
        selection_regex = selection_regex.replace("0",str(i))

        print("Training for Fold " + str(i) + " as Test")
        print(save_name,selection_regex)
        
        execute(data_path,selection_regex,modes,train_batch,val_batch,test_batch,epochs,save_name,[weight,weighteeg,weightecg])

        break


















# adj_matrix = torch.Tensor([
#                             [1,1,1,1,1,0,0,0,0,0,0,1,0,1],
#                             [1,1,1,1,1,0,0,0,0,0,0,0,0,0],
#                             [1,1,1,1,1,0,0,0,0,0,0,0,0,1],
#                             [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
#                             [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
#                             [0,0,0,1,1,1,1,0,0,0,0,0,0,0],
#                             [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
#                             [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
#                             [0,0,0,0,0,0,0,1,1,1,1,0,0,0],
#                             [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
#                             [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
#                             [1,0,0,0,0,0,0,0,0,1,1,1,1,1],
#                             [0,0,0,0,0,0,0,0,0,1,1,1,1,1],
#                             [1,0,1,0,0,0,0,0,0,1,1,1,1,1]
#                             ])

# # print(adj_matrix.shape)
# node_feats = torch.arange(48, dtype=torch.float32).view(1, 14, 4)

# # adj_matrix = torch.Tensor([[[1, 1, 0, 0],
# #                             [1, 1, 1, 1],
# #                             [0, 1, 1, 1],
# #                             [0, 1, 1, 1]],[[1, 1, 0, 0],
# #                             [1, 1, 1, 1],
# #                             [0, 1, 1, 1],
# #                             [0, 1, 1, 1]]])

# print("Node features:\n", node_feats)
# print("\nAdjacency matrix:\n", adj_matrix)
# print(node_feats.shape)
# print(adj_matrix.shape)

# # layer = GCNLayer(c_in=2, c_out=2)
# # layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
# # layer.projection.bias.data = torch.Tensor([0., 0.])

# # with torch.no_grad():
# #     out_feats = layer(node_feats, adj_matrix)


# layer = GATLayer(2, 2, num_heads=2)
# # layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
# # layer.projection.bias.data = torch.Tensor([0., 0.])
# # layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

# with torch.no_grad():
#     out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

# print("Adjacency matrix", adj_matrix)
# print("Input features", node_feats)
# print("Output features", out_feats)