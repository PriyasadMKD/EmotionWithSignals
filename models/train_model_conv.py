
import utils
from model_conv import Conv3DNet
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

def create_batches_rnd(data,batch_size,modes,classes,phase,standardize):

    batch_labels = np.zeros((batch_size))
    batch_features_array= np.zeros((batch_size,1,128,8,9))
    # Subject_0_Video_3_4_3_2_Segment138.0_ECG_STIM.npy

    # # node --> AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

    loc_dict = {0:[0,3],1:[1,0],2:[1,2],3:[2,1],4:[3,0],5:[5,0],6:[7,3],7:[7,5],8:[5,8],9:[3,8],10:[2,7],11:[1,6],12:[1,8],13:[0,5]}


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
        
        batch_labels[i] = class_label

        temp_data = np.zeros((8,9,128,1))
        node_data = (np.load(data[i].strip(),allow_pickle=True)-standardize[0])/standardize[1]

        for j in range(14):
            locx,locy = loc_dict[j]
            temp_data[locx][locy] = np.expand_dims(node_data[j],axis=1)
        
        temp_data = np.transpose(temp_data,(3,2,0,1))

        batch_features_array[i] = temp_data

    print(batch_features_array)

    inputs = Variable(torch.from_numpy(batch_features_array).float().cuda().contiguous())
    labels=Variable(torch.from_numpy(batch_labels).float().cuda().contiguous())
    
    return inputs,labels 


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

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

        test_flag = 0
        model.train()

        loss_sum_tr=0
        acc_sum_tr=0

        pbar = trange(train_batches,unit="it", position=0, leave=True)

        for i in pbar:
  
            batch_data = train_data[i*train_batch:(i+1)*train_batch]
            
            inputs,labels = create_batches_rnd(batch_data,train_batch,modes,classes,"train",standardize)

            prediction = model(inputs)

            print(prediction)

            bce_loss = cost(prediction, labels.unsqueeze(1))
            acc = binary_acc(prediction, labels.unsqueeze(1))

            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            loss_sum_tr = loss_sum_tr+bce_loss.detach()
            acc_sum_tr = acc_sum_tr+acc.detach()

            pbar.set_description(f"Epoch {epoch+1}/{epochs} : loss = {utils.torch_round(loss_sum_tr/(i+1),4)}, acc = {utils.torch_round(acc_sum_tr/(i+1),4)}")


        pbar.close()

        loss_tot_tr = loss_sum_tr/train_batches
        acc_tot_tr = acc_sum_tr/train_batches

        print("End of epoch %i training : loss_tr=%f acc_tr=%f " % (epoch+1, utils.torch_round(loss_tot_tr,4),utils.torch_round(acc_tot_tr,4)))
        
        accuracy_history_tr.append(utils.torch_round(acc_tot_tr,4))
        loss_history_tr.append(utils.torch_round(loss_tot_tr,4))

        # ============================= VAL ======================================

        model.eval()

        test_flag = 1
        loss_sum_val = 0
        acc_sum_val = 0
        
        with torch.no_grad():
            
            for i in range(val_batches):
                
                batch_data = val_data[i*val_batch:(i+1)*val_batch]
                
                inputs,labels = create_batches_rnd(batch_data,val_batch,modes,classes,"val",standardize)

                prediction = model(inputs)

                bce_loss = cost(prediction, labels.unsqueeze(1))
                acc = binary_acc(prediction,labels.unsqueeze(1))
                loss_sum_val = loss_sum_val+bce_loss.detach()
                acc_sum_val = acc_sum_val+acc.detach()

        
        loss_tot_val = loss_sum_val/val_batches
        acc_tot_val = acc_sum_val/val_batches

        print("End of epoch %i validation : loss_val=%f acc_val=%f " % (epoch+1, utils.torch_round(loss_tot_val,4),utils.torch_round(acc_tot_val,4)))

        if utils.torch_round(acc_tot_val)>best_val_acc:
            best_val_acc = utils.torch_round(acc_tot_val)
            checkpoint={'model':model.state_dict()}  
            torch.save(checkpoint,save+''+save_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(best_val_acc)+'.pkl')
            print("Weight saved to " + save+''+save_name.split("/")[-1].replace(".pkl","")+"_"+str(epoch+1)+"_"+str(best_val_acc)+'.pkl')


        accuracy_history_val.append(utils.torch_round(acc_tot_val,4))
        loss_history_val.append(utils.torch_round(loss_tot_val,4))

        #    =============================== TEST =====================================

        model.eval()

        test_flag = 1
        loss_sum_te = 0
        acc_sum_te = 0
        
        with torch.no_grad():
            
            for i in range(test_batches):

                batch_data = test_data[i*test_batch:(i+1)*test_batch]
                inputs,labels = create_batches_rnd(batch_data,test_batch,modes,classes,"test",standardize)

                prediction = model(inputs)

                bce_loss = cost(prediction, labels.unsqueeze(1))
                acc = binary_acc(prediction, labels.unsqueeze(1))

                loss_sum_te = loss_sum_te+bce_loss.detach()
                acc_sum_te = acc_sum_te+acc.detach()

        loss_tot_te = loss_sum_te/test_batches
        acc_tot_te = acc_sum_te/test_batches

        print("End of epoch %i : loss_test=%f acc_test=%f" % (epoch+1,utils.torch_round(loss_tot_te,4),utils.torch_round(acc_tot_te,4)))

        accuracy_history_te.append(utils.torch_round(acc_tot_te,4))
        loss_history_te.append(utils.torch_round(loss_tot_te,4))

  
    pickle_dict = {}

    pickle_dict["tr_losss"] = loss_history_tr
    pickle_dict["tr_acc"] = accuracy_history_tr
    
    pickle_dict["val_loss"] = loss_history_val
    pickle_dict["val_acc"] = accuracy_history_val

    pickle_dict["te_loss"] = loss_history_te
    pickle_dict["te_acc"] = accuracy_history_te
    
    with open(save_name, 'wb') as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_model(weight):

    # this section needs to be re-written according to the model and the encoders

    graph_model = Conv3DNet()

    if weight !=None:   
        graph_model.load_state_dict(torch.load(weight)['model'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph_model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(graph_model.parameters(), lr=0.001)

    return graph_model,optimizer,criterion

def execute(data_path,selection_regex,modes,train_batch,val_batch,test_batch,epochs,save_name,weight):

    train_set,val_set,test_set,subject_dataarray = utils.get_10_fold_EEG(selection_regex)

    dataarray= np.stack(subject_dataarray, axis=0)

    mean,std = np.mean(dataarray),np.std(dataarray)
    print(mean,std)

    model,optimizer,criterion = get_model(weight)
    
    # models are defined and intialized
    classes = 2

    generator(model,optimizer,train_set,val_set,test_set,train_batch,val_batch,test_batch,modes,epochs,classes,"../weights/",criterion,save_name,[mean,std])


if __name__ == "__main__":

    # Argument parsing starts here

    parser = argparse.ArgumentParser(description='Argument Passing for TUH_Seizure Dataset')

    parser.add_argument('--btr', help='training batch size', default=64)
    parser.add_argument('--btv', help='validation batch size', default=16)
    parser.add_argument('--bte', help='evaluation batch size', default=16)
    parser.add_argument('-m', help='mode limit', default=14)
    parser.add_argument('-e', help='epochs', default=60)
    parser.add_argument('--path', help='path to the fold files', default="/home/n10370986/Dreamer/data/segmented/")
    parser.add_argument('-f', help="unique class for test file", default="EEG_0")
    parser.add_argument('--sn', help="save name to the file", default='/home/n10370986/Dreamer/history/Conv3D_EEG_FOLD_0.pkl')
    parser.add_argument('-w', help="weight", default=None)
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

    execute(data_path,selection_regex,modes,train_batch,val_batch,test_batch,epochs,save_name,weight)




















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