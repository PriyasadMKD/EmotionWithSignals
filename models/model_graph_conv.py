import torch.nn as nn
from graph_model import GATLayer,GCNLayer
from torch.autograd import Variable
from torch_sinc import SincConv_fast as SincConv
import torch
import torch.nn.functional as F
import numpy as np

class EncoderNetwork(nn.Module):

    def __init__(
        self,
        input_shape,
        cnn_N_filt,
        cnn_len_filt,
        cnn_max_pool_len,
        sampling_rate
    ):

        super(EncoderNetwork,self).__init__()

        self.input_shape = input_shape
        self.cnn_N_filt = cnn_N_filt    
        self.cnn_len_filt = cnn_len_filt
        self.cnn_max_pool_len = cnn_max_pool_len
        self.sampling_rate  = sampling_rate

        self.sinc_conv = SincConv(self.cnn_N_filt[0],self.cnn_len_filt[0],self.sampling_rate)
        self.conv1d2 = nn.Conv1d(self.cnn_N_filt[0],self.cnn_N_filt[1],self.cnn_len_filt[1])
        self.conv1d3 = nn.Conv1d(self.cnn_N_filt[1],self.cnn_N_filt[2],self.cnn_len_filt[2])
        self.maxpool1 = nn.MaxPool1d(self.cnn_max_pool_len[0])
        self.maxpool2 = nn.MaxPool1d(self.cnn_max_pool_len[1])
        self.maxpool3 = nn.MaxPool1d(self.cnn_max_pool_len[2])
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()

    def forward(self,inputs):

        # rinput = torch.reshape(inputs,(125,1,250))
        x1 = self.sinc_conv(inputs)
        x1 = self.maxpool1(x1)
        x1 = self.lrelu1(x1)

        # print(x1.shape)
        x = x1
        # print(x.shape)
        x = self.conv1d2(x)
        x = self.maxpool2(x)
        x = self.lrelu2(x)
        x = self.conv1d3(x)
        x = self.maxpool3(x)
        x = self.lrelu3(x)
        x = self.flatten(x)

        # print(x.shape)
        return x
      

class GraphNet(nn.Module):

    def __init__(
        self,
        modes=14,
        num_class=1,
        cnn_N_filt=[80,40,40],
        cnn_len_filt=[16,5,5],
        cnn_max_pool_len=[2,2,2],
        sampling_rate=128,
    ):
    
        super(GraphNet,self).__init__()

        self.modes = modes
        self.num_class = num_class
        self.cnn_N_filt = cnn_N_filt
        self.cnn_len_filt = cnn_len_filt
        self.cnn_max_pool_len = cnn_max_pool_len
        self.sampling_rate = sampling_rate
        
        self.encoder_network_list = []

        for i in range(self.modes):
            self.encoder_network_list.append(EncoderNetwork([1,128],self.cnn_N_filt,self.cnn_len_filt,self.cnn_max_pool_len,self.sampling_rate))

        self.base1 = GATLayer(440,440,2)
        self.base2 = GATLayer(440,440,2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(440*14,1024)
        
        self.lin2a = nn.Linear(1024,64)
        self.lin3a = nn.Linear(64,2)

        self.lin2v = nn.Linear(1024,64)
        self.lin3v = nn.Linear(64,2)

        self.lin2d = nn.Linear(1024,64)
        self.lin3d = nn.Linear(64,2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        self.batchnorm1 = nn.BatchNorm1d(1024)

        self.batchnorm2a = nn.BatchNorm1d(64)
        self.batchnorm2v = nn.BatchNorm1d(64)
        self.batchnorm2d = nn.BatchNorm1d(64)


        self.encoder_network_list = torch.nn.ModuleList(self.encoder_network_list)

    def forward(self,x):

        x,adj_matrix = x

        fused_out = None

        for i in range(self.modes):
            
            model_i_data = torch.unsqueeze(x[:,i,:],1)

            mode_i_out = self.encoder_network_list[i](model_i_data)
            
            if fused_out == None:
                fused_out = torch.unsqueeze(mode_i_out,axis=1)
            else:
                fused_out = torch.cat((fused_out,torch.unsqueeze(mode_i_out,axis=1)),axis=1)
        
        # print(fused_out.shape)

        x = self.base1(fused_out,adj_matrix)
        # print(x.shape)
        # print(x)
        x = self.base2(x,adj_matrix)
        # print(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.lin1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        xa = self.lin2a(x)
        xa = self.batchnorm2a(xa)
        xa = self.relu(xa)
        xa = self.dropout(xa)
        xa = self.lin3a(xa)

        xv = self.lin2v(x)
        xv = self.batchnorm2v(xv)
        xv = self.relu(xv)
        xv = self.dropout(xv)
        xv = self.lin3v(xv)

        xd = self.lin2d(x)
        xd = self.batchnorm2d(xd)
        xd = self.relu(xd)
        xd = self.dropout(xd)
        xd = self.lin3d(xd)

        return [xa,xv,xd]
