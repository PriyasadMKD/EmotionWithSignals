import torch.nn as nn
from graph_model import GATLayer,GCNLayer

class Conv3DNet(nn.Module):

    def __init__(
        self
    ):
    
        super(Conv3DNet,self).__init__()

        self.base1 = nn.Conv3d(1,64,3)
        self.base2 = nn.Conv3d(64,64,3)
        self.maxp = nn.MaxPool3d(2)
        # (N,C,D,H,W)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(256*62,512)
        # self.lin2 = nn.Linear(512,64)
        self.lin3 = nn.Linear(512,1)
        # self.out = nn.Sigmoid()
        # self.posnorm = nn.BatchNorm1d(512)
        # self.negnorm = nn.BatchNorm1d(512)
        # self.anchornorm = nn.BatchNorm1d(512)
        
    def forward(self,x):

        x = self.base1(x)
        # print(x.shape)
        x = self.base2(x)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.lin1(x)
        # x = self.lin2(x)
        x = self.lin3(x)
        # x = self.out(x)

        return x
