import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
        self.cnn3 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn4 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4)
        self.line1 = nn.Linear(8,1)

        self.cnn21 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.cnn22 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn23 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, dilation=2)
        self.line2 = nn.Linear(4*64, 3)
        self.relu2 = nn.ReLU()

    def reset_parameters(self):
        self.cnn1.reset_parameters() 
        self.cnn2.reset_parameters() 
        self.cnn3.reset_parameters() 
        self.cnn4.reset_parameters() 
        self.line1.reset_parameters()

        self.cnn21.reset_parameters()
        self.cnn22.reset_parameters()
        self.cnn23.reset_parameters()
        self.line2.reset_parameters()

    def forward(self, x):
        # print(x.size()) [2, 1, t]
        x1 = self.cnn1(x[..., 3:-1])#[6] -> [6]
        x2 = self.cnn2(x[..., 2:-1])#[7] -> [6] 
        x3 = self.cnn3(x[..., 1:-1])
        x4 = self.cnn4(x[..., :-1]) #[10]->
        # print(x1.size(), x2.size(), x4.size())
        x11=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
        x12=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
        x13=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
        x14=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
        # for i in range(x1.size(2)):
            # x11[..., i] = self.line1(x1[..., i])
            # x12[..., i] = self.line1(x2[..., i])
            # x13[..., i] = self.line1(x3[..., i])
            # x14[..., i] = self.line1(x4[..., i])
        # print(x11.size())
        out1 = torch.cat((x11,x12,x13,x14), dim=1) 
        # print(out1.size(), x[..., 4:].size())
        x20 = torch.zeros((out1.size(0), 1, out1.size(1)+1, out1.size(2))).to('cuda:1')
        x20[:, 0, ...] = torch.cat((x[..., 4:], out1), dim=1) #[C, 5, t-4]
        x21 = self.relu2(self.cnn21(x20))
        x22 = self.relu2(self.cnn22(x21))
        x23 = self.relu2(self.cnn23(x22[...,0,:]))
        out2 = self.line2(x23.reshape([x23.size(0), -1]))
        
        # print(x20.size(), x21.size(), x22.size(), x23.size())
        return out1, out2 

