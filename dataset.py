import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TemporalData(Dataset):
    def __init__(self, split='train', root='/home/zhangruiwen/01research/02toyota/03code/mywork/direction/result_txt'):
        self.fpath = os.path.join(root, split+'_txt')
        self.filelist = os.listdir(self.fpath)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        idata = np.loadtxt(os.path.join(self.fpath, self.filelist[index]), dtype=np.float32)
        # print(idata.shape, type(idata[0,0]))
        ind = np.random.randint(idata.shape[0]-23)
        
        data = np.array([idata[ind:ind+24, 1]])
        if self.filelist[index].find('along') != -1: label = 0
        elif self.filelist[index].find('l2r') != -1: label = 1
        elif self.filelist[index].find('r2l') != -1: label = 2
        else: print('ERROR: No label.')
        
        return torch.from_numpy(data), torch.from_numpy(np.array(label)), torch.from_numpy(data[:, 4:])

class TemporalDataFold(Dataset):
    def __init__(self, split='train', root='/home/zhangruiwen/01research/02toyota/03code/mywork/data'):
        self.fpath = os.path.join(root, split+'_txt')
        self.filelist = os.listdir(self.fpath)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        idata = np.loadtxt(os.path.join(self.fpath, self.filelist[index]), dtype=np.float32)
        ind = np.random.randint(idata.shape[0]-23)
        
        data = np.array([idata[ind:ind+24, 1]])
        if self.filelist[index].find('along') != -1: label = 0
        elif self.filelist[index].find('l2r') != -1: label = 1
        elif self.filelist[index].find('r2l') != -1: label = 2
        else: print('ERROR: No label.')
        
        return torch.from_numpy(data), torch.from_numpy(np.array(label)), torch.from_numpy(data[:, 4:])


if __name__ == '__main__':
    train_dataloader = DataLoader(TemporalData(split='train'), batch_size=1, shuffle=True, num_workers=4)
    for inputs, labels in train_dataloader:
        print(labels, inputs.size())