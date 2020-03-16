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
    def __init__(self, split='train', K_k=(3,0), root='/home/zhangruiwen/01research/02toyota/03code/mywork/data/'):
        if split=='test':
            self.fpath = os.path.join(root, f'{K_k[0]}-fold/{K_k[1]}')
            self.filelist = [os.path.join(self.fpath, f) for f in os.listdir(self.fpath)]
        else:
            self.filelist = []
            for k in range(K_k[0]):
                if k != K_k[1]:
                    self.fpath = os.path.join(root, f'{K_k[0]}-fold/{k}')
                    self.filelist += [os.path.join(self.fpath, f) for f in os.listdir(self.fpath)]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # idata = np.loadtxt(os.path.join(self.fpath, self.filelist[index]), dtype=np.float32)
        idata = np.loadtxt(self.filelist[index], dtype=np.float32)
        ind = np.random.randint(idata.shape[0]-23)
        
        data = np.array([idata[ind:ind+24, 1]])
        if self.filelist[index].find('along') != -1: label = 0
        elif self.filelist[index].find('l2r') != -1: label = 1
        elif self.filelist[index].find('r2l') != -1: label = 2
        else: print('ERROR: No label.')
        
        return torch.from_numpy(data), torch.from_numpy(np.array(label)), torch.from_numpy(data[:, 4:])


if __name__ == '__main__':
    train_dataloader = DataLoader(TemporalDataFold(split='train'), batch_size=1, shuffle=True, num_workers=4)
    print(len(TemporalDataFold(split='test')))
    # for inputs, labels, l in train_dataloader:
    #     print(labels, inputs.size())