import os
import time
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData, TemporalDataFold
from model import RNN

def Loss(criterionsq, criterion, out1, out2, labelsq, label):
    loss0 = criterionsq(out1[:,0,:], labelsq)
    loss1 = criterionsq(out1[:,1,:], labelsq)
    loss2 = criterionsq(out1[:,2,:], labelsq)
    loss3 = criterionsq(out1[:,3,:], labelsq)
    loss = loss0*0.1 +loss1*0.1 +loss2*0.2+loss3*0.2 + criterion(out2, label)*0.4
    # loss = criterion(out2, label)
    return loss

if __name__ == '__main__':
    for group in range(1):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device being used:", device)

        model = RNN()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        criterionsq = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epoch = 1000
        k_fold = 7
        
        print(' Begin training.\n')
        for k in range(k_fold):
            print(f'----------------------- {k}/{k_fold} --------------------------')
            train_data = TemporalDataFold(split='train', K_k=(k_fold, k))
            test_data  = TemporalDataFold(split='test',  K_k=(k_fold, k))

            train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
            test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)                  
            model.reset_parameters()
            valid_acc = 0
            for epoch in range(num_epoch):
                model.train()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(train_dataloader):
                    data = Variable(data).to(device)
                    label = Variable(label).to(device)
                    labelsq = Variable(labelsq).to(device).squeeze()
                    
                    optimizer.zero_grad()
                    out1, out2 = model(data)
                    loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
                    running_loss += loss.data.item() * label.size(0)
                    _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                    num_correct = (pred == label).sum()
                    running_acc += num_correct.data.item()
                    loss.backward()
                    optimizer.step()
                num = len(train_data)
                print(f'{time.time()} {group} {k} [Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}')

                model.eval()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(test_dataloader):
                    data = Variable(data).to(device)
                    label = Variable(label).to(device)
                    labelsq = Variable(labelsq).to(device).squeeze()
                    with torch.no_grad():
                        out1, out2 = model(data)
                    loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
                    running_loss += loss.data.item() * label.size(0)
                    _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                    num_correct = (pred == label).sum()
                    running_acc += num_correct.data.item() 
                num = len(test_data)
                print(f'{time.time()} {group} {k} [valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

                if running_acc/num >= valid_acc:
                    valid_acc = running_acc/num
                    os.makedirs(f'./para{k_fold}/{group}', exist_ok=True)
                    torch.save(model.state_dict(), f'./para{k_fold}/{group}/{k}_dl.pt')

    #'''