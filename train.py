import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData, TemporalDataFold
from model import RNN

'''
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


train_data = TemporalData(split='valid')
valid_data = TemporalData(split='train')
test_data  = TemporalData(split='test')

train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_data, batch_size=50, shuffle=True, num_workers=4)
test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)

model = RNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epoch = 5000

print(' Begin training.\n')

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label) in enumerate(train_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()
        # print(running_loss, running_acc, pred, len(train_data))
        loss.backward()
        optimizer.step()
    print(f'[Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(train_data))} Acc: {running_acc/(len(train_data))}')

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label) in enumerate(test_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        with torch.no_grad():
            out = model(data)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()   
    print(f'[test ] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(test_data))} Acc: {running_acc/(len(test_data))}')

    running_loss = 0.0
    running_acc = 0.0
    for i, (data, label) in enumerate(valid_dataloader):
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        with torch.no_grad():
            out = model(data)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()   
    print(f'[valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/(len(valid_data))} Acc: {running_acc/(len(valid_data))}\n')
'''

if __name__ == '__main__':
    for group in range(1):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device being used:", device)

        train_data = TemporalDataFold(split='train')
        test_data  = TemporalDataFold(split='test')

        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
        test_dataloader  = DataLoader(test_data, batch_size=50, num_workers=4)

        model = RNN()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        criterionsq = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        num_epoch = 1000
        k_fold = 3

        print(' Begin training.\n')
        for k in range(k_fold):
            print(f'----------------------- {k}/{k_fold} --------------------------')
            model.reset_parameters()
            valid_acc = 0
            for epoch in range(num_epoch):
                valid_tmp = []
                model.train()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(train_dataloader):
                    data = Variable(data).to(device)
                    label = Variable(label).to(device)
                    labelsq = Variable(labelsq).to(device).squeeze(0)
                    if i>=k*22 and i<(k+1)*22+1:
                    # if i>=k*15 and i<(k+1)*15+2:
                        valid_tmp.append((data, label, labelsq))
                    else:
                        optimizer.zero_grad()
                        out1 = model(data)
                        loss = criterion(out1, label)
                        # loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
                        running_loss += loss.data.item() * label.size(0)
                        # _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                        _, pred = torch.max(out1, 1)
                        num_correct = (pred == label).sum()
                        running_acc += num_correct.data.item()
                        loss.backward()
                        optimizer.step()
                        # print(labelsq.size(), out1.size())
                # num = len(train_data)-valid_tmp[0][0].size(0)
                num = len(train_data)-len(valid_tmp)
                print(f'{time.time()} {group} {k} [Train] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}')

                model.eval()
                running_loss, running_acc = 0.0, 0.0
                for i, (data, label, labelsq) in enumerate(valid_tmp):
                    with torch.no_grad():
                        out1 = model(data)
                        loss = criterion(out1, label)
                    # loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
            
                    running_loss += loss.data.item() * label.size(0)
                    # _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                    _, pred = torch.max(out1, 1)
                    num_correct = (pred == label).sum()
                    running_acc += num_correct.data.item() 
                # num = valid_tmp[0][0].size(0)
                num = len(valid_tmp)
                print(f'{time.time()} {group} {k} [valid] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

                if running_acc/num >= valid_acc:
                    valid_acc = running_acc/num
                    os.makedirs(f'./para{k_fold}/{group}', exist_ok=True)
                    torch.save(model.state_dict(), f'./para{k_fold}/{group}/{k}_dl.pt')

#'''