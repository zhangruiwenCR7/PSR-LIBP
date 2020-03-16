import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData, TemporalDataFold
from model import RNN
import numpy as np

def Loss(criterionsq, criterion, out1, out2, labelsq, label):
    loss0 = criterionsq(out1[:,0,:], labelsq)
    loss1 = criterionsq(out1[:,1,:], labelsq)
    loss2 = criterionsq(out1[:,2,:], labelsq)
    loss3 = criterionsq(out1[:,3,:], labelsq)
    loss = loss0*0.1 +loss1*0.1 +loss2*0.2+loss3*0.2 + criterion(out2, label)*0.4
    return loss

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    test_data  = TemporalDataFold(split='test')
    test_dataloader  = DataLoader(test_data, batch_size=100, num_workers=4)

    model = RNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterionsq = nn.MSELoss()

    # k_fold, para = 5, [99,99,97,99,98]
    # k_fold, para = 3, [99,99,96]
    # k_fold, para = 3, [95,98,99]
    k_fold, para = 5, [99,99,98,99,98]
    num_epoch = 100
    group = 10
    acc = []
    for g in range(group):
        for k in range(k_fold):
            epoch = para[k]
            model.reset_parameters()
            # model.load_state_dict(torch.load('para/'+str(k)+'_dl_'+str(para[k])+'.pt'))
            model.load_state_dict(torch.load(f'./para{k_fold}/{g}/{k}_dl.pt'))
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
            acc.append(running_acc/num)
            print(f'{g} {k} [test] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

    print(f'mean:{np.mean(acc)} std:{np.std(acc)}')