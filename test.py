import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TemporalData, TemporalDataFold
from model import RNN
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    test_data  = TemporalDataFold(split='test')
    test_dataloader  = DataLoader(test_data, batch_size=100, num_workers=4)

    model = RNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterionsq = nn.MSELoss()

    k_fold, para = 3, [199,197,198,199,199]
    # k_fold, para = 3, [196,195,196]
    num_epoch = 200

    acc = []
    for g in range(10):
        for k in range(k_fold):
            epoch = para[k]
            model.reset_parameters()
            model.load_state_dict(torch.load(f'./para{k_fold}/{g}/{k}_dl.pt'))
            model.eval()
            running_loss, running_acc = 0.0, 0.0
            for i, (data, label, labelsq) in enumerate(test_dataloader):
                data = Variable(data).to(device)
                label = Variable(label).to(device)
                labelsq = Variable(labelsq).to(device).squeeze()
                with torch.no_grad():
                    out1 = model(data)
                # loss = Loss(criterionsq, criterion, out1, out2, labelsq, label)
                loss = criterion(out1, label)
                running_loss += loss.data.item() * label.size(0)
                _, pred = torch.max(out1, 1)
                # _, pred = torch.max(F.log_softmax(out2, dim=1), 1)
                num_correct = (pred == label).sum()
                running_acc += num_correct.data.item() 
            num = len(test_data)
            acc.append(running_acc/num)
            print(f'{g} {k} [test] Epoch: {epoch}/{num_epoch} Loss: {running_loss/num} Acc: {running_acc/num}\n')

    print(f'mean:{np.mean(acc)} std:{np.std(acc)}')