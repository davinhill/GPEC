from torch.optim import SGD, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import sys 
from tqdm import tqdm 
import torch.nn as nn 
import torch 

sys.path.append('./')
from GPEC.utils import * # utility functions
dname = utils_io.get_filedir(__file__)
sys.path.append(dname)
from nn_datasets import load_mnist

import numpy as np
def supervised_acc(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_bar = tqdm(data_loader)
    seen = 0 
    correct = 0 
    with torch.no_grad():
        for data, target in test_bar:
            data, target = data.to(device), target.to(device) 
            seen = seen + target.shape[0]
            logits   = model(data)
            preds = torch.argmax(logits, dim=1)

            correct += torch.sum(preds == target.squeeze())

            test_bar.set_description('Acc@1:{:.2f}'
                                        .format(correct / seen * 100))
        
    return (correct / seen * 100).detach().cpu().numpy()

def train_network(model, epochs, train_loader, val_loader, lr, optimizer, milestones, gamma, savedir, l2_reg = 0.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay = l2_reg)
    else:
        optimizer = torch.optim.SGD(model.parameters(),  lr=lr, weight_decay = l2_reg)
    #scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scheduler = ReduceLROnPlateau(optimizer, factor=gamma, patience = 5, mode = 'min')
  
    for epoch in range(0, epochs):
        model.train() 
        train_bar = tqdm(train_loader)
        total_loss   = 0.0
        batches_seen = 0 
        correct = 0
        seen = 0
        #Train For One Epoch 
        for items in train_bar:
            data, labels  = items[0].to(device), items[1].to(device)
            #im = data[0,:]
            #plt.figure(); plt.imshow(np.reshape(im.detach().cpu().numpy(), (28,28))); plt.savefig('ab.png')
            logits        = model(data)
            loss_function = nn.CrossEntropyLoss().to(device) 
            loss          = loss_function(logits,labels.long())
            loss          = loss / len(labels)
            
            preds         = torch.argmax(logits, dim = 1)
            correct       += torch.sum(preds == labels.squeeze())
            seen          += len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.detach().cpu().numpy() 
            batches_seen += 1 

            batch_accy = correct/seen
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs, total_loss / batches_seen, batch_accy))

        model.eval() 
        #Eval on Val
        print("Evaluating Model")
        val_accy = supervised_acc(model, val_loader)
        print(val_accy)    

        #Save Model 
        print("Saving Current Model, Overwriting Previous Epochs Model")
        torch.save(model.state_dict(), '{}/model.pth'.format(savedir)) 

        scheduler.step(total_loss) 

    return model, batch_accy.item()*100, val_accy




# _, _, _, tr_loader, _, _, _, val_loader, _, _, _, _ = load_mnist()
# print("hi")
