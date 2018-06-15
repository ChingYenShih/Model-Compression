import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.net import basic_vgg, facenet
import utils
import torch.nn as nn

def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct = 0
    total_loss = 0
    count = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device).float() / 255.0, y_batch.to(device)

        optimizer.zero_grad()

        pred = net(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)
        pbar.set_description('Epoch: {}; Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
            format(epoch + 1, total_loss / count, correct / count * 100))

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device).float() / 255.0, y_batch.to(device)
 
        pred = net(x_batch)

        loss = criterion(pred, y_batch).detach()
        
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)

        #pbar.set_description('Validation stage: Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
        #    format(total_loss / count, correct / count * 100))
    acc = correct / count * 100
    return acc
class EarlyStop():
    def __init__(self, saved_model_path, patience = 10000, mode = 'max'):
        self.saved_model_path = saved_model_path
        self.patience = patience
        self.mode = mode
        
        self.best = 0 if (self.mode == 'max') else np.Inf
        self.current_patience = 0
    def run(self, acc, model):
        condition = (acc > self.best) if (self.mode == 'max') else (acc <= self.best)
        if(condition):
            self.best = acc
            self.current_patience = 0
            with open('{}'.format(self.saved_model_path), 'wb') as f:
                torch.save(model, f)
        else:
            self.current_patience += 1
            if(self.patience == self.current_patience):
                print('Validation mean acc: {:.4f}, early stop patience: [{}/{}]'.\
                      format(acc, self.current_patience,self.patience))
                return True
        print('Validation mean acc: {:.2f}%, early stop[{}/{}], validation max acc: {:.2f}%'.\
              format(acc, self.current_patience,self.patience, self.best))
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    parser.add_argument('-m', '--saved_model', default = 'saved_model/basic.model', help = 'Saved model path')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    x_train = torch.load('/mnt/data/r06942052/preproc_data/train_img.pt')
    x_val = torch.load('/mnt/data/r06942052/preproc_data/val_img.pt')
    y_train = torch.load('/mnt/data/r06942052/preproc_data/train_id.pt')
    y_val = torch.load('/mnt/data/r06942052/preproc_data/val_id.pt')

    #mapping new id
    bm_train = y_train.numpy()
    bm_val   = y_val.numpy()
    count_train = np.zeros((10177))
    for i in range(10177):
        count_train[i] = np.sum(bm_train == i)
    mapping = np.vstack((count_train.nonzero()[0].reshape(1, -1), np.arange(2360).reshape(1, -1)))
    mapping = mapping.astype('float')
    for j in range(2360):
        y_train[y_train == mapping[0, j]] = mapping[1, j]
        y_val[    y_val == mapping[0, j]] = mapping[1, j]

    train_set = Data.TensorDataset(x_train, y_train)
    val_set = Data.TensorDataset(x_val, y_val)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True) 
    val_loader   = Data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False) 

    criterion = nn.CrossEntropyLoss().to(device)
    #net = basic_vgg().to(device)
    net = facenet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-5)
    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5,0.999))

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 10000, mode = 'max')

    for epoch in range(50000):
        train(net, optimizer, criterion, train_loader, epoch)
        val_acc = valid(net, criterion, val_loader)
        
        if(earlystop.run(val_acc, net)):
            break
