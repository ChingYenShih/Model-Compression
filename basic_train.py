import numpy as np
import os
import sys
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from net import basic_vgg, facenet
import utils
import torch.nn as nn

def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct = 0
    total_loss = 0
    count = 0
    for a_batch, p_batch, n_batch in pbar:
        a_batch = a_batch.to(device).float() / 255.0
        p_batch = p_batch.to(device).float() / 255.0
        n_batch = n_batch.to(device).float() / 255.0

        optimizer.zero_grad()

        a_embedding = net(a_batch)
        p_embedding = net(p_batch)
        n_embedding = net(n_batch)
        loss = criterion(a_embedding, p_embedding , n_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        count += len(a_batch)
        pbar.set_description('Epoch: {}; Avg loss: {:.4f}'.format(epoch + 1, total_loss / count))

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for a_batch, p_batch, n_batch in pbar:
        a_batch = a_batch.to(device).float() / 255.0
        p_batch = p_batch.to(device).float() / 255.0
        n_batch = n_batch.to(device).float() / 255.0

        a_embedding = net(a_batch)
        p_embedding = net(p_batch)
        n_embedding = net(n_batch)
        loss = criterion(a_embedding, p_embedding , n_embedding).detach()

        total_loss += loss.item()

        count += len(x_batch)
        pbar.set_description('Epoch: {}; Avg loss: {:.4f}'.format(epoch + 1, total_loss / count))
    return total_loss/count

class EarlyStop():
    def __init__(self, saved_model_path, patience = 10000, mode = 'max'):
        self.saved_model_path = saved_model_path
        self.patience = patience
        self.mode = mode
        
        self.best = 0 if (self.mode == 'max') else np.Inf
        self.current_patience = 0
    def run(self, value, model):
        condition = (value > self.best) if (self.mode == 'max') else (value <= self.best)
        if(condition):
            self.best = value
            self.current_patience = 0
            with open('{}'.format(self.saved_model_path), 'wb') as f:
                torch.save(model, f)
        else:
            self.current_patience += 1
            if(self.patience == self.current_patience):
                print('Validation mean value: {:.4f}, early stop patience: [{}/{}]'.\
                      format(value, self.current_patience,self.patience))
                return True
        print('Validation mean value: {:.2f}%, early stop[{}/{}], validation max value: {:.2f}%'.\
              format(value, self.current_patience,self.patience, self.best))
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 10, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    parser.add_argument('-m', '--saved_model', default = 'saved_model/basic.model', help = 'Saved model path')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    sys.stdout.write('Loading data.pt...')
    sys.stdout.flush()
    x_train = torch.load('/data/r06942052/preproc_data/train_img.pt')
    x_val = torch.load('/data/r06942052/preproc_data/val_img.pt')
    y_train = torch.load('/data/r06942052/preproc_data/train_id.pt')
    y_val = torch.load('/data/r06942052/preproc_data/val_id.pt')
    sys.stdout.write('Done\n')

    #Triplet selection
    np.random.seed(69)
    bm_train = y_train.numpy()
    bm_val   = y_val.numpy()
    count_train = np.zeros((10177))
    for i in range(10177):
        count_train[i] = np.sum(bm_train == i)
    mapping = np.vstack((count_train.nonzero()[0].reshape(1, -1), np.arange(2360).reshape(1, -1)))
    mapping = mapping.astype('float')

    anchor_train, positive_train, negative_train = [], [], []
    anchor_val, positive_val, negative_val= [], [], []
    for j in range(2360):
        sys.stdout.write('\rTriplet_Selection... : [{:}/2360]'.format(j+1))
        sys.stdout.flush()
        a_train = x_train[y_train == mapping[0, j]]
        sample = np.random.choice(np.arange(len(x_train)-len(a_train)),
                                         len(a_train), replace=False)
        temp = np.arange(len(a_train))
        np.random.shuffle(temp)
        p_train = a_train[temp]
        n_train = x_train[(y_train != mapping[0, j]).nonzero()[sample]]

        a_val = x_val[y_val == mapping[0, j]]
        sample = np.random.choice(np.arange(len(x_val)-len(a_val)), 
                                       args.batch_size-len(a_val), replace=False)
        temp = np.arange(len(a_val))
        np.random.shuffle(temp)
        p_val = a_val[temp]
        n_val = x_val[(y_val != mapping[0, j]).nonzero()[sample]]

        anchor_train.append(a_train)
        positive_train.append(p_train)
        negative_train.append(n_train)
        anchor_val.append(a_val)
        positive_val.append(p_val)
        negative_val.append(n_val)

    anchor_train = torch.cat(anchor_train)
    positive_train = torch.cat(positive_train)
    negative_train = torch.cat(negative_train)
    anchor_val = torch.cat(anchor_val)
    positive_val = torch.cat(positive_val)
    negative_val = torch.cat(negative_val)

    train_set = Data.TensorDataset(anchor_train, positive_train, negative_train)
    val_set = Data.TensorDataset(anchor_val, positive_val, negative_val)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True) 
    val_loader   = Data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False) 

    #criterion = nn.CrossEntropyLoss().to(device)
    #net = basic_vgg().to(device)
    criterion = nn.TripletMarginLoss().to(device)
    net = facenet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5,0.999))

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 10000, mode = 'min')

    for epoch in range(50000):
        train(net, optimizer, criterion, train_loader, epoch)
        val_acc = valid(net, criterion, val_loader)
        
        if(earlystop.run(val_acc, net)):
            break
