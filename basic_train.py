import numpy as np
import os
import sys
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
    for batch_idx, (a_batch, p_batch, n_batch) in enumerate(pbar):
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

        count += 1
        pbar.set_description('Epoch: {}; loss: {:.4f}; Avg loss: {:.4f}'.format(epoch + 1, loss.item(), total_loss / count))
        if boardX:
            writer.add_scalar('Triplet Train Loss', 
            loss.item(),
            epoch*len(pbar)+batch_idx)

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for batch_idx, (a_batch, p_batch, n_batch) in enumerate(pbar):
        a_batch = a_batch.to(device).float() / 255.0
        p_batch = p_batch.to(device).float() / 255.0
        n_batch = n_batch.to(device).float() / 255.0

        a_embedding = net(a_batch)
        p_embedding = net(p_batch)
        n_embedding = net(n_batch)
        loss = criterion(a_embedding, p_embedding , n_embedding).detach()

        total_loss += loss.item()

        count += 1 
        if boardX:
            writer.add_scalar('Triplet valid Loss', 
            loss.item(),
            epoch*len(pbar)+batch_idx)
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
        print('Validation mean value: {:.2f}, early stop[{}/{}], validation max value: {:.2f}'.\
              format(value, self.current_patience,self.patience, self.best))
        return False

def update_loader(x, y, mapping, batch_size):
    anchor, positive, negative= [], [], []
    for j in range(2360):
        sys.stdout.write('\rTriplet_Selection... : [{:}/2360]'.format(j+1))
        sys.stdout.flush()
        a= x[y== mapping[0, j]]
        sample = np.random.choice(np.arange(len(x)-len(a)),
                                         len(a), replace=False)
        temp = np.arange(len(a))
        np.random.shuffle(temp)
        p= a[temp]
        n= x[((y!= mapping[0, j]).nonzero()[sample]).view(-1)]

        anchor.append(a)
        positive.append(p)
        negative.append(n)

    anchor= torch.cat(anchor)
    positive= torch.cat(positive)
    negative= torch.cat(negative)

    triplet = Data.TensorDataset(anchor, positive, negative)
    return Data.DataLoader(dataset=triplet, batch_size=batch_size, shuffle=True) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 10, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    parser.add_argument('-m', '--saved_model', default = 'saved_model/basic.model', help = 'Saved model path')
    parser.add_argument('-tb', '--tensorboard', default = 'record', help = 'record training info')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))
    presave_loader = False
    boardX = True
    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/'+args.tensorboard)

    sys.stdout.write('Loading data.pt...')
    sys.stdout.flush()
    x_train = torch.load('/mnt/data/r06942052/preproc_data/train_crop_img.pt')
    x_val = torch.load('/mnt/data/r06942052/preproc_data/val_crop_img.pt')
    y_train = torch.load('/mnt/data/r06942052/preproc_data/train_id.pt')
    y_val = torch.load('/mnt/data/r06942052/preproc_data/val_id.pt')
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

    #torch.save(train_loader, '/mnt/data/r06942052/triplet_train_crop_loader.pt')
    #torch.save(val_loader, '/mnt/data/r06942052/triplet_val_crop_loader.pt')

    #criterion = nn.CrossEntropyLoss().to(device)
    #net = basic_vgg().to(device)

    criterion = nn.TripletMarginLoss(margin = 1).to(device)
    net = facenet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5,0.999))

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 5, mode = 'min')

    val_loader = update_loader(x_val, y_val, mapping, args.batch_size)
    for epoch in range(50):
        train_loader = update_loader(x_train, y_train, mapping, args.batch_size)
        train(net, optimizer, criterion, train_loader, epoch)
        val_loss = valid(net, criterion, val_loader)
        
        if(earlystop.run(val_loss, net)):
            break
