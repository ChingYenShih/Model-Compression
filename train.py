import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.vgg11_bn_mobile import vgg11_bn_MobileNet 
from model.vgg11_bn_fire import vgg11_bn_fire
from model.vgg11_bn_depth_fire import vgg11_bn_depth_fire
from model.vgg11_bn import vgg11_bn
import utils
import torch.nn as nn
import scipy.misc

def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct, total_loss, count = 0,0,0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        pred, _ = net(x_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        loss.backward()
        optimizer.step()

        count += len(x_batch)
        pbar.set_description('Epoch: {}; Avg Class loss: {:.4f}; Avg acc: {:.2f}%'.\
            format(epoch + 1, total_loss/count*len(x_batch) , correct / count * 100))

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
 
        pred,_ = net(x_batch)

        loss = criterion(pred, y_batch)
        
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
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-3, help = 'Learning rate')
    #parser.add_argument('-log', '--log', default = 'No_discription', help = 'model')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    print("Loading data")
    mapping = np.load(os.path.join('preproc_data','map.npz'))['map'].reshape(1)[0]
    x_train, y_train = utils.read_preproc_data(os.path.join('preproc_data', 'train.npz'))
    x_val, y_val = utils.read_preproc_data(os.path.join('preproc_data', 'val.npz'))

    train_loader = utils.get_data_loader(x_train, y_train, mapping, data_aug = True, batch_size = args.batch_size, shuffle = True)
    val_loader = utils.get_data_loader(x_val, y_val, mapping, data_aug = False, batch_size = args.batch_size, shuffle = False)

    print("Initialize model and loss")
    criterion = nn.CrossEntropyLoss()
    #net = basic_vgg()
    net = vgg11_bn_MobileNet()
    #net = vgg11_bn_fire()
    #net = vgg11_bn_depth_fire()
    #net = vgg11_bn()
    #net = torch.load('saved_model/vgg_shrink/vgg2')
    
    print(net)
    net.to(device)
    #opt_class = torch.optim.Adam(net.parameters(), lr = args.learning_rate)
    opt_class = torch.optim.Adam(net.parameters(), lr = args.learning_rate, betas = (0.5,0.999), weight_decay = 1e-6)
 

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 100000, mode = 'max')
    
    for epoch in range(100000):
        train(net, opt_class, criterion, train_loader, epoch)
        val_acc = valid(net, criterion, val_loader)
        if(earlystop.run(val_acc, net)):
            break
    