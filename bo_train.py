import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.net import basic_vgg
from model.net import MobileNet
from model.vgg11_bn_mobile import vgg11_bn_MobileNet 
import utils
import torch.nn as nn

def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct = 0
    total_loss = 0
    count = 0
    classL = criterion[0]
    tripletL = criterion[1]
    for batch in pbar:
        optimizer.zero_grad()
        b = batch
        img, imp, imn, idp, idn = b[0].to(device), b[1].to(device), b[2].to(device), \
                                  b[3].to(device), b[4].to(device)

        (pred1,feat1), (pred2,feat2), (pred3,feat3) = net(img), net(imp), net(imn)
        c_loss = classL(pred1, idp) + classL(pred2, idp) + classL(pred3, idn)
        trip_loss = tripletL(feat1, feat2, feat3)
        loss = 0.01*c_loss + trip_loss
        total_loss += loss.item()
        _, pred_class = torch.max(pred1, 1)
        correct += (pred_class == idp).sum().item()

        loss.backward()
        optimizer.step()

        count += len(img)
        pbar.set_description('Epoch: {}; Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
            format(epoch + 1, total_loss / count, correct / count * 100))

def train_feature(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct, total_loss, count = 0,0,0
    for batch in pbar:
        optimizer.zero_grad()
        b = batch
        img, imp, imn, idp, idn = b[0].to(device), b[1].to(device), b[2].to(device), \
                                  b[3].to(device), b[4].to(device)
        (pred1,feat1), (pred2,feat2), (pred3,feat3) = net(img), net(imp), net(imn)
        loss = criterion(feat1, feat2, feat3)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        count += 1
        pbar.set_description('Epoch: {}; Avg Triplet loss: {:.4f}; '.\
            format(epoch + 1, total_loss / count ))

def train_classifier(net, optimizer, criterion, loader, epoch):
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
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    print("Loading data")
    mapping = np.load(os.path.join('preproc_data','map.npz'))['map'].reshape(1)[0]
    x_train, y_train = utils.read_preproc_data(os.path.join('preproc_data', 'train.npz'))
    x_val, y_val = utils.read_preproc_data(os.path.join('preproc_data', 'val.npz'))
    #train_loader = utils.get_new_loader(x_train, y_train, mapping, batch_size = args.batch_size, shuffle = True)
    #train_feat_loader = utils.get_new_loader(x_train, y_train, mapping, batch_size = args.batch_size, shuffle = True)
    train_class_loader = utils.get_data_loader(x_train, y_train, mapping, batch_size = args.batch_size, shuffle = True)
    val_loader = utils.get_data_loader(x_val, y_val, mapping,batch_size = args.batch_size, shuffle = False)

    print("Initialize model and loss")
    criterion = (nn.CrossEntropyLoss(),nn.TripletMarginLoss())
    #net = basic_vgg()
    net = MobileNet()
    net = vgg11_bn_MobileNet()
    """
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    """
    net.to(device)
    print(net)
    #opt_feat = torch.optim.SGD(net.features.parameters(), lr = 5e-3)
    opt_class = torch.optim.Adam(net.parameters(), lr = 1e-3)
    """
    opt_class = torch.optim.Adam([
                        {'params': net.features.parameters(), 'lr':5e-6},
                        {'params': net.classifier.parameters(), 'lr': 5e-3}
                    ], lr=1e-3)
    """
    #optimizer = [opt_feat, opt_class]

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 50, mode = 'max')

    """
    for epoch in range(100):
        #train(net, optimizer, criterion, train_loader, epoch)
        train_feature(net, opt_feat, criterion[1], train_feat_loader, epoch)
    """
    for epoch in range(500):
        train_classifier(net, opt_class, criterion[0], train_class_loader, epoch)
        val_acc = valid(net, criterion[0], val_loader)
        if(earlystop.run(val_acc, net)):
            break
