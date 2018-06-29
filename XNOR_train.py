import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.XNOR.XNOR_vgg11_bn import XNOR_vgg11_bn
import model.XNOR.util as util
import utils
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    #for p in (net.parameters()):
    #    print(p.requires_grad)
    net.train()
    #for p in (net.parameters()):
    #    print(p.requires_grad)
    correct, total_loss, count = 0,0,0
    for x_batch, y_batch in pbar:

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        bin_op.binarization()
        pred, _ = net(x_batch)
        
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        loss.backward()
        bin_op.restore()
        bin_op.updateBinaryGradWeight()


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
    bin_op.binarization()

    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #x_batch = x_batch.cpu().numpy()
        #print(type(x_batch))
        #cv2.imwrite('s.png',np.transpose(x_batch[0]*255,(1,2,0)))
        #plt.show()
        pred,_ = net(x_batch)

        loss = criterion(pred, y_batch)
        
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)

        #pbar.set_description('Validation stage: Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
        #    format(total_loss / count, correct / count * 100))
    bin_op.restore()

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


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

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

    net = XNOR_vgg11_bn()
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = float(m.weight.data[0].nelement())
            m.weight.data = m.weight.data.normal_(0, 1.0/c)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = m.weight.data.zero_().add(1.0)
    
    global bin_op
    bin_op = util.BinOp(net)


    print(net)
    net.to(device)
    opt_class = torch.optim.Adam(net.parameters(), lr = args.learning_rate, betas = (0.5,0.999), weight_decay = 1e-6)
 

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 100000, mode = 'max')
    
    for epoch in range(100000):
        train(net, opt_class, criterion, train_loader, epoch)
        if(epoch == 0):
            for p in (net.parameters()):
                print(p)
        val_acc = valid(net, criterion, val_loader)
        if(earlystop.run(val_acc, net)):
            break
        if(val_acc > 70):
            if(epoch % 15 == 0):
                adjust_learning_rate(opt_class, decay_rate=.5)
    