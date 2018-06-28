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
import copy
from collections import namedtuple, OrderedDict
import quant

def quantize(model_raw, quant_method = 'log'):
    bn_bits = 4
    param_bits = 4
    if param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            else:
                bits = param_bits

            if quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=0.0)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
        model_raw.load_state_dict(state_dict_quant)
    return model_raw

def train(net, optimizer, criterion, loader, epoch):
    pbar = tqdm(iter(loader))
    net.train()
    correct, total_loss, count = 0,0,0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        q_net = quantize(copy.deepcopy(net), quant_method = 'log')

        pred, _ = q_net(x_batch)
        #pred, _ = net(x_batch)
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
    q_correct = 0
    total_loss = 0
    count = 0
    q_net = quantize(copy.deepcopy(net), quant_method = 'log')

    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
 

        q_pred,_ = q_net(x_batch)
        pred,_ = net(x_batch)

        loss = criterion(pred, y_batch)
        
        total_loss += loss.item()
        _, q_pred_class = torch.max(q_pred, 1)
        _, pred_class = torch.max(pred, 1)
        q_correct += (q_pred_class == y_batch).sum().item()
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)

        pbar.set_description('Validation stage: Avg loss: {:.4f}; Avg acc: {:.2f}%; Q Avg acc: {:.2f}'.\
            format(total_loss / count, correct / count * 100, q_correct/count*100))
    acc = q_correct/count*100
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
 
    train_loader = utils.get_data_loader(x_val, y_val, mapping, data_aug = True, batch_size = args.batch_size, shuffle = True)
    val_loader = utils.get_data_loader(x_val, y_val, mapping, data_aug = False, batch_size = args.batch_size, shuffle = False)

    print("Initialize model and loss")
    criterion = nn.CrossEntropyLoss()
    #net = basic_vgg()
    #net = vgg11_bn_MobileNet()
    #net = vgg11_bn_fire()
    net = vgg11_bn_depth_fire()
    #net = torch.load('saved_model/vgg11_bn.model')
    print(net)
    net.to(device)
    opt_class = torch.optim.Adam(net.parameters(), lr = args.learning_rate)
    #opt_class = torch.optim.Adam(net.parameters(), lr = args.learning_rate, betas = (0.5,0.999), weight_decay = 1e-6)
 

    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = 100000, mode = 'max')
    

    for epoch in range(100000):
        train(net, opt_class, criterion, train_loader, epoch)
        val_acc = valid(net, criterion, val_loader)
        #if(earlystop.run(val_acc, net)):
        #    break
        #if(val_acc > 70):
        #    if(epoch % 15 == 0):
        #        adjust_learning_rate(opt_class, decay_rate=.5)
    