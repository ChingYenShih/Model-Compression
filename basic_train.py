import numpy as np
import os
import sys
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.net import *
from model.vgg11_bn_depth_fire import *
from model.vgg11_bn_fire import *
import utils
import torch.nn as nn

def train_triplet(net, optimizer, criterion, criterion_classifier, loader, epoch, stage):
    pbar = tqdm(iter(loader))
    net.train()
    correct = 0.0
    total_t_loss = 0
    total_c_loss = 0
    for batch_idx, (a_batch, p_batch, n_batch, p_id, n_id) in enumerate(pbar):
        a_batch = (a_batch.to(device).float() / 255.0)*2-1
        p_batch = (p_batch.to(device).float() / 255.0)*2-1
        n_batch = (n_batch.to(device).float() / 255.0)*2-1
        p_id = p_id.to(device).long()
        n_id = n_id.to(device).long()

        optimizer.zero_grad()

        a_embedding, p_embedding, n_embedding = net(a_batch), net(p_batch), net(n_batch)

        a_class = net.forward_classifier(a_batch)
        p_class = net.forward_classifier(p_batch)
        n_class = net.forward_classifier(n_batch)

        predicted_labels = torch.cat([a_class, p_class, n_class])
        true_labels      = torch.cat([p_id, p_id, n_id])

        triplet_loss = criterion(a_embedding, p_embedding , n_embedding)
        cross_loss =  criterion_classifier(predicted_labels, true_labels)
        loss = triplet_loss + cross_loss
        loss.backward()
        optimizer.step()

        total_t_loss += triplet_loss.item()
        total_c_loss += cross_loss.item()

        c = (torch.sum(torch.argmax(a_class, 1) == p_id) + 
             torch.sum(torch.argmax(p_class, 1) == p_id) + 
             torch.sum(torch.argmax(n_class, 1) == n_id)).float()
        correct += c
        pbar.set_description('Epoch: {}; Avg triplet_loss: {:.4f}; Avg cross_loss: {:.4f}; Avg acc: {:.2f}'.format(epoch + 1, 
                total_t_loss / (batch_idx+1), total_c_loss / (batch_idx+1), correct/3/(batch_idx*loader.batch_size+len(p_id))))
        if boardX:
            writer.add_scalar('Train Triplet Loss', triplet_loss.item(), batch_idx+1+stage)
            writer.add_scalar('Train Cross Loss', cross_loss.item(), batch_idx+1+stage)
            writer.add_scalar('Train Acc', correct/3/(batch_idx*loader.batch_size+len(p_id)), batch_idx+1+stage)
    return len(pbar)+stage

def valid_triplet(net, criterion, criterion_classifier, loader, epoch, stage):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0.0
    total_loss = 0
    for batch_idx, (a_batch, p_batch, n_batch, p_id, n_id) in enumerate(pbar):
        a_batch = (a_batch.to(device).float() / 255.0)*2-1
        p_batch = (p_batch.to(device).float() / 255.0)*2-1
        n_batch = (n_batch.to(device).float() / 255.0)*2-1
        p_id = p_id.to(device).long()
        n_id = n_id.to(device).long()

        a_embedding = net(a_batch)
        p_embedding = net(p_batch)
        n_embedding = net(n_batch)
        a_class = net.forward_classifier(a_batch)
        p_class = net.forward_classifier(p_batch)
        n_class = net.forward_classifier(n_batch)
        loss = criterion(a_embedding, p_embedding , n_embedding).detach()
        loss += (criterion_classifier(a_class, p_id) + 
                 criterion_classifier(p_class, p_id) + 
                 criterion_classifier(n_class, n_id)).detach() / 3.

        total_loss += loss.item()

        c = (torch.sum(torch.argmax(a_class, 1) == p_id) + 
             torch.sum(torch.argmax(p_class, 1) == p_id) + 
             torch.sum(torch.argmax(n_class, 1) == n_id)).float()
        correct += c
        pbar.set_description('Epoch: {}; Avg loss: {:.4f}; Avg acc: {:.2f}'.format(epoch + 1, 
                    total_loss / (batch_idx+1), correct/3/(batch_idx*loader.batch_size+len(p_id))))
    if boardX:
        writer.add_scalar('Valid Loss', total_loss / (batch_idx+1), len(pbar)+stage)
        writer.add_scalar('Valid Acc', correct/3/(batch_idx*loader.batch_size+len(p_id)), len(pbar)+stage)
    return correct/3/(batch_idx*loader.batch_size+len(p_id)), len(pbar)+stage

def train_cross(net, optimizer, criterion_classifier, loader, epoch, stage):
    pbar = tqdm(iter(loader))
    net.train()
    correct = 0.0
    total_c_loss = 0
    for batch_idx, (img, tag) in enumerate(pbar):
        img = img.to(device)
        tag = tag.to(device)

        optimizer.zero_grad()

        predicted_tag, _= net(img)

        cross_loss =  criterion_classifier(predicted_tag, tag)
        loss = cross_loss
        loss.backward()
        optimizer.step()

        total_c_loss += cross_loss.item()

        c = torch.sum(torch.argmax(predicted_tag, 1) == tag).float()
        correct += c
        pbar.set_description('Epoch: {}; Avg cross_loss: {:.4f}; Avg acc: {:.2f}'.format(epoch + 1, 
                total_c_loss / (batch_idx+1), correct/(batch_idx*loader.batch_size+len(tag))))
        if boardX:
            writer.add_scalar('Train Cross Loss', cross_loss.item(), batch_idx+1+stage)
            writer.add_scalar('Train Acc', correct/(batch_idx*loader.batch_size+len(tag)), batch_idx+1+stage)
    return correct/(batch_idx*loader.batch_size+len(tag)), len(pbar)+stage

def valid_cross(net, criterion_classifier, loader, epoch, stage):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0.0
    total_loss = 0
    for batch_idx, (img, tag) in enumerate(pbar):
        img = img.to(device)
        tag = tag.to(device)

        optimizer.zero_grad()

        predicted_tag, _= net(img)

        cross_loss =  criterion_classifier(predicted_tag, tag).detach()
        loss = cross_loss


        total_loss += loss.item()
        c = torch.sum(torch.argmax(predicted_tag, 1) == tag).float()
        correct += c

        pbar.set_description('Epoch: {}; Avg loss: {:.4f}; Avg acc: {:.2f}'.format(epoch + 1, 
                    total_loss / (batch_idx+1), correct/(batch_idx*loader.batch_size+len(tag))))
    if boardX:
        writer.add_scalar('Valid Loss', total_loss / (batch_idx+1), len(pbar)+stage)
        writer.add_scalar('Valid Acc', correct/(batch_idx*loader.batch_size+len(tag)), len(pbar)+stage)
    return correct/(batch_idx*loader.batch_size+len(tag)), len(pbar)+stage

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
                self.current_patience = 0
                self.best = 0 
                return True, self.current_patience
        print('Validation mean value: {:.2f}, early stop[{}/{}], validation max value: {:.2f}'.\
              format(value, self.current_patience,self.patience, self.best))
        return False, self.current_patience

def generate_triplet_loader(x, y, batch_size, n_people):
    anchor, positive, negative= [], [], []
    positive_id, negative_id = [], []
    for j in range(n_people):
        sys.stdout.write('\rTriplet_Selection... : [{}/{}]'.format(j+1, n_people))
        sys.stdout.flush()

        a = x[(y == j).nonzero().view(-1)]
        n_pool = ((y != j) * (y < n_people)).nonzero()
        sample = np.random.choice(np.arange(len(n_pool)), len(a), replace=False)
        p= a[torch.randperm(len(a))]
        p_id = torch.Tensor([j]).expand(len(p))
        n= x[(n_pool[sample]).view(-1)]
        n_id = y[(n_pool[sample]).view(-1)]

        anchor.append(a)
        positive.append(p)
        negative.append(n)
        positive_id.append(p_id)
        negative_id.append(n_id)

    anchor= torch.cat(anchor)
    positive= torch.cat(positive)
    negative= torch.cat(negative)
    positive_id = torch.cat(positive_id)
    negative_id = torch.cat(negative_id)

    triplet = Data.TensorDataset(anchor, positive, negative, positive_id, negative_id)
    return Data.DataLoader(dataset=triplet, batch_size=batch_size, shuffle=True) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 10, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    parser.add_argument('-m', '--saved_model', default = 'saved_model/basic.model', help = 'Saved model path')
    parser.add_argument('-tb', '--tensorboard', default = 'record', help = 'record training info')
    parser.add_argument('-np', '--n_people', type = int, default = 2360, help = 'number of people')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))
    presave_loader = False
    boardX = True 
    if boardX:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('runs/'+args.tensorboard)

    sys.stdout.write('Loading data...')
    sys.stdout.flush()
    mapping = np.load(os.path.join('preproc_data','map.npz'))['map'].reshape(1)[0]
    x_train, y_train = utils.read_preproc_data(os.path.join('preproc_data', 'train.npz'))
    x_val, y_val = utils.read_preproc_data(os.path.join('preproc_data', 'val.npz'))
    sys.stdout.write('Done\n')

    train_class_loader = utils.get_data_loader(x_train, y_train, mapping, batch_size = args.batch_size, 
                                               shuffle = True, dataAUG=True)
    val_loader = utils.get_data_loader(x_val, y_val, mapping, batch_size = args.batch_size,
                                               shuffle = False, dataAUG=False)

    criterion = nn.TripletMarginLoss(margin = 1).to(device)
    criterion_classifier = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    patience = 50
    earlystop = EarlyStop(saved_model_path = args.saved_model, patience = patience, mode = 'max')

    stage_n_people = [args.n_people]
    #stage_n_people = [args.n_people//8, args.n_people]

    net = None
    #net = torch.load('./saved_model/basic.model')
    train_stage = 0
    val_stage = 0
    triplet = False

    for n_people in stage_n_people:
        #net = resnet18(n_people).to(device)
        #net = vgg16(n_people).to(device)
        #net = vgg11_bn_MobileNet(num_classes=n_people).to(device)
        #net = vgg11_a2_MobileNet(num_classes=n_people).to(device)
        net = vgg11_bn_depth_fire().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.5,0.999), weight_decay=1e-6)
        #val_triplet_loader = generate_triplet_loader(x_val, y_val, args.batch_size, n_people)
        #train_triplet_loader = generate_triplet_loader(x_train, y_train, args.batch_size, n_people)
        for epoch in range(500):
            t_acc, train_stage = train_cross(net, optimizer, criterion_classifier, train_class_loader, epoch, train_stage)
            #if epoch > 0 and (epoch % 3) == 0:
            #    train_stage = train_triplet(net, optimizer, criterion, criterion_classifier, 
            #                                train_triplet_loader, epoch, train_stage)
            acc, val_stage = valid_cross(net, criterion_classifier, val_loader, epoch, val_stage)
            stop, p = earlystop.run(acc, net)
            if (stop):
                break
