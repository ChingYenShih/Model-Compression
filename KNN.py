import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.net import basic_vgg
import utils
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

def extract_feature(net, loader):
    pbar = tqdm(iter(loader))
    count = 0
    feats, labels = [], []
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
 
        feat = net.features(x_batch)
        feats.append(feat.detach().cpu().view(-1).numpy())
        labels.append(y_batch.view(-1).item())

    return feats, labels
class EarlyStop():
    def __init__(self, saved_model_path, patience = 10000, mode = 'max'):
        self.saved_model_path = saved_model_path
        self.patience = patience
        self.mode = mode
        
        self.best = 0 if (self.mode == 'max') else np.Inf
        self.current_patience = 0
    def run(self, acc, model):
        condition = (acc > self.best) if (self.mode == 'max') else (acc <= self.best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    parser.add_argument('-m', '--saved_model', default = 'saved_model/basic.model', help = 'Saved model path')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))
    mapping = np.load(os.path.join('preproc_data','map.npz'))['map'].reshape(1)[0]
    print("Loading data")
    x_train, y_train = utils.read_preproc_data(os.path.join('preproc_data', 'train.npz'))
    x_val, y_val = utils.read_preproc_data(os.path.join('preproc_data', 'val.npz'))
    train_class_loader = utils.get_data_loader(x_train, y_train, mapping, batch_size = args.batch_size, shuffle = False)
    val_loader = utils.get_data_loader(x_val, y_val, mapping,batch_size = args.batch_size, shuffle = False)

    print("Initialize model")
    net = torch.load(args.saved_model)
    net.eval()
    net.to(device)
    
    train_feat, train_id = extract_feature(net, train_class_loader)
    val_feat, val_id = extract_feature(net, val_loader)
    neigh5 = KNeighborsClassifier(n_neighbors=5)
    neigh5.fit(train_feat, train_id)
    neigh3 = KNeighborsClassifier(n_neighbors=3)
    neigh3.fit(train_feat, train_id)
    neigh1 = KNeighborsClassifier(n_neighbors=1)
    neigh1.fit(train_feat, train_id)
    pred5 = neigh5.predict(val_feat)
    pred3 = neigh3.predict(val_feat)
    pred1 = neigh1.predict(val_feat)
    correct5 = sum(val_id==pred5)
    correct3 = sum(val_id==pred3)
    correct1 = sum(val_id==pred1)
    print(correct5,correct3,correct1)
