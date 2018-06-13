import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
from model.net import network
def get_data_loader(x, y, shuffle = True):
    dataset = Data.TensorDataset(data_tensor = torch.from_numpy(x_train).float(),\
                                       target_tensor = torch.from_numpy(y_train).long())
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size,
        shuffle = shuffle,
        num_workers = 8
    )
    return loader
def train(net, optimizer, criterion, loader):
    pbar = tqdm(iter(loader))
    correct = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device) / 255.0, y_batch.to(device)
        #pred = net(x_batch)
        #_, pred_class = torch.max(pred, 1)

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device) / 255.0, y_batch.to(device)
        #pred = net(x_batch
        #_, pred_class = torch.max(pred, 1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    train_dic = np.load(os.path.join('preproc_data', 'train.npz'))
    x_train = train_dic['x']
    y_train = train_dic['y']

    val_dic = np.load(os.path.join('preproc_data', 'val.npz'))
    x_val = val_dic['x']
    y_val = val_dic['y']

    train_loader = get_data_loader(x_train, y_train, shuffle = True)
    val_loader = get_data_loader(x_val, y_val, shuffle = False)

    criterion = nn.CrossEntropyLoss()
    net = network.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-5)

    for i in range(50000):
        train(net, optimizer, criterion, train_loader)
        valid(net, criterion, valid_loader)
