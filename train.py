import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.utils.data as Data
def train():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic model training process')
    parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    args.parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    train_dic = np.load(os.path.join('preproc_data', 'train.npz'))
    x_train = train_dic['x']
    y_train = train_dic['y']

    val_dic = np.load(os.path.join('preproc_data', 'val.npz'))
    x_val = val_dic['x']
    y_val = val_dic['y']

    train_dataset = Data.TensorDataset(data_tensor = torch.from_numpy(x_train).float())

    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(.parameters(), lr = 5e-5)
    for i in range(50000):
        train()
