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
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Set up face_datas')
    parser.add_argument('-b', '--batch_size', type = int, default = 30, help = 'Set batch size')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id))

    sys.stdout.write('Loading data.pt...')
    sys.stdout.flush()
    x_train = torch.load('/mnt/data/r06942052/preproc_data/train_crop_img.pt')
    x_val = torch.load('/mnt/data/r06942052/preproc_data/val_crop_img.pt')
    y_train = torch.load('/mnt/data/r06942052/preproc_data/train_id.pt')
    y_val = torch.load('/mnt/data/r06942052/preproc_data/val_id.pt')
    sys.stdout.write('Done\n')

    net = torch.load('saved_model/basic.model').to(device)
    net.eval()

    bm_train = y_train.numpy()
    count_train = np.zeros((10177))
    for i in range(10177):
        count_train[i] = np.sum(bm_train == i)
    mapping = np.vstack((count_train.nonzero()[0].reshape(1, -1), np.arange(2360).reshape(1, -1)))
    mapping = mapping.astype('float')

    all_anchor = []
    all_Id = []
    all_val = []
    for j in range(2360):
        anchor_faces = x_train[y_train == mapping[0, j]]
        anchor = net((anchor_faces.float()/255).to(device)).detach().cpu()
        Id = torch.Tensor([j]).expand(len(anchor_faces))
        all_anchor.append(anchor)
        all_Id.append(Id)
        
        y_val[y_val == mapping[0, j]] = mapping[1, j]

    all_anchor = torch.cat(all_anchor).numpy()
    all_Id = torch.cat(all_Id).numpy()

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit( all_anchor, all_Id)

    val_set = Data.TensorDataset(x_val, y_val)
    val_loader   = Data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False) 

    pbar = tqdm(iter(val_loader))
    correct = 0
    count = 0

    for batch_idx, (x_batch, y_batch) in enumerate(pbar):
        x_batch = x_batch.to(device).float() / 255.0
        y_batch = y_batch.numpy()

        x_embedding = net(x_batch).detach().cpu().numpy()
        result = neigh.predict(x_embedding)        
        correct += np.sum((result == y_batch))

    acc = correct / len(y_val) * 100.
    print('Validation Accuracy: {}%'.format(acc))


