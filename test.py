import os
import scipy.misc
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from model.net import *

class TestDataset(Dataset):
    def __init__(self, img):
        self.x = torch.from_numpy(img).float()
    def __getitem__(self, index):
        return self.x[index]
    def __len__(self):
        return len(self.x)

def get_data_loader(x, batch_size = 32, shuffle = False):
    dataset = TestDataset(x)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 8
    )
    return loader

def read_test_data(path):
    if(os.path.isfile('/mnt/data/r06942052/preproc_data/test_img.pt')):
        x = torch.load('/mnt/data/r06942052/preproc_data/test_img.pt')
        return x
    else:
        img_name_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        img_name_list.sort()
        x = []
        for img_name in img_name_list:
            img_path = os.path.join(path, img_name)
            img = np.transpose(scipy.misc.imread(img_path), (2, 0, 1)) # 3 x 218 x 178
            img = img[:, 60:185, 40:150]
            img = torch.from_numpy(img).view(1, 3, 125, 110)
            x.append(img)
        torch.save(torch.cat(x), '/mnt/data/r06942052/preproc_data/test_img.pt') 
        return torch.cat(x)

def read_mapping(path):
    if(os.path.isfile('/mnt/data/r06942052/preproc_data/mapping.pt')):
        x = torch.load('/mnt/data/r06942052/preproc_data/mapping.pt')
        return x
    else:
        y_train = torch.load('/mnt/data/r06942052/preproc_data/train_id.pt')

        #Triplet selection
        bm_train = y_train.numpy()
        count_train = np.zeros((10177))

        for i in range(10177):
            count_train[i] = np.sum(bm_train == i)
        mapping = np.vstack((count_train.nonzero()[0].reshape(1, -1), np.arange(2360).reshape(1, -1)))
        mapping = mapping.astype('float')
        x = torch.save(mapping, '/mnt/data/r06942052/preproc_data/mapping.pt')
        return mapping

def test(x_test_all, mapping):
    net = torch.load(args.model).to(device)
    net.eval()
    f = open(args.output_path, 'w')
    
    f.write('id,ans\n')
    for i, x_test in enumerate(x_test_all):
        x_test = x_test.float().to(device) / 255.0
        x_test = x_test.unsqueeze(0)

        pred = net.forward_classifier(x_test).detach()
        _, pred_class = torch.max(pred, 1)
        
        f.write('{},{}\n'.format(i + 1, mapping[0, pred_class.item()].astype('int')))
    f.close

def test_batch(loader):
    net = torch.load(args.model).to(device)
    net.eval()
    f = open(args.output_path, 'w')
    counter = 1
    for x_test in loader:
        x_test = x_test.to(device) / 255.0
        pred = net(x_test).detach()
        _, pred_classes = torch.max(pred,1)
        pred_classes = pred_classes.cpu().numpy()

        for pred_class in pred_classes:
            print(counter)
            f.write('{},{}\n'.format(counter, pred_class))
            counter += 1
    f.close
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Testing and save result')
    parser.add_argument('-i', '--input_dir', default = os.path.join('dataset', 'test'), help = 'Test image directory')
    parser.add_argument('-m', '--model', help = 'Select which model to test')
    parser.add_argument('-o', '--output_path', default = 'result/result.csv', help = 'Saved file path')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    x_test_all = read_test_data(path = args.input_dir)
    mapping = read_mapping(path = args.input_dir)
    test(x_test_all, mapping)

    '''
        if want to test with batch, comment line 86 and 87,
        and run below two lines
    '''
    #loader = get_data_loader(x_test_all)
    #test_batch(loader)
