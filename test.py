import os
import scipy.misc
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as trans
import torch.nn as nn
import utils
from tqdm import tqdm
from collections import namedtuple, OrderedDict
import time
import quant

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

def valid(net, criterion, loader):
    pbar = tqdm(iter(loader))
    net.eval()
    correct = 0
    total_loss = 0
    count = 0
    for x_batch, y_batch in pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
 
        pred = net.forward_classifier(x_batch)

        loss = criterion(pred, y_batch)
        
        total_loss += loss.item()
        _, pred_class = torch.max(pred, 1)
        correct += (pred_class == y_batch).sum().item()

        count += len(x_batch)

        pbar.set_description('Validation stage: Avg loss: {:.4f}; Avg acc: {:.2f}%'.\
            format(total_loss / count, correct / count * 100))
    acc = correct / count * 100
    return acc

def cal_val_acc(net):

    net = net.to(device)
    net.eval()
    mapping = np.load("/mnt/data/r06942052/preproc_data/map.npz")
    mapping = mapping['map'].reshape(1)[0] 
    criterion = nn.CrossEntropyLoss()

    x_val, y_val = utils.read_preproc_data(os.path.join('/mnt/data/r06942052/preproc_data', 'val.npz'))
    val_loader = utils.get_data_loader(x_val, y_val, mapping, dataAUG = False, batch_size = 32, shuffle = False)
    val_acc = valid(net, criterion, val_loader)


def quantize(model_raw, quant_method = 'log'):
    bn_bits = args.bn_bits
    #bn_bits = 32
    param_bits = args.param_bits
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
            #print(k, bits)
            #print(v_quant)
        model_raw.load_state_dict(state_dict_quant)
    return model_raw

def read_test_data(path):
    #if(os.path.isfile('preproc_data/test.npz')):
    #    dic = np.load('preproc_data/test.npz')
    #    x = dic['x']
    #    return x
    #else:
    img_name_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    img_name_list.sort()
    x = []

    for img_name in img_name_list:
        img_path = os.path.join(path, img_name)
        img = scipy.misc.imread(img_path) # 3 x 218 x 178
        x.append(Image.fromarray(img))
    #np.savez('preproc_data/test', x = np.array(x))
    #return np.array(x)
    return x

def test(x_test_all):
    net = torch.load(args.model).to(device)
    net.eval()
    mapping = np.load("/mnt/data/r06942052/preproc_data/inv_map.npz")
    mapping = mapping['inv_map'].reshape(1)[0] 

    f = open(args.output_path, 'w')
    f.write('id,ans\n')
    transform = trans.Compose([
                trans.CenterCrop(120),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
                ])
    for i, x_test in enumerate(x_test_all):
        """
        x_test = torch.from_numpy(x_test).float().to(device) / 255.0
        """
        x_test = transform(x_test)
        x_test = x_test.unsqueeze(0).to(device)

        pred = net.forward_classifier(x_test)[0].detach()
        _, pred_class = torch.max(pred, 0)

        print(i,end='\r')
        f.write('{},{}\n'.format(i + 1, mapping[pred_class.item()]))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Testing and save result')
    parser.add_argument('-i', '--input_dir', default = os.path.join('dataset', 'test'), help = 'Test image directory')
    parser.add_argument('-m', '--model', help = 'Select which model to test')
    parser.add_argument('-o', '--output_path', default = 'result/result.csv', help = 'Saved file path')
    parser.add_argument('-bn', '--bn_bits', type = int, default = 16, help = 'Quantized number of bits(bn)')
    parser.add_argument('-p', '--param_bits', type = int, default = 16, help = 'Quantized number of bits(param)')
    parser.add_argument('-qm', '--quant_method', type = int, default = 1, help = 'Quantization method')
    parser.add_argument('-d', '--device_id', type = int, default = 0, help = 'Set GPU device')
    args = parser.parse_args()


    device = torch.device("cuda:{}".format(args.device_id))
    method_list = ['linear', 'log', 'minmax', 'other']

    q_net = quantize(torch.load(args.model), quant_method = method_list[args.quant_method])
    cal_val_acc(q_net)
    #torch.save(q_net, 'half.pth')
    net = torch.load(args.model)
    cal_val_acc(net)
 

    x_test_all = read_test_data(path = args.input_dir)
    test(x_test_all)

    '''
        if want to test with batch, comment line 86 and 87,
        and run below two lines
    '''
    #loader = get_data_loader(x_test_all)
    #test_batch(loader)
