import os
import scipy.misc
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as trans

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
    mapping = np.load("./preproc_data/inv_map.npz")
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

        pred = net(x_test)[0].detach()
        _, pred_class = torch.max(pred, 1)

        print(i)
        f.write('{},{}\n'.format(i + 1, mapping[pred_class.item()]))
    f.close()
def test_batch(loader):
    net = torch.load(args.model).to(device)
    net.eval()
    f = open(args.output_path, 'w')
    f.write('id,ans\n')
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
    test(x_test_all)

    '''
        if want to test with batch, comment line 86 and 87,
        and run below two lines
    '''
    #loader = get_data_loader(x_test_all)
    #test_batch(loader)
