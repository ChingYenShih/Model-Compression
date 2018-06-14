import os
import scipy.misc
import numpy as np
import argparse
import torch
def save_as_tensor(mode = 'train'):
    '''
    given 'train' or 'val' mode
    save image(x) & id(y) as tensor file @ preproc_data
    '''
    id_txt_path = os.path.join('dataset', '{}_id.txt'.format(mode))

    x, y = [], []
    dic = {}
    with open(id_txt_path, 'r') as f:
        for line in f:
            image_name = line.split(' ')[0]
            id = torch.Tensor([int(line.split(' ')[1])]).long()
            image_path = os.path.join('dataset', mode, image_name)
            img = np.transpose(scipy.misc.imread(image_path), (2, 0, 1)) # 3 x 218 x 178
            img = torch.from_numpy(img).view(1, 3, 218, 178)

            x.append(img)
            y.append(id)


    torch.save(torch.cat(x), '/data/r06942052/preproc_data/{}_img.pt'.format(mode)) 
    torch.save(torch.cat(y), '/data/r06942052/preproc_data/{}_id.pt'.format(mode))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Save data as tensor')
    parser.add_argument('-m', '--mode', default = 'train', help = 'Read train or val data')
    args = parser.parse_args()

    save_as_tensor(mode = args.mode)
