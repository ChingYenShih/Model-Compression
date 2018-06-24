import os
import scipy.misc
import numpy as np
import argparse
import skimage.transform
import torch
def save_as_tensor(mode = 'train'):
    '''
    given 'train' or 'val' mode
    save image(x) & id(y) as tensor file @ preproc_data
    '''
    id_txt_path = os.path.join('dataset', '{}_id.txt'.format(mode))

    x, y = [], []
    with open(id_txt_path, 'r') as f:
        for line in f:
            image_name = line.split(' ')[0]
            id = torch.Tensor([int(line.split(' ')[1])]).long()
            image_path = os.path.join('dataset', mode, image_name)
            img = scipy.misc.imread(image_path) # 3 x 218 x 178
            img = img[60:185, 40:150, :]
            img = np.transpose(img, (2, 0, 1)) # 3 x 125 x 110
            img = torch.from_numpy(img).view(1, 3, 125, 110)

            x.append(img)
            y.append(id)


    torch.save(torch.cat(x), '/mnt/data/r06942052/preproc_data/{}_crop_125110_img.pt'.format(mode)) 
    #torch.save(torch.cat(y), '/mnt/data/r06942052/preproc_data/{}_id.pt'.format(mode))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Save data as tensor')
    parser.add_argument('-m', '--mode', default = 'train', help = 'Read train or val data')
    args = parser.parse_args()

    save_as_tensor(mode = args.mode)
