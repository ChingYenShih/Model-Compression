import os
import scipy.misc
import numpy as np
import argparse
import skimage.transform
import torch
def save_as_pickle(mode = 'train'):
    '''
    given 'train' or 'val' mode
    save image(x) & id(y) as tensor file @ preproc_data
    '''
    id_txt_path = os.path.join('dataset', '{}_id.txt'.format(mode))

    x, y = [], []
    with open(id_txt_path, 'r') as f:
        for line in f:
            image_name = line.split(' ')[0]
            id = int(line.split(' ')[1])
            image_path = os.path.join('dataset', mode, image_name)
            img = scipy.misc.imread(image_path) # 3 x 218 x 178

            x.append(img)
            y.append(id)


    np.savez('/mnt/data/r06942052/preproc_data/{}'.format(mode), x=np.array(x), y=np.array(y))
    #torch.save(torch.cat(x), '/mnt/data/r06942052/preproc_data/{}_crop_125110_img.pt'.format(mode)) 
    #torch.save(torch.cat(y), '/mnt/data/r06942052/preproc_data/{}_id.pt'.format(mode))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Save data as tensor')
    parser.add_argument('-m', '--mode', default = 'train', help = 'Read train or val data')
    args = parser.parse_args()

    save_as_pickle(mode = args.mode)
