import os
import scipy.misc
import numpy as np
import pickle
import argparse
def save_as_pickle(mode = 'train'):
    '''
    given 'train' or 'val' mode
    save image(x) & id(y) as pickle file @ preproc_data
    '''
    id_txt_path = os.path.join('dataset', '{}_id.txt'.format(mode))

    x, y = [], []
    dic = {}
    with open(id_txt_path, 'r') as f:
        for line in f:
            image_name = line.split(' ')[0]
            id = int(line.split(' ')[1])
            image_path = os.path.join('dataset', mode, image_name)
            #img = np.transpose(scipy.misc.imread(image_path), (2, 0, 1)) # 3 x 218 x 178
            img = scipy.misc.imread(image_path) # 3 x 218 x 178

            x.append(img)
            y.append(id)
    #dic['x'] = np.array(x)
    #dic['y'] = np.array(y)

    np.savez('preproc_data/{}'.format(mode), x = np.array(x), y = np.array(y))
    #with open('preproc_data/{}.p'.format(mode), 'wb') as f:
    #    pickle.dump(dic, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Save data as pickle')
    parser.add_argument('-m', '--mode', default = 'train', help = 'Read train or val data')
    args = parser.parse_args()

    save_as_pickle(mode = args.mode)
