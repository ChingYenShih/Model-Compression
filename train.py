import numpy as np
import os

if __name__ == '__main__':
    train_dic = np.load(os.path.join('preproc_data', 'train.npz'))
    x_train = train_dic['x']
    y_train = train_dic['y']

    val_dic = np.load(os.path.join('preproc_data', 'val.npz'))
    x_val = val_dic['x']
    y_val = val_dic['y']


    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
