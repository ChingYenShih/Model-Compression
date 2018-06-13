import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
def read_preproc_data(path):
    dic = np.load(path)
    x = dic['x']
    y = dic['y']

    return x, y

class MyDataset(Dataset):
    def __init__(self, img, id):
        self.x = torch.from_numpy(img).float()
        self.y = torch.from_numpy(id).long()
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

def get_data_loader(x, y, batch_size = 32, shuffle = True):
    dataset = MyDataset(x, y)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 8
    )
    return loader