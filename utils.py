import torch.utils.data as Data
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as trans
import scipy.misc
def read_preproc_data(path):
    dic = np.load(path)
    x = dic['x']
    y = dic['y']

    return x, y

class BoDataset(Dataset):
    def __init__(self, x, y, mapping):
        print("Loading dataset")
        self.transform = trans.Compose([
                        trans.RandomHorizontalFlip(),
                        trans.CenterCrop(120),
                        trans.RandomRotation(30),
                        trans.ToTensor(),
                        trans.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),
                        ])
        ids = self.map_label(y, mapping)
        self.num_ids = np.amax(ids)+1
        self.x = self.sort_data(x,ids)
        self.length = [len(self.x[i]) for i in range(len(self.x))]
        self.count = 0
        self.anchors = self.set_anchors()
    def set_anchors(self):
        anchors = []
        for i, length in enumerate(self.length):
            anchors.append(int(np.random.choice(length,1)))
        return anchors
    def map_label(self, id, mapping):
        label = [mapping[lab] for i,lab in enumerate(id)]
        """
        for i,lab in enumerate(id):
            label.append(mapping[lab])
        """
        return np.array(label)
    def sort_data(self,imgs,ids):
        x = [[] for i in range(self.num_ids)] 
        for i in range(len(imgs)):
            img = Image.fromarray(np.uint8(imgs[i]))
            id = ids[i]
            #img, id = imgs[i], ids[i]
            x[id].append(img)
        return x
    def parse_idx(self, idx):
        for i, length in enumerate(self.length):
            if idx - length < 0:
                return i
            else:
                idx -= length
    def __getitem__(self, index):
        idp = self.parse_idx(index)
        idn =  int(np.random.choice(self.num_ids,1))
        while idp == idn:
            idn = int(np.random.choice(self.num_ids,1))
        num1 = self.anchors[idp]
        num2 = int(np.random.choice(self.length[idp],1))
        while num1 == num2:
            num2 = int(np.random.choice(self.length[idp],1))
        num3 = int(np.random.choice(self.length[idn],1))
        np_img = [self.x[idp][num1],self.x[idp][num2],self.x[idn][num3]]

        """
        coin = np.random.randint(2, size=3)
        for i,img in enumerate(np_img):
            if coin[i]:
                img = img[:,::-1,:]
        """
        img1 = self.transform(np_img[0]).float()
        img2 = self.transform(np_img[1]).float()
        img3 = self.transform(np_img[2]).float()
        self.count +=1
        if self.count == self.length:
            self.anchors = self.set_anchors()
            self.count = 0
        return(img1, img2, img3, idp, idn)
    def __len__(self):
        return sum(self.length)

def get_new_loader(x,y , mapp, batch_size = 32, shuffle = True):
    dataset = BoDataset(x, y , mapp)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 8
    )
    return loader

class MyDataset(Dataset):
    def __init__(self, img, id, mapping, data_aug):
        self.transform = trans.Compose([
                        trans.RandomHorizontalFlip(),
                        trans.CenterCrop(120),
                        trans.RandomRotation(30),
                        trans.ToTensor(),
                        trans.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])])
                
        self.transform_val = trans.Compose([
                trans.CenterCrop(120),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
                ])
        self.x = [Image.fromarray(np.uint8(img[i])) for i in range(len(img))] 
        self.y = torch.from_numpy(self.map_label(id,mapping)).long()
        self.data_aug = data_aug
    def trans_img(self, img,id, mapping):
        imgs = [Image.fromarray(np.uint8(img[i])) for i in range(len(img))] 
        ids = torch.from_numpy(self.map_label(id,mapping)).long()
        x, y= [],[]
        for i in range(3):
            print(i)
            for j,image in enumerate(imgs):
                x.append(self.transform(image))
                y.append(ids[j])
        return x,y
    def map_label(self, id, mapping):
        label = []
        for i,lab in enumerate(id):
            label.append(mapping[lab])
        return np.array(label)
    def __getitem__(self, index):
        if(self.data_aug == True):
            img = self.transform(self.x[index])
        else:
            img= self.transform_val(self.x[index])
        id = self.y[index]
        return img,id
    def __len__(self):
        return len(self.x)

def get_data_loader(x, y, mapp, data_aug, batch_size = 32, shuffle = True):
    dataset = MyDataset(x, y, mapp, data_aug)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 12
    )
    return loader
