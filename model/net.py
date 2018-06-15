import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data
import torch.nn.functional as F

class basic_vgg(nn.Module):
    def __init__(self):
        super(basic_vgg, self).__init__()
        vgg = models.vgg16(pretrained = False)
        num_classes = 2360
        self.features = vgg.features # batch, 512, 6, 5
        self.classifier = nn.Sequential(
                nn.Linear(512*6*5, 4096, bias = True),
                nn.ReLU(),
                nn.Linear(4096, 4096, bias = True),
                nn.ReLU(),
                nn.Linear(4096, num_classes),
            )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size
    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(x.shape[:-1][0], x.shape[-1]//self._pool_size, self_pool_size).max(-1)
        return m

class facenet(nn.Module):
    def __init__(self):
        super(facenet, self).__init__()
        self.lrn    = nn.LocalResponseNorm(2)
        self.maxpool= nn.MaxPool2d(3, stride=2)
        self.conv1  = nn.Conv2d(  3,  64, (7,7), stride=2)
        self.conv2a = nn.Conv2d( 64,  64, (1,1), stride=1)
        self.conv2  = nn.Conv2d( 64, 192, (3,3), stride=1)
        self.conv3a = nn.Conv2d(192, 192, (1,1), stride=1)
        self.conv3  = nn.Conv2d(192, 384, (3,3), stride=1)
        self.conv4a = nn.Conv2d(384, 384, (1,1), stride=1)
        self.conv4  = nn.Conv2d(384, 256, (3,3), stride=1)
        self.conv5a = nn.Conv2d(256, 256, (1,1), stride=1)
        self.conv5  = nn.Conv2d(256, 256, (3,3), stride=1)
        self.conv6a = nn.Conv2d(256, 256, (1,1), stride=1)
        self.conv6  = nn.Conv2d(256, 256, (3,3), stride=1)
        self.feature = nn.Sequential(
                nn.Linear(4*2*256, 32*128*2, bias = True),
                nn.ReLU(),
                Maxout(2),
                nn.Linear( 32*128, 32*128*2, bias = True),
                nn.ReLU(),
                Maxout(2),
                nn.Linear(32*128, 128),
            )
    def forward(self, x):
        x = lrn(maxpool(F.relu(conv1(x))))
        x = maxpool(lrn(F.relu(conv2(F.relu(conv2a(x))))))
        x =     maxpool(F.relu(conv3(F.relu(conv3a(x)))))
        x =             F.relu(conv4(F.relu(conv4a(x))))
        x =             F.relu(conv5(F.relu(conv5a(x))))
        x =     maxpool(F.relu(conv6(F.relu(conv6a(x)))))
        x = x.view(x.size(0), -1)
        x = self.features(x)
        xn = torch.norm(x, p=2, dim=1).detach()
        x = x.div(xn.expand_as(x))
        return x
