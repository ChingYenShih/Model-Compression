import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class resnet18(nn.Module):
    def __init__(self, n_classes):
        super(resnet18, self).__init__()
        resnet = models.resnet18(pretrained = False)
        self.resnet_feature = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.Conv2d(512,512,(4,4)),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            )
        self.classify = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.resnet_feature(x)
        x = x.view(x.size(0), -1)
        xn = torch.norm(x, p=2, dim=1).view(-1,1)
        alpha = 10
        x = x.div(xn.expand_as(x)) * alpha
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        x = x.view(x.size(0), -1)
        x  = self.classify(x)
        return x

#class Maxout(nn.Module):
#    def __init__(self, pool_size):
#        super().__init__()
#        self._pool_size = pool_size
#    def forward(self, x):
#        assert x.shape[-1] % self._pool_size == 0, \
#            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
#        m, i = x.view(x.shape[:-1][0], x.shape[-1]//self._pool_size, self._pool_size).max(-1)
#        return m

class facenet(nn.Module):
    def __init__(self, n_embeddings, n_classes, pretrained_model=None):
        super(facenet, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(  3,  64, (7,7), stride=2, padding = 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.LocalResponseNorm(2),
                nn.Conv2d( 64,  64, (1,1), stride=1),
                nn.ReLU(),
                nn.Conv2d( 64, 192, (3,3), stride=1, padding = 1),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(192, 192, (1,1), stride=1),
                nn.ReLU(),
                nn.Conv2d(192, 384, (3,3), stride=1, padding = 1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(384, 384, (1,1), stride=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, (3,3), stride=1, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, (1,1), stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3,3), stride=1, padding = 1),
                nn.BatchNorm2d(256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Conv2d(256, 256, (1,1), stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3,3), stride=1, padding = 1),
                nn.BatchNorm2d(256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        self.embedding = nn.Sequential(
                nn.Linear(7*7*256, 32*128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(32*128, n_embeddings),
            )

        self.classify = nn.Sequential(
                nn.Linear(n_embeddings, n_classes),
            )

        if pretrained_model!=None:
            for idx, (p_name, p_module) in enumerate(pretrained_model.named_modules()):
                if (p_name != '' and p_name != 'classify' and len(p_name.split('.'))==1):
                    copy = 'self.'+ p_name +'=p_module'
                    exec(copy)
            
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        xn = torch.norm(x, p=2, dim=1).view(-1,1)
        x = x.div(xn.expand_as(x))
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        x  = self.classify(x)
        return x


class vgg16(nn.Module):
    def __init__(self, n_classes):
        super(vgg16, self).__init__()
        self.vgg_feature = nn.Sequential(
            nn.Conv2d(  3,  64, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d( 64,  64, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 64, 128, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten()
            )
        self.classify = nn.Sequential(
                nn.Dropout(0.7),
                nn.Linear(512*3*3, n_classes),
            )

    def forward(self, x):
        x = self.vgg_feature(x)
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        x  = self.classify(x)
        return x

class vgg(nn.Module):
    def __init__(self, n_classes):
        super(vgg, self).__init__()
        self.vgg_feature = nn.Sequential(
            nn.Conv2d(  3,  64, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d( 64,  64, (3,3), stride=2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d( 64, 128, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), stride=2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), stride=2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (3,3), stride=1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (3,3), stride=2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten()
            )
        self.classify = nn.Sequential(
                nn.Dropout(0.7),
                nn.Linear(512*4*4, n_classes),
            )

    def forward(self, x):
        x = self.vgg_feature(x)
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        x  = self.classify(x)
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
