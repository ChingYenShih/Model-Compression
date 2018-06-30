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

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            features,
            nn.Dropout(0.3),
            #nn.BatchNorm2d(512),
            #nn.Conv2d(512,256,(3,3)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(3,3)),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, bias=True),
        )
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        xn = torch.norm(x, p=2, dim=1).view(-1,1)
        x = x.div(xn.expand_as(x))
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        x = x.view(x.size(0), -1)
        xn = torch.norm(x, p=2, dim=1).view(-1,1)
        x = x.div(xn.expand_as(x))
        x  = self.classify(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = conv_dw(in_channels, v, 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'a1': [64, 'M', 128, 'M', 256, 256, 'M',256, 256, 'M', 256, 512, 'M'],
    'a2': [16, 'M',  32, 'M',  64,  64, 'M',128, 128, 'M', 256, 256, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11_bn_MobileNet( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model
    
def vgg11_shallow_MobileNet( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['a1'], batch_norm=True), **kwargs)
    return model

def vgg11_a2_MobileNet( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['a2'], batch_norm=True), **kwargs)
    return model

def vgg16_bn_MobileNet( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model
