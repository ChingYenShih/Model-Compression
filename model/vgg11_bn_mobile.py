import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
                #resnet_feat,
                features,
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm2d(512),
                #nn.Conv2d(512, 256, 3),#, padding=1),
                #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv_dw(512, 256, 3),
                nn.ReLU(),
                Flatten(),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(256),
                nn.Linear(256, 128, bias = True)
                #nn.BatchNorm1d(1024),
                #nn.Linear(1024, 128, bias = True)
            )
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        feat = x.view(x.size(0), -1)
        x = self.classifier(feat)
        return x, feat

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
    #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A': [ 32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_bn_MobileNet( **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), num_classes=2360,**kwargs)
    return model
