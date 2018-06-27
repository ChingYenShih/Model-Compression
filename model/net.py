import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class basic_vgg(nn.Module):
    def __init__(self):
        super(basic_vgg, self).__init__()

        num_classes = 2360

        vgg11_feat = models.vgg11_bn(pretrained = False).features

        self.features = nn.Sequential(
                vgg11_feat,
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 256, 3),#, padding=1),
                nn.ReLU(),
                Flatten(),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(256),
                nn.Linear(256, 128, bias = True)
            )
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(128),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        feat = x.view(x.size(0), -1)
        x = self.classifier(feat)
        return x, feat



# Orignal MobileNet Architecture
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2)
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Suquential(
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128, bias = True),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2360, bias = True)
            )

    def forward(self, x):
        x = self.model(x)
        feat = x.view(-1, 1024)
        x = self.fc(feat)
        return x,feat

