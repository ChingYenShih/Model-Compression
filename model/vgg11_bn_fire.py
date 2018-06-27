import torch.nn as nn
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data
import torch
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class vgg11_bn_fire(nn.Module):
    def __init__(self):
        super(vgg11_bn_fire, self).__init__()
        s_ratio = 0.125
        num_classes = 2360
        #vgg11_feat = models.vgg11_bn(pretrained = False).features
        vgg11_feat = nn.Sequential(
 			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

 			#nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(64, int((64*2)*s_ratio) , 64, 64),
 			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 			#nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(128, int((128*2)*s_ratio) , 128, 128),
 			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			#nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(256, int((128*2)*s_ratio) , 128, 128),
 			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 			#nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(256, int((256*2)*s_ratio) , 256, 256),
 			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(512, int((256*2)*s_ratio) , 256, 256),
 			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(512, int((256*2)*s_ratio) , 256, 256),
 			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 			Fire(512, int((256*2)*s_ratio) , 256, 256),
 			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
 			nn.ReLU(),
 			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        	)
        self.features = nn.Sequential(
                vgg11_feat,
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm2d(512),
                #nn.Conv2d(512, 256, 3),#, padding=1),  # size: 32 256 1 1
                Fire(512, int((128*2)*s_ratio) , 128, 128), # size: 32 256 3 3
                nn.ReLU(),
                Flatten(),
                nn.Dropout(p = 0.3), # There is a bug of dropout in pytorch 0.4.0 ???
                nn.BatchNorm1d(256*3*3),
                nn.Linear(256*3*3, 128, bias = True)

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
