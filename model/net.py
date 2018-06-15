import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data

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

