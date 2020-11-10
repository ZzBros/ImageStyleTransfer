import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        # 用来构建loss的VGG-16特征层
        self.select = ['3', '8', '15' ,'22']
        for parma in self.vgg.parameters():
            parma.requires_grad = False

    def forward(self, x):
        #提取特征层
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features
        
        
class Net(nn.Module):
    """
        快速风格迁移网络
    """
    def __init__(self, inChannels):
        super(Net, self).__init__()
        self.pad = nn.ReflectionPad2d(40)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.cov = [nn.Conv2d(inChannels, 32, 9, padding=4),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1)]
        self.resBlock = [nn.Conv2d(128, 128, 3),
                         nn.Conv2d(128, 128, 3)] * 5
        self.covT = [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                     nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                     nn.ConvTranspose2d(32, inChannels, 9, padding=4)]
        self.bnor = [nn.BatchNorm2d(32), nn.BatchNorm2d(64),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(128),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(128),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(128),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(128),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(128),
                    nn.BatchNorm2d(128), nn.BatchNorm2d(64),
                    nn.BatchNorm2d(32), nn.BatchNorm2d(inChannels)]
        
        
    def forward(self, x):
        x = self.pad(x)
        bn = 0
        for c in self.cov:
            x = c(x)
            x = self.relu(self.bnor[bn](x))
            bn += 1
                
        res = None
        for i, r in enumerate(self.resBlock):
            if i % 2 == 0:
                res = x[:,:,2:-2,2:-2]
                x = r(x)
                x = self.relu(self.bnor[bn](x))
                bn += 1
            else:
                x = r(x)
                x = self.bnor[bn](x) + res
                bn += 1
                        
        for cT in self.covT[: -1]:
            x = cT(x)
            x = self.relu(self.bnor[bn](x))  
            bn += 1
                
        x = self.covT[-1](x)
        x = (self.tanh(self.bnor[bn](x)) + 1) / 2 * 255
        
        return x.to(dev)