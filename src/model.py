import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from collections import OrderedDict 
from config import B, S, C, IMG_SIZE
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

class TransitionL(nn.Module):
    def __init__(self, in_features, out_features):
        """[summary]

        Args:
            in_features ([type]): [description]
            out_features ([type]): [description]
        """
        super(TransitionL, self).__init__()
        self.transit = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_features, out_features, 1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    def forward(self, x):
        return self.transit(x)
    
class _DenseLayer(nn.Module):
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        """[summary]

        Args:
            num_input_features ([type]): [description]
            growth_rate ([type]): [description]
            bn_size ([type]): [description]
            drop_rate ([type]): [description]
            memory_efficient (bool, optional): [description]. Defaults to False.
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
    
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        """[summary]

        Args:
            num_layers ([type]): [description]
            num_input_features ([type]): [description]
            bn_size ([type]): [description]
            growth_rate ([type]): [description]
            drop_rate ([type]): [description]
            memory_efficient (bool, optional): [description]. Defaults to False.
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, memory_efficient=False):
        """Densenet

        Args:
            growth_rate (int, optional): Num channel stack. Defaults to 32.
            block_config (tuple, optional): Num layers each block. Defaults to (6, 12, 24, 16).
            num_init_features (int, optional): input features map. Defaults to 64.
            bn_size (int, optional): bottleneck params. Defaults to 4.
            drop_rate (int, optional): trigger dropout > 0. Defaults to 0.
            memory_efficient (bool, optional): saving memory. Defaults to False.
        """

        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple denseblocks based on config 
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to 
                # downsample
                trans = TransitionL(num_features,
                                    num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.lastconv = nn.Conv2d(num_features, num_features, 1, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features_map = self.lastconv(features)
        return features_map

class YOLOD(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self):
        super(YOLOD, self).__init__()
        self.feature_extractor = DenseNet()
        self.grid = S
        self.num_classes = C

        self.linear_layers = nn.Sequential(
            nn.Linear(S*S*1024, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.grid*self.grid*(self.num_classes + B*5))
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        flatten = torch.flatten(features)
        print(flatten.size())
        flatten = flatten.view(x.size()[0], -1)
        print(flatten.size())

        linear_vec = self.linear_layers(flatten)
        output = linear_vec.view(-1, self.grid, self.grid, self.num_classes + B*5)
        return output
    
if __name__ == '__main__':
    yoloS = YOLOD()
    print(yoloS(torch.ones([2, 3, 448, 448])).shape)