import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionConv
from attention_augmented_conv import AugmentedConv


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=64, args=None):

        super(Bottleneck, self).__init__()
        self.stride = stride
        groups = args.groups # Number of attention heads
        width = int(out_channels * (base_width / 64.))\
            if args.attention_conv\
            else int(out_channels * (base_width / 64.)) * groups

        additional_args = {'groups':groups, 'R':args.R, 'z_init':args.z_init, 'adaptive_span':args.adaptive_span} \
                            if args.all_attention else {'bias': False}

        kernel_size = args.attention_kernel if args.all_attention else 3
        padding = 3 if kernel_size==7 else 1  #NEED TO CHANGE THIS FOR WHEN ADAPTIVE

        layer = None

        if args.attention_conv:
            # Assume dk = 40, dv = 4. TODO: Not sure why we use these settings
            dk = 40
            dv = 4
            layer = AugmentedConv(width, width, kernel_size, dk, dv, groups, shape=width)
        elif args.all_attention:
            layer = AttentionConv(width, width, kernel_size=kernel_size, padding=padding, **additional_args)
        else:
            layer = nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding, **additional_args)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            layer,
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, args=None):
        super(Model, self).__init__()
        divider = 2 if args.small_version else 1
        layer_channels = None #These two sets of channels give approximately equal #params between all_attention and all_conv
        if args.all_attention:
            layer_channels = [64,128,128,256,256]
        else:
            layer_channels = [64//divider, 128//divider, 256//divider, 512//divider]

        self.args = args
        self.in_places = 64//divider
        self.all_attention = args.all_attention
        self.attention_kernel = args.attention_kernel

        self.init = nn.Sequential(
            # CIFAR10
            nn.Conv2d(3, 64 // divider, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64 // divider),
            nn.ReLU()

            # For ImageNet
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, layer_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, layer_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, layer_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, layer_channels[3], num_blocks[3], stride=2)
        self.dense = nn.Linear(layer_channels[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride, args=self.args)) #in_places is #input_channels
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def ResNet26(num_classes=1000, args=None):
    if args.small_version:
        #Decided to use same architecture for both all conv and all attention for better comparison
        num_blocks = [1]*4 #[1, 2, 2, 1] if args.all_attention else [1]*4
    else:
        num_blocks = [1, 2, 4, 1]

    return Model(Bottleneck, num_blocks, num_classes=num_classes, args=args)


def ResNet38(num_classes=1000, all_attention=False):
    return Model(Bottleneck, [2, 3, 5, 2], num_classes=num_classes)


def ResNet50(num_classes=1000, all_attention=False):
    return Model(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


# temp = torch.randn((2, 3, 224, 224))
# model = ResNet38(num_classes=1000)
# print(get_model_parameters(model))
