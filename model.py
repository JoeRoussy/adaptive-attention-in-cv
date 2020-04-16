import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionConv
from attention_augmented_conv import AugmentedConv


#TODO Make width not increase with # groups
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=64, args=None):

        super(Bottleneck, self).__init__()
        self.stride = stride
        groups = args.groups # Number of attention heads

        '''

        # TODO : Doubt in width, when base_width != 64?
        width = int(out_channels * (base_width / 64.))\
            if args.attention_conv\
            else int(out_channels * (base_width / 64.)) * groups
        '''
        width = out_channels

        additional_args = {'groups':groups, 'R':args.R, 'z_init':args.z_init, 'adaptive_span':args.adaptive_span} \
                            if args.all_attention else {'bias': False}

        kernel_size = args.attention_kernel if args.all_attention else 3
        padding = int((kernel_size - 1) / 2)

        layer = None

        if args.attention_conv:
            layer = AugmentedConv(width, width, kernel_size, args.dk, args.dv, groups, shape=width)
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
        divider = 2 #if args.small_version else 1
        layer_channels = None #These two sets of channels give approximately equal #params between all_attention and all_conv
        
        if args.all_attention:
            #layer_channels = [64,128,128,256,256]
            layer_channels = [32, 64, 128] if args.smallest_version else [64//divider, 128//divider, 256//divider, 512//divider] # [96, 128, 128, 256]
        else:
            layer_channels = [32, 64, 128] if args.smallest_version\
                else [64//divider, 128//divider, 256//divider, 512//divider]

        self.args = args
        self.in_places = 64 if args.dataset == 'TinyImageNet' else 32
        self.all_attention = args.all_attention
        self.attention_kernel = args.attention_kernel

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ) if args.dataset == 'TinyImageNet' else nn.Sequential(
            nn.Conv2d(3, 64 // divider, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64 // divider),
            nn.ReLU()
        )
        self.layers = nn.ModuleList()

        strides = [2]*3 if args.smallest_version else [1] + [2]*3
        for i in range(len(layer_channels)):
            self.layers.append(self._make_layer(block, layer_channels[i], num_blocks[i], stride=strides[i]))

        self.dense = nn.Linear(layer_channels[-1] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride, args=self.args)) #in_places is #input_channels
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def get_span_l1(self, args):
        num_abs_spans = 0
        if args.all_attention:
            for l in self.layers:
                for l2 in l:
                    sum_layer = l2.conv2[0].adaptive_mask.current_val.abs().sum()
                    num_abs_spans += sum_layer

        return num_abs_spans

    def clamp_span(self):
        for l in self.layers:
            for l2 in l:
                l2.conv2[0].adaptive_mask.clamp_param()

    def forward(self, x):
        # TODO(Joe): See if there is some other modification we can make so we don't need to have different pooling kernels at the end of the model
        pooling_kernel_size = 2 if self.args.dataset == 'TinyImageNet' else 4

        out = self.init(x)
        
        for layer in self.layers:
            out = layer(out)

        out = F.avg_pool2d(out, pooling_kernel_size)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def ResNet26(num_classes=1000, args=None):
    if args.smallest_version:
        num_blocks = [1]*3
    elif args.small_version:
        #Decided to use same architecture for both all conv and all attention for better comparison
        num_blocks = [1]*4 #[1, 2, 2, 1] if args.all_attention else [1]*4
    else:
        num_blocks = [1,3,4,1] #Now all attention is 3.02M and CNN is 3.09 M params
        #num_blocks = [1, 2, 4, 1]

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
