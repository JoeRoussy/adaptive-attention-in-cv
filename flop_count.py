import torch
from thop import profile
import torch.nn.functional as F
from attention import AttentionConv


def count_conv2d(m, x):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    y = m(x)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)
    return total_ops, y


def count_softmax(x):
    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return total_ops

