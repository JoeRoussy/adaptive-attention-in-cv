import torch
from thop import profile
import torch.nn.functional as F
from attention import AttentionConv
from model import Bottleneck
from config import get_args
from preprocess import load_data
from model import ResNet26


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


def count_adaptive_flops(m, x, mask_len):
    total_ops = 0
    one_d_mask = m.mask_template + m.current_val * m._max_size
    # TODO HIGH Shakti : check if flops holds for multiple groups!
    total_ops = total_ops + 1*m.current_val.numel() + m.mask_template.numel()

    # one_d_mask = one_d_mask / m._ramp_size + 1
    total_ops = total_ops + one_d_mask.numel()

    # one_d_mask = one_d_mask.clamp(0, 1)
    # one_d_mask = one_d_mask[:, -mask_len:]

    kernel_size = 2 * mask_len + 1
    mask = torch.ones(kernel_size, kernel_size)

    # the next block has no flops usage
    # left, right = 0, kernel_size - 1
    # for i in range(one_d_mask.shape[1]):
    #     bottom, top = left, right
    #     indices = [[j, left] for j in range(bottom, top + 1)]  # left edge indices
    #     indices += [[bottom, j] for j in range(left + 1, right + 1)]  # bottom edge minus overlap with left
    #     indices += [[top, j] for j in range(left + 1, right + 1)]  # top minus overlap with left
    #     indices += [[j, right] for j in range(bottom + 1, top)]  # right minus overlap with bottom and top
    #     rows, cols = zip(*indices)
    #     mask[rows, cols] = one_d_mask[0, i]
    #
    #     left += 1
    #     right -= 1

    # mask = mask.view(1, 1, 1, 1, -1)

    # next line doesnt add any value, is only needed for flops
    # x = x * mask
    total_ops += x.numel()

    x = x / (x.sum(-1, keepdim=True) + 1e-8)
    total_ops = total_ops + x[0, :, :, :, 0].numel() * count_softmax(x[:, 0, 0, 0, :])

    return total_ops


def count_attention_flops(m, x):
    # x = x[0]
    total_count = 0
    batch, channels, height, width = x.size()
    max_size = None
    if m.adaptive_span:
        max_size = m.adaptive_mask.get_current_max_size()
        kernel_size = int(2 * max_size + 1)
        padding = int((kernel_size - 1) / 2)

    else:
        kernel_size = m.kernel_size
        padding = m.padding

    padded_x = F.pad(x, [padding, padding, padding, padding])

    total_ops, q_out = count_conv2d(m.query_conv, x)
    total_count += total_ops
    # m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops, k_out = count_conv2d(m.key_conv, padded_x)
    # m.total_ops += torch.DoubleTensor([int(total_ops)])
    total_count += total_ops


    total_ops, v_out = count_conv2d(m.value_conv, padded_x)
    # m.total_ops += torch.DoubleTensor([int(total_ops)])
    total_count += total_ops

    k_out = k_out.unfold(2, kernel_size, m.stride).unfold(3, kernel_size, m.stride)
    v_out = v_out.unfold(2, kernel_size, m.stride).unfold(3, kernel_size, m.stride)

    if m.adaptive_span:
        start_ind = (m.kernel_size // 2) - (kernel_size // 2)
        end_ind = (m.kernel_size // 2) + (kernel_size // 2)
        rel_h = m.rel_h[:, :, :, start_ind:end_ind + 1, :]
        rel_w = m.rel_w[:, :, :, :, start_ind:end_ind + 1]
    else:
        rel_h = m.rel_h
        rel_w = m.rel_w

    k_out_h, k_out_w = k_out.split(m.out_channels // 2, dim=1)
    k_out = torch.cat((k_out_h + rel_h, k_out_w + rel_w), dim=1)

    k_out = k_out.contiguous().view(batch, m.groups, m.out_channels // m.groups, height, width, -1)
    v_out = v_out.contiguous().view(batch, m.groups, m.out_channels // m.groups, height, width, -1)

    q_out = q_out.view(batch, m.groups, m.out_channels // m.groups, height, width, 1)

    out = (q_out * k_out).sum(dim=2)
    # TODO HIGH Shakti: CHeck if this multiplication is correct?
    total_ops = q_out.numel() * k_out.size(-1)
    # m.total_ops += torch.DoubleTensor([int(total_ops)])
    total_count += total_ops

    out2 = F.softmax(out, dim=-1)
    # get the softmax count for one batch, one set of features
    total_ops = count_softmax(out[:, 0, 0, 0, :])
    # now multiply with the total groups, and width x height
    total_ops = total_ops * out[0, :, :, :, 0].numel()
    # m.total_ops += torch.DoubleTensor([int(total_ops)])
    total_count += total_ops


    if m.adaptive_span:
        total_ops = count_adaptive_flops(m.adaptive_mask, out2, int(max_size))
        # m.total_ops += torch.DoubleTensor([int(total_ops)])
        total_count += total_ops

    # out3 = (out2.unsqueeze(dim=2) * v_out).sum(dim=-1).view(batch, -1, height, width)
    # m.total_ops += v_out.numel()
    total_count += total_ops

    return total_count

def count_batchnorm2d(m, x):
    nelements = x.numel()
    # TODO : Check this m.training
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    return total_ops


def count_avgpool2d(y):
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    return total_ops


def count_bootleneck(m, x, y):
    x = x[0]

    # out = m.conv1(x)
    # conv1 consists of 3 layers
    # self.conv1 = nn.Sequential(
    #     nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
    #     nn.BatchNorm2d(width),
    #     nn.ReLU(),
    # )

    total_ops, out = count_conv2d(m.conv1[0], x)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops = count_batchnorm2d(m.conv1[1], out)
    m.total_ops += torch.DoubleTensor([int(total_ops)])
    out = m.conv1[1](out)

    out = F.relu(out)
    total_ops = out.numel()
    m.total_ops += torch.DoubleTensor([int(total_ops)])


    # out += m.conv2(out)
    # conv2 consists of a layer and dropouts
    if args.all_attention:
        total_ops = count_attention_flops(m.conv2[0], out)
        out = m.conv2(out)
        m.total_ops += torch.DoubleTensor([int(total_ops)]) + torch.DoubleTensor([int(out.numel())])
    else:
        total_ops, out = count_conv2d(m.conv2[0], out)
        m.total_ops += torch.DoubleTensor([int(total_ops)]) + torch.DoubleTensor([int(out.numel())])

    total_ops = count_batchnorm2d(m.norm, out)
    m.total_ops += torch.DoubleTensor([int(total_ops)])
    out = m.norm(out)

    # out = m.conv3(out)
    # self.conv3 = nn.Sequential(
    #     nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
    #     nn.BatchNorm2d(self.expansion * out_channels),
    # )
    total_ops, out = count_conv2d(m.conv3[0], out)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops = count_batchnorm2d(m.conv3[1], out)
    m.total_ops += torch.DoubleTensor([int(total_ops)])
    out = m.conv3[1](out)


    if m.stride >= 2:
        out = F.avg_pool2d(out, (m.stride, m.stride))
        total_ops = count_avgpool2d(out)
        m.total_ops += torch.DoubleTensor([int(total_ops)])

    # out += m.shortcut(x)
    # self.shortcut = nn.Sequential(
    #     nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
    #     nn.BatchNorm2d(self.expansion * out_channels)
    # )
    total_ops, out_1 = count_conv2d(m.shortcut[0], x)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops = count_batchnorm2d(m.shortcut[1], out_1)
    m.total_ops += torch.DoubleTensor([int(total_ops)])
    out += m.shortcut[1](out_1)
    # out += m.shortcut(x)
    total_ops = x.numel()
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    out = F.relu(out)
    total_ops = out.numel()
    m.total_ops += torch.DoubleTensor([int(total_ops)])


if __name__ == '__main__':
    input = torch.randn((2, 3, 32, 32))
    args, logger = get_args()
    num_classes = 100
    model = ResNet26(num_classes=num_classes, args=args)
    # conv = AttentionConv(3, 16, kernel_size=3, padding=1, adaptive_span=True)
    macs, params = profile(model, inputs=(input, ), custom_ops={Bottleneck: count_bootleneck}, verbose=True)
    print(macs, params)
