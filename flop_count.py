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


def count_attention_flops(m, x, y):
    x = x[0]
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
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops, k_out = count_conv2d(m.key_conv, padded_x)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    total_ops, v_out = count_conv2d(m.value_conv, padded_x)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

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
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    out2 = F.softmax(out, dim=-1)
    # get the softmax count for one batch, one set of features
    total_ops = count_softmax(out[:, 0, 0, 0, :])
    # now multiply with the total groups, and width x height
    total_ops = total_ops * out[0, :, :, :, 0].numel()
    m.total_ops += torch.DoubleTensor([int(total_ops)])

    if m.adaptive_span:
        total_ops = count_adaptive_flops(m.adaptive_mask, out2, int(max_size))
        m.total_ops += torch.DoubleTensor([int(total_ops)])

    # out3 = (out2.unsqueeze(dim=2) * v_out).sum(dim=-1).view(batch, -1, height, width)
    m.total_ops += v_out.numel()


if __name__ == '__main__':
    input = torch.randn((2, 3, 32, 32))
    conv = AttentionConv(3, 16, kernel_size=3, padding=1, adaptive_span=True)
    macs, params = profile(conv, inputs=(input, ), custom_ops={AttentionConv: count_attention_flops})
    print(macs, params)
