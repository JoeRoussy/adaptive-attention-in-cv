import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels

        # Note– we will assume a kernel_size x kernel_size context window for applying attention.
        self.kernel_size = kernel_size
        # Once we move to adaptive attention span, this kernel_size will depend on the
        # R parameter– the softmask distance for attention

        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # Note the different usage of kernel_size for rel_w and rel_h. They are 2 one dimensional arrays
        # Reason they divide by two is that in the paper they just concat rel_h and rel_w to be the positional
        # embedding vector
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        # this is interesting!
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # the query should come from the pixel under consideration, while the keys and values should come from the
        # context window
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        #now k_out has shape (bsz, out_channels, height, width,3,3) where kernel =(3,3) (so has keys for each block which makes it easy to apply attention)

        #now we add relative height to the first half of the output channels and relative width to the second half
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        #for now suppose groups is 1, RETHINK THIS IF NOT
        #this operation just flattens the kernels in the last two dimensions (does this properly from example I did)
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        #DONT UNDERSTAND WHY HE MULTIPLIED LIKE THIS, HIS IS COMMENTED OUT
        #out = q_out * k_out
        #out = F.softmax(out, dim=-1)
        #out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        #I think way to do this is is multiply (broadcast over last dimension) then sum dim=2 (acts as dot product)
        #TO DO: Check that this still works with groups > 1 (I think may need to do a flattening after in this case
        out = (q_out*k_out).sum(dim=2)
        out2 = F.softmax(out, dim=-1)
        out3 = (out2.unsqueeze(dim=2) * v_out).sum(dim=-1).squeeze(dim=1) #Check if can condense this in one einstein

        return out3


    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


if __name__ == '__main__':
    temp = torch.randn((2, 3, 32, 32))
    conv = AttentionConv(3, 16, kernel_size=3, padding=1)
    print(conv(temp).size())
