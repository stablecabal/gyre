import torch
from basicsr.archs.rrdbnet_arch import ResidualDenseBlock, RRDBNet
from torch import nn

# Adapted in part from https://github.com/ncarraz/ESRGANplus/blob/master/codes/models/modules/block.py#L232


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device("cuda"))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualDenseBlock_Plus(ResidualDenseBlock):
    def __init__(self, num_feat=64, num_grow_ch=32, gaussian_noise=True):
        super().__init__(num_feat, num_grow_ch)

        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(num_feat, num_grow_ch)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x2 = x2 + self.conv1x1(x)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        res = x5 * 0.2 + x
        res = self.noise(res) if self.noise is not None else res

        return res


class RRDBNet_Plus(RRDBNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def replace_block(module):
            for name, child in module.named_children():
                if isinstance(child, ResidualDenseBlock):
                    module._modules[name] = ResidualDenseBlock_Plus()
                else:
                    replace_block(child)

        replace_block(self)
