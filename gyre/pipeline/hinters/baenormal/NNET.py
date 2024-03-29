# Originally from https://github.com/baegwangbin/surface_normal_uncertainty/
# Distributed under the MIT license

# Changes:
# - Imports changed to relative

import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules.decoder import Decoder
from .submodules.encoder import Encoder


class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(args)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        return self.decoder.parameters()

    def forward(self, img, **kwargs):
        return self.decoder(self.encoder(img), **kwargs)
