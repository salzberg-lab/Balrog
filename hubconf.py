dependencies = ['torch', 'os']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def geneTCN():
    """ Pretrained Temporal Convolutional Network for bacterial gene identification."""
    
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, "weights/geneTCN.pt")
    if torch.cuda.device_count() > 0:
        state_dict = torch.load(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))

    input_channels = 21
    n_classes = 1
    kernel_size = 8
    dropout = 0.05
    hidden_units_per_layer = 25
    levels = 5
    channel_sizes = [hidden_units_per_layer] * levels
    
    model = TCN_allhidden(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    model.load_state_dict(state_dict)
    return model

def tisTCN():
    """ Pretrained Temporal Convolutional Network for bacterial translation initiation site identification."""

    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, "weights/tisTCN.pt")
    if torch.cuda.device_count() > 0:
        state_dict = torch.load(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))

    input_channels = 4
    n_classes = 1
    kernel_size = 6
    dropout = 0.05
    hidden_units_per_layer = 25
    levels = 5
    channel_sizes = [hidden_units_per_layer] * levels

    model = TCN_logit(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    model.load_state_dict(state_dict)
    return model
    
class TCN_allhidden(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_allhidden, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o_all = torch.stack([self.linear(y1[:, :, i]) for i in range(y1.shape[2])], dim=2) # return all outputs so we can select the correct index for the actual length of the non-padded sequence
        return o_all 

class TCN_logit(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_logit, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o # just need logit of last class for binary cross entropy logit loss

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Code above was sourced from https://github.com/locuslab/TCN on 03-31-20
# Bai, S., Zico Kolter, J. & Koltun, V. 
# An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 
# arXiv [cs.LG] (2018)

# Modifications and additions have been made by Markus Sommer to enable 
# classification of genetic sequence

# MIT License

# Copyright (c) 2018 CMU Locus Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
