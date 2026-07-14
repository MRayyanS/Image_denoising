
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *




# ============================================================================
#  Some basic CNN modules used throughout
# ============================================================================


# Bottleneck convolutional blocks (without depthwise separable comnvolutions) 
""" 
- features are lifted to high dimension and shrunk back to fewer channels by pointwise convolutions
- Extremely useful in early layers of the network, keeps the model lean
"""

class ExpCompModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(ExpCompModule, self).__init__()

        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=1, bias=False),
            nn.Conv2d(middle_ch, middle_ch, kernel_size=5, padding='same', bias=False, padding_mode='replicate', groups=middle_ch),
            nn.GELU(),
            nn.Conv2d(middle_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_op(x)
        return x

class ExpCompBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules=5):
        super(ExpCompBlock, self).__init__()
        
        # Create a list of modules
        self.res_modules = nn.ModuleList([
            ExpCompModule(in_ch, middle_ch) for _ in range(num_modules)
        ])
    
    def forward(self, x):
        for res_module in self.res_modules:
            x = res_module(x)
        return x
    


class CompExpModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super().__init__()

        self.conv_op = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding='same', bias=False, padding_mode='replicate', groups=in_ch),
            nn.Conv2d(in_ch, middle_ch, kernel_size=1, bias=False),
            nn.Conv2d(middle_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_op(x)
        return x
    
class CompExpBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules=5):
        super().__init__()
        
        # Create a list of modules
        self.res_modules = nn.ModuleList([
            CompExpModule(in_ch, middle_ch) for _ in range(num_modules)
        ])
    
    def forward(self, x):
        for res_module in self.res_modules:
            x = res_module(x)
        return x


# Bottleneck modules with depthwise separable convolutions for residual learning
class DepthSepModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(DepthSepModule, self).__init__()

        # Depthwise Separable convolutional module
        self.conv_dws = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=1), nn.BatchNorm2d(middle_ch),
            nn.Conv2d(middle_ch, middle_ch, kernel_size=3, padding=1, groups=middle_ch), nn.BatchNorm2d(middle_ch),
            nn.GELU(),
            nn.Conv2d(middle_ch, in_ch, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_dws(x) + x
        return x

class DepthSepBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules):
        super(DepthSepBlock, self).__init__()

        self.res_modules = nn.ModuleList([
            DepthSepModule(in_ch, middle_ch) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.res_modules:
            x = module(x)
        return x

    


# MLP head using 1x1 convolutions
class Conv_MLP(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super().__init__()

        self.conv_mlp = nn.Sequential( nn.Conv2d(in_ch, middle_ch, kernel_size=1), nn.GELU(), nn.Conv2d(middle_ch, out_ch, kernel_size=1) )

    def forward(self, xin):
        x = self.conv_mlp(xin) + xin
        return x
    


# ConvNext like modules
class ConvNext_module(nn.Module):
    def __init__(self, in_ch_dim, conv_out_ch, kernel_dim, mlp_middle_dim):
        super().__init__()

        self.normalize  = nn.BatchNorm2d(in_ch_dim)
        self.conv_layer = nn.Conv2d( in_ch_dim, conv_out_ch, kernel_size=kernel_dim, padding='same', groups=in_ch_dim )
        self.mlp_head   = nn.Sequential( nn.Conv2d(conv_out_ch, mlp_middle_dim, kernel_size=1), nn.GELU(), nn.Conv2d(mlp_middle_dim, in_ch_dim, kernel_size=1) )

    def forward(self, xin):
        x = self.mlp_head(self.conv_layer(self.normalize(xin))) + xin
        return x



class ConvNext_Blocks(nn.Module):
    def __init__(self, ch_dimensions: list, num_blocks: int):
        """
        Args:
            ch_dimensions: [in_ch_dim, conv_out_dim, kernel_size_dim, mlp_middle_dim]
        """
        super().__init__()

        self.in_ch_dim, self.conv_out_dim, self.kernel_size_dim, self.mlp_middle_dim = ch_dimensions
        self.convnext_modules = nn.ModuleList([
            ConvNext_module(self.in_ch_dim, self.conv_out_dim, self.kernel_size_dim, self.mlp_middle_dim) for _ in range(num_blocks)
        ])


    def forward(self, x):
        
        for module in self.convnext_modules:
            x = module(x)
        
        return x


