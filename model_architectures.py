from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from conv_modules import *

import time


# ==================================================================================================
# Patch and conv feature extraction
# ==================================================================================================

# Initial patch based feature extraction
class Patch_feature_extractor(nn.Module):
    def __init__(self, patch_dim=7, num_ch=128, im_color: str = 'color'):
        super().__init__()

        self.token_dim = 3*patch_dim**2 if im_color == 'color' else patch_dim**2
        self.patch_dim = patch_dim

        # The analysis operator that maps the unfolded patches to a lower-dimensional embedding space
        # reminiscent of 'phiadj' operator in canonical sparse coding algorithms
        self.analysis_op = nn.Sequential(
            nn.Conv2d(self.token_dim, num_ch, kernel_size=1, padding='same', bias=False)
        )
 
    def forward(self, xin):
        B, C, H, W = xin.size()

        # extract overlapping patches to create initial tokens
        x = F.unfold(xin, kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Shape: (batch_size, channels*patch_dim^2, num_patches)
        x = x.view(B, -1, H, W)  # Reshape to (batch_size, channels*patch_dim^2, H, W)

        x = self.analysis_op(x)

        return x


# Final recunstruction with patches
class Patch_reconstructor(nn.Module):
    def __init__(self, num_ch=128, patch_dim=7, im_color: str = 'color') -> None:
        super().__init__()

        self.token_dim = 3*patch_dim**2 if im_color == 'color' else patch_dim**2
        self.patch_dim = patch_dim

        # reminiscent of the 'phi' operator in canonical sparse coding algorithms that maps the embedded features back to the patch space
        self.synthesis_op = nn.Sequential(
            nn.Conv2d(num_ch, self.token_dim, kernel_size=1, bias=False)
        )

        # forward pass contains patch aggregation and averaging them to reconstruct the image

    def forward(self, x):
        x = self.synthesis_op(x)

        B, C, H, W = x.size()
        x = x.view(B, C, H*W)  # Reshape to (batch_size, channels*patch_dim^2, H*W)

        # put all the overlapping patches and average them to reconstruct the image
        mask = F.fold(torch.ones_like(x), output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)
        x   = F.fold(x, output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Reconstruct the image from patches

        return x / mask
    


# Convolutional feature extraction and reconstruction blocks for image denoising
class Conv_feature_extractor(nn.Module):
    def __init__(self, token_dim=64, patch_dim=7, im_color: str = 'color'):
        super(Conv_feature_extractor, self).__init__()

        self.in_ch = 3 if im_color == 'color' else 1

        # Super crucial to have larger filter size in the first layer to capture more context
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(32, 64, kernel_size=3, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(64, token_dim, kernel_size=1, bias=False)
            # ExpCompBlock(in_ch=token_dim, middle_ch=128, num_modules=2)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


# Final Convolutional reconstruction block
class Conv_reconstructor(nn.Module):
    def __init__(self, num_ch=128, patch_dim = 7, im_color: str = 'color') -> None:
        super().__init__()

        self.out_ch = 3 if im_color == 'color' else 1

        self.reconstructor = nn.Sequential(
            # final linear operator
            nn.Conv2d(num_ch, num_ch, kernel_size=1, bias=False), 
            # depthwise conv replacing  averaging over patches 
            nn.Conv2d(num_ch, num_ch, kernel_size=patch_dim, padding='same', groups=num_ch, bias=False, padding_mode='replicate'),
            # final output layer to get back to 3 channels
            nn.Conv2d(num_ch, 3, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.reconstructor(x)
        return x







# ==================================================================================================
#  Conv only models for Image restoration
# ==================================================================================================


class Conv_refinement_module(nn.Module):
    def __init__(self, token_dim=64, middle_ch=128):
        super().__init__()

        self.conv_component = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=5, padding='same', bias=False, padding_mode='replicate', groups=token_dim),
            nn.Conv2d(token_dim, middle_ch, kernel_size=1, bias=False),
            # nn.GELU(),
            # nn.Conv2d(middle_ch, middle_ch, kernel_size=7, padding='same', bias=False, padding_mode='replicate', groups=middle_ch),
            nn.Conv2d(middle_ch, token_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=5, padding='same', bias=False, padding_mode='replicate', groups=token_dim)
        )

        # self.mixer = nn.Sequential(
        #     nn.Conv2d(2*token_dim, token_dim, kernel_size=1, bias=False)
        # )
    
    def forward(self, xin, num_iter):
        x = xin
        for _ in range(num_iter):
            x = self.conv_component(x)
        
        # x = torch.cat([x, xin], dim=1)
        # x = self.mixer(x)
        return x 

    
class Conv_refinement_Block(nn.Module):
    def __init__(self, token_dim=64, middle_ch=128, num_modules = 3, unrolled_iter: list = [1, 1, 1]):
        super().__init__()

        self.num_modules = num_modules
        self.unrolled_iter = unrolled_iter

        self.refinement_modules = nn.ModuleList([
            Conv_refinement_module(token_dim=token_dim, middle_ch=middle_ch) for _ in range(num_modules)
        ])

    def forward(self, xin, num_iter = None):
        x = xin
        s = torch.zeros_like(x).to(x.device)
        for m, module in enumerate(self.refinement_modules):
            s = s + module(x, num_iter[m] if num_iter is not None else self.unrolled_iter[m])  
            x = xin - s
        return s


class Conv_only_denoiser(nn.Module):
    def __init__(self, patch_dim=7, token_dim=64, middle_ch = 128, num_modules=3, unrolled_iter = [1, 1, 1], im_color: str = 'color'):
        super().__init__()

        self.num_modules = num_modules
        self.unrolled_iter = unrolled_iter

        self.feature_extractor = Conv_feature_extractor(token_dim=token_dim, patch_dim=patch_dim, im_color=im_color)
        self.refinement_block = Conv_refinement_Block(token_dim=token_dim, middle_ch=middle_ch, num_modules=num_modules, unrolled_iter = unrolled_iter,)
        self.reconstructor = Patch_reconstructor(num_ch=token_dim, patch_dim=patch_dim, im_color=im_color)

    def forward(self, x, num_iter = None):
        x = self.feature_extractor(x)
        x = self.refinement_block(x, num_iter = num_iter if num_iter is not None else self.unrolled_iter)
        x = self.reconstructor(x)

        return x






# ==================================================================================================
# Transformer modules for Image restoration
# ==================================================================================================


# Transposed Attention module as in "Restormer"
class MH_TransposedAttention(nn.Module):
    def __init__(self, token_dim=32, num_heads=2):
        super().__init__()

        self.num_channels = token_dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.WQ = nn.Sequential(
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, stride=4), nn.GELU(),
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, stride=4)
            )
        
        self.WK = nn.Sequential(
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, stride=4), nn.GELU(),
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, stride=4)
            )

        self.WV = nn.Sequential(
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, padding='same', padding_mode='replicate'), nn.GELU(),
            nn.Conv2d(num_heads*token_dim, num_heads*token_dim, kernel_size=5, bias=False, groups=num_heads*token_dim, padding='same', padding_mode='replicate')
            )

        self.linear_mixing = nn.Conv2d(token_dim*num_heads, token_dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x shape: [B, C, H, W] -> [1, 32, 400, 380]
        B, C, H, W = x.shape

        # Generate Q, K, V
        Q = self.WQ(x).view(B, self.num_heads, self.num_channels, -1)
        K = self.WK(x).view(B, self.num_heads, self.num_channels, -1)
        V = self.WV(x).view(B, self.num_heads, self.num_channels, -1)

        # Transposed Attention Matrix multiplication: [C, HW] x [HW, C] -> [C, C]
        # Extremely lightweight: 32x32 matrix calculation!
        attn = (Q @ K.transpose(-2, -1)) * self.temperature / ((H*W) ** 0.5)
        attn = attn.softmax(dim=-1)

        # Apply weights to values: [C, C] x [C, HW] -> [C, HW]
        out = attn @ V
        
        del attn, Q, K, V

        # Reshape back to original image dimensions
        out = out.reshape(B, self.num_heads * self.num_channels, H, W)
        out = self.linear_mixing(out)
        return out


class MHTA_Transformer_module(nn.Module):
    def __init__(self, token_dim, num_att_heads, mlp_middle_dim):
        super().__init__()

        self.multi_head_attention = MH_TransposedAttention(token_dim=token_dim, num_heads=num_att_heads)

        self.mlp_head = nn.Sequential(
            nn.Conv2d(token_dim, mlp_middle_dim, kernel_size=1, bias=False), nn.GELU(), nn.Conv2d(mlp_middle_dim, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, xin, num_iter):
        for iter in range(num_iter):
            x = self.multi_head_attention(xin) + xin
            x = self.mlp_head(x) + x
        return x + xin
    

class MHTA_Transformer_Blocks(nn.Module):
    def __init__(self, token_dim: int, num_att_heads: int, mlp_middle_dim: int, num_modules: int, unrolled_iter: list):
        super().__init__()

        self.unrolled_iter = unrolled_iter
        self.MHTA_transformer_modules = nn.ModuleList([
            MHTA_Transformer_module(token_dim, num_att_heads, mlp_middle_dim) for m in range(num_modules)
        ])

    def forward(self, xin, num_iter = None):
        x = xin
        for m, module in enumerate(self.MHTA_transformer_modules):
            x = module(x, num_iter[m] if num_iter is not None else self.unrolled_iter[m])
        return x + xin


class MHTA_Transformer_Denoiser(nn.Module):
    def __init__(self, token_dim: int = 32, mlp_middle_dim: int = 64, num_att_heads: int = 1, num_modules: int = 3, unrolled_iter = [1, 1, 1], im_color: str = 'color'):
        """
        Args:
            token_dim: number of channels in the input image
            mlp_middle_dim: number of channels in the middle of the MLP block
            num_att_heads: number of attention heads in each transformer module
            num_modules: number of transformer modules
            num_iter: number of iterations for the transformer module
        """
        super().__init__()
        self.token_dim, self.mlp_middle_dim = token_dim, mlp_middle_dim
        self.num_att_heads = num_att_heads
        self.num_modules = num_modules
        self.unrolled_iter = unrolled_iter

        self.feature_extractor = Conv_feature_extractor(token_dim=self.token_dim, patch_dim=7, im_color=im_color)
        self.refinement_block = MHTA_Transformer_Blocks(self.token_dim, self.num_att_heads, self.mlp_middle_dim, self.num_modules, self.unrolled_iter)
        self.reconstructor = Patch_reconstructor(num_ch=self.token_dim, patch_dim=7, im_color='color')

    def forward(self, x, num_iter = None):
        x = self.feature_extractor(x)
        x = self.refinement_block(x, num_iter)
        x = self.reconstructor(x)
        return x
    



# classical attention based transformers
class MH_Attention_Vision_module(nn.Module):
    def __init__(self, token_dim, att_emb_dim, att_out_dim, num_att_heads: int):
        super().__init__()

        self.token_dim, self.att_emb_dim, self.att_out_dim = token_dim, att_emb_dim, att_out_dim
        self.num_att_heads = num_att_heads

        self.temperature = nn.Parameter(torch.ones(num_att_heads, 1, 1))

        self.WQ = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WK = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WV = nn.Conv2d(self.token_dim, self.att_out_dim * self.num_att_heads, kernel_size=1, bias=False)

        self.linear_mixing = nn.Conv2d(self.att_out_dim * self.num_att_heads, self.token_dim, kernel_size=1, bias=False)

    def forward(self, xin):
        B, C, H, W = xin.shape

        Q = self.WQ(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        K = self.WK(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        V = self.WV(xin).view(B, self.num_att_heads, self.att_out_dim, -1)

        attention = K.transpose(-2, -1) @ Q / (self.temperature * (self.att_emb_dim ** 0.5) )
        attention = F.softmax(attention, dim=-1)

        output = V @ attention
        output = output.view(B, self.num_att_heads * self.att_out_dim, H, W)

        output = self.linear_mixing(output)
        return output

# multi head attention blocks
class MHA_Transformer_Vision_module(nn.Module):
    def __init__(self, ch_dimensions: list, num_att_heads: int):
        """
        Args:
            ch_dimensions: [in_ch_dim, att_emb_ch, att_out_dim, mlp_middle_dim]
        """
        super().__init__()

        self.num_attention_heads = num_att_heads
        self.in_ch_dim, self.att_emb_dim, self.att_out_dim, self.mlp_middle_dim = ch_dimensions

        # normalization ---> multiple attention heads ---> mixing
        self.normalize_att = nn.BatchNorm2d(self.in_ch_dim)
        self.multi_head_attention = MH_Attention_Vision_module(self.in_ch_dim, self.att_emb_dim, self.att_out_dim, num_att_heads)

        # normalization ---> mlp head
        self.normalize_mlp = nn.BatchNorm2d(self.in_ch_dim)
        self.mlp_head = nn.Sequential(
            nn.Conv2d(self.in_ch_dim, self.mlp_middle_dim, kernel_size=1), nn.GELU(), nn.Conv2d(self.mlp_middle_dim, self.in_ch_dim, kernel_size=1)
            )

    
    def forward(self, xin):
        # forward pass for normalization ---> attention head
        x = self.multi_head_attention(self.normalize_att(xin)) + xin

        # forward pass for the mlp head
        x = self.mlp_head(self.normalize_mlp(x)) + x
        return x + xin
    

class MHA_Transformer_Blocks(nn.Module):
    def __init__(self, ch_dimensions: list, num_att_heads: int, num_modules: int):
        """
        Args:
            ch_dimension: [in_ch_dim, att_emb_dim, att_out_dim, mlp_middle_dim]
            num_att_heads: number of attention heads in each transformer module
            num_modules: number of transformer modules
        """
        super().__init__()

        self.mha_transformer_modules = nn.ModuleList([
            MHA_Transformer_Vision_module(ch_dimensions, num_att_heads) for _ in range(num_modules)
        ])

    def forward(self, xin):
        x = xin
        for module in self.mha_transformer_modules:
            x = module(x)
        return x + xin


class MHA_Transformer_Denoiser(nn.Module):
    def __init__(self, patch_dim=7, token_dim=32, num_att_heads=32, att_emb_dim=8, att_out_dim=32, M=10, mlp_middle_dim=64, num_modules=3, im_color: str = 'color'):
        super().__init__()
        
        self.feature_extractor = Conv_feature_extractor(token_dim, patch_dim=patch_dim, im_color=im_color)
        self.rnn = MHA_Transformer_Blocks([token_dim, att_emb_dim, att_out_dim, mlp_middle_dim], num_att_heads, num_modules)
        self.reconstructor = Patch_reconstructor(token_dim, patch_dim=patch_dim, im_color=im_color)

    def forward(self, x):
        features   = self.feature_extractor(x)
        rnn_output = self.rnn(features)
        recon_img  = self.reconstructor(rnn_output)
        return recon_img



