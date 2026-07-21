from math import tau
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlgoParams:
    def __init__(self, denoiser_type='lsc', algo='ISTA', num_iter=3, activation='relu', normalize_phi=True):
        self.denoiser_type = denoiser_type
        self.algo = algo
        self.num_iter = num_iter
        self.activation = activation
        self.normalize_phi = normalize_phi



def activation_function(x, tau, activation='relu'):
    if activation == 'soft_threshold':
        return torch.sign(x) * F.relu(torch.abs(x) - tau)
    elif activation == 'leaky_relu':
        return F.leaky_relu(x - tau, negative_slope=0.125)
    else:
        return F.relu(x - tau)
  

# ==================================================================================================
# modules for unfolded sparse coding denoising
# ==================================================================================================

class Phi_op(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Define the forward linear map phi using a 1x1 convolution
        # bias must be False for the mathematical adjoint property to hold
        
        self.phi = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.init_with_dct()
        self.normalize_phi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward mapping: phi @ x"""
        return self.phi(x)

    def phiadj(self, y: torch.Tensor) -> torch.Tensor:
        """Applies the adjoint mapping: phi_adjoint @ y
        
        y should have the shape [B, out_channels, H, W]
        Returns a tensor of shape [B, in_channels, H, W]
        """
        # conv_transpose2d uses the transpose of the weight matrix
        return F.conv_transpose2d(y, self.phi.weight)

    def phiadj_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the combined mapping: phi_adjoint @ phi @ x
        
        x should have the shape [B, in_channels, H, W]
        Returns a tensor of shape [B, in_channels, H, W]
        """
        return self.phiadj(self.forward(x))
    
    def normalize_phi(self):
        """Normalizes the phi weights along dim=0 (columns/atoms mapping to patch features)"""
        with torch.no_grad():
            self.phi.weight.copy_(torch.nn.functional.normalize(self.phi.weight, p=2, dim=0))

    def init_with_dct(self):
        """Initializes phi.weight with DCT basis"""
        
        out_ch = self.phi.out_channels
        in_ch = self.phi.in_channels  # This is your token_dim
        
        # 1. Generate an identity matrix and apply the built-in orthogonal DCT
        dct_matrix = scipy.fft.dct(np.eye(max(in_ch, out_ch)), norm='ortho', axis=0)
        
        # 2. Slice the first out_channels rows as columns of phi
        dct_sliced = dct_matrix[:out_ch, :in_ch]
        
        with torch.no_grad():
            self.phi.weight.copy_(torch.from_numpy(dct_sliced).float().unsqueeze(-1).unsqueeze(-1))


# Initial patch based feature extraction
class Patch_feature_extraction(nn.Module):
    def __init__(self, patch_dim=7):
        super().__init__()

        self.patch_dim = patch_dim
 
    def forward(self, phi_op, xin):
        B, C, H, W = xin.size()

        # extract overlapping patches to create initial tokens
        x = F.unfold(xin, kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Shape: (batch_size, channels*patch_dim^2, num_patches)
        x = x.view(B, -1, H, W)  # Reshape to (batch_size, channels*patch_dim^2, H, W)

        mean_feature = torch.mean(x, dim=[2,3], keepdim=True)
        x = x - mean_feature
        x   = phi_op.phiadj(x)

        return x, mean_feature


# Final recunstruction with patches
class Patch_reconstruction(nn.Module):
    def __init__(self, patch_dim=7) -> None:
        super().__init__()

        self.patch_dim = patch_dim

        # forward pass contains patch aggregation and averaging them to reconstruct the image

    def forward(self, phi_op, x, mean_feature):
        x = phi_op(x)
        x = x + mean_feature

        B, C, H, W = x.size()
        x = x.view(B, C, H*W)  # Reshape to (batch_size, channels*patch_dim^2, H*W)

        # put all the overlapping patches and average them to reconstruct the image
        mask = F.fold(torch.ones_like(x), output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)
        x   = F.fold(x, output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Reconstruct the image from patches
        x = x / mask
        return x
  

# Canonical LISTA RNN - unfolding of proximal gradient descent
class SparseCoding_RNN(nn.Module):
    def __init__(self, train_iter=3, token_dim=128):
        super().__init__()

        self.train_iter  = train_iter
        self.gd_stepsize = nn.Parameter(torch.ones(1))

        self.tau_mlp = nn.Sequential(
            nn.Conv2d(token_dim, 32, kernel_size=1, bias=False),
            nn.Conv2d(32, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=False)
        )
        

    def forward(self, phi_op, xin, algo_params: AlgoParams):
        tau = self.tau_mlp(xin)

        x = torch.zeros_like(xin)  # Initialize x with zeros, same shape as xin
        if algo_params.algo == 'ISTA':
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                x = x - self.gd_stepsize * (phi_op.phiadj_phi(x) - xin)
                x = activation_function(x, tau, activation=algo_params.activation)

        elif algo_params.algo == 'FLIPS':
            G = - xin
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                # Compute update directions for x and G
                Dx = activation_function(x - G, tau, activation=algo_params.activation) - x
                DG = phi_op.phiadj_phi(Dx)

                # Compute FW-stepsize
                numerator   = - torch.sum(G * Dx, dim = 1, keepdim=True) 
                denominator =   torch.sum(DG * Dx, dim = 1, keepdim=True)
                gamma = torch.clamp(numerator.div_(denominator.add_(1e-4)).add_(1e-3), min=0.0, max=1.0)

                # Update x and G
                if  iter == algo_params.num_iter - 1:
                    x = x + gamma * Dx
                else:
                    x = x + gamma * Dx
                    G = G + gamma * DG

        elif algo_params.algo == 'new_ISTA':
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                x = x + activation_function(x - self.gd_stepsize*(phi_op.phiadj_phi(x) + x - xin), tau, activation=algo_params.activation)
                x = x + activation_function(x - self.gd_stepsize*(phi_op.phiadj_phi(x) + x - xin), tau, activation=algo_params.activation)
        return x
    

class LSC_denoiser(nn.Module):
    def __init__(self, patch_dim=7, token_dim=128, train_iter=3, im_color: str = 'color'):
        super().__init__()

        self.patch_dim  = patch_dim
        self.token_dim  = token_dim

        self.color_ch = 3 if im_color == 'color' else 1

        self.phi_op = Phi_op(in_channels=self.token_dim, out_channels=self.color_ch*self.patch_dim**2)

        self.feature_extractor = Patch_feature_extraction(patch_dim=self.patch_dim)
        self.rnn = SparseCoding_RNN(train_iter = train_iter, token_dim=self.token_dim)
        self.reconstructor = Patch_reconstruction(patch_dim=self.patch_dim)


    def forward(self, x, algo_params: AlgoParams):
        x, x_mean = self.feature_extractor(self.phi_op, x)
        x = self.rnn(self.phi_op, x, algo_params)
        x = self.reconstructor(self.phi_op, x, x_mean)
        return x
    


# ==================================================================================================
# Feature extraction and reconstruction modules for RNN, Conv, and Transformer denoisers
# ==================================================================================================

class ExpComp_Module(nn.Module):
    def __init__(self, in_ch, kernel_dim=5):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_ch, 4*in_ch, kernel_size=kernel_dim, bias=False, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(4*in_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):        
        return self.conv_layer(x)


class ExpComp_Block(nn.Module):
    def __init__(self, in_ch, kernel_dim=5, num_modules=5):
        super().__init__()

        self.expcomp_modules = nn.ModuleList([
            ExpComp_Module(in_ch, kernel_dim) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.expcomp_modules:
            x = module(x)
        return x
    

class Depth_Sep_Conv(nn.Module):
    def __init__(self, token_dim=128, kernel_dim=5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=kernel_dim, padding='same', bias=False, padding_mode='replicate', groups=token_dim),
            nn.Conv2d(token_dim, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.layer(x)


class Depth_Sep_Conv_Block(nn.Module):
    def __init__(self, token_dim=128, kernel_dim=5, num_modules=3):
        super().__init__()
        self.num_modules = num_modules
        self.depth_sep_modules = nn.ModuleList([
            Depth_Sep_Conv(token_dim=token_dim, kernel_dim=kernel_dim) for _ in range(num_modules)
        ])
    
    def forward(self, x):
        for module in self.depth_sep_modules:
            x = module(x)
        return x


class Conv_Feature_Extraction(nn.Module):
    def __init__(self, kernel_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.token_dim = token_dim

        self.analysis_op = nn.Sequential(
            nn.Conv2d(color_ch, 4*token_dim, kernel_size=kernel_dim, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(4*token_dim, 8, kernel_size=1, bias=False),
            ExpComp_Block(in_ch=8, kernel_dim=kernel_dim, num_modules=num_modules),
            nn.Conv2d(8, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.analysis_op(x)
        return x


class Conv_Patch_Reconstruction(nn.Module):
    def __init__(self, patch_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.patch_dim = patch_dim
        self.synthesis_op = nn.Sequential(
            Depth_Sep_Conv_Block(token_dim=token_dim, kernel_dim=5, num_modules=num_modules),
            nn.Conv2d(token_dim, color_ch*patch_dim**2, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        x = self.synthesis_op(x)
        
        B, C, H, W = x.size()
        x = x.view(B, C, H*W)  # Reshape to (batch_size, channels*patch_dim^2, H*W)

        mask = torch.ones_like(x)
        x = F.fold(x, output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Reconstruct the image from patches
        mask = F.fold(mask, output_size=(H, W), kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)
        x = x / mask

        return x
    
class Fully_Conv_Reconstruction(nn.Module):
    def __init__(self, kernel_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.synthesis_op = nn.Sequential(
            Depth_Sep_Conv_Block(token_dim=token_dim, kernel_dim=kernel_dim, num_modules=num_modules),
            nn.Conv2d(token_dim, 4*token_dim, kernel_size=1, bias=False),
            nn.Conv2d(4*token_dim, color_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.synthesis_op(x)
        return x



# ==================================================================================================
# modules for RNN denoising network
# ==================================================================================================

class Denoising_filter(nn.Module):
    def __init__(self, token_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(token_dim, 16, kernel_size=1, bias=False),
            nn.Conv2d(16, token_dim, kernel_size=1, bias=False)
        )

        self.mixer = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=9, bias=False, padding='same', padding_mode='replicate', groups=token_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.mixer(x) + x
        return x


class Denoising_RNN_Block(nn.Module):
    def __init__(self, train_iter = 3, token_dim=128):
        super().__init__()

        self.train_iter = train_iter

        self.tau_mlp = nn.Sequential(
            nn.Conv2d(token_dim, 32, kernel_size=1, bias=False),
            nn.Conv2d(32, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=False)
        )

        self.filter = Denoising_filter(token_dim=token_dim)
        
    def forward(self, xin, algo_params: AlgoParams):

        tau = self.tau_mlp(xin)
        mean_feature = torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin - mean_feature

        x = torch.zeros_like(xin)
        if algo_params.algo == 'ISTA':
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                x = activation_function(x - self.filter(x) + xin, tau, activation=algo_params.activation)

        elif algo_params.algo == 'new_ISTA':
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                x = x + activation_function(xin - self.filter(x), tau, activation=algo_params.activation)

        elif algo_params.algo == 'FLIPS':
            G = - xin
            for iter in range(algo_params.num_iter if algo_params.num_iter is not None else self.train_iter):
                # Compute update directions for x and G
                Dx = activation_function(x - G, tau, activation=algo_params.activation) - x
                DG = self.filter(Dx)

                # Compute FW-stepsize
                numerator   = - torch.sum(G * Dx, dim = 1, keepdim=True) 
                denominator =   torch.sum(DG * Dx, dim = 1, keepdim=True)
                gamma = torch.clamp(numerator.div_(denominator.add_(1e-4)).add_(1e-3), min=0.0, max=1.0)

                # Update x and G
                x = x + gamma * Dx
                G = G + gamma * DG

        x = x + mean_feature
        return x


class RNN_denoiser(nn.Module):
    def __init__(self, patch_dim=7, token_dim=128, train_iter=3, im_color: str = 'color'):
        super().__init__()

        self.color_ch = 3 if im_color == 'color' else 1
        self.patch_dim = patch_dim
        self.token_dim = token_dim

        self.feature_extractor = Conv_Feature_Extraction(kernel_dim=self.patch_dim, token_dim=self.token_dim, num_modules=1, color_ch=self.color_ch)
        self.rnn = Denoising_RNN_Block(train_iter=train_iter, token_dim=self.token_dim)
        self.reconstructor = Conv_Patch_Reconstruction(patch_dim=self.patch_dim, token_dim=self.token_dim, num_modules=1, color_ch=self.color_ch)

    def forward(self, x, algo_params: AlgoParams):
        x = self.feature_extractor(x)
        x = self.rnn(x, algo_params)
        x = self.reconstructor(x)
        return x



# ==================================================================================================
# Transformer modules for IMgae denoising
# ==================================================================================================

    
def pixel_unshuffler(x, downscale_factor):
    pad_h = (downscale_factor - (x.shape[2] % downscale_factor)) % downscale_factor
    pad_w = (downscale_factor - (x.shape[3] % downscale_factor)) % downscale_factor
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    x = F.pixel_unshuffle(x, downscale_factor=downscale_factor)

    return x


def pixel_shuffler(x, upscale_factor, im_size):
    x = F.pixel_shuffle(x, upscale_factor=upscale_factor)
    return x[:, :, :im_size[0], :im_size[1]]


class Vision_Multi_Head_Attention(nn.Module):
    def __init__(self, token_dim, att_emb_dim, att_out_dim, num_att_heads: int):
        super().__init__()
            
        self.token_dim, self.att_emb_dim, self.att_out_dim, self.num_att_heads = token_dim, att_emb_dim, att_out_dim, num_att_heads

        self.WQ = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WK = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)

        # self.WV = nn.Sequential(
        #     nn.Conv2d(self.token_dim, self.att_out_dim * self.num_att_heads, kernel_size=1, bias=False)
        #     )

        # self.linear_mixing = nn.Sequential(
        #     nn.Conv2d(self.att_out_dim * self.num_att_heads, self.token_dim, kernel_size=1, bias=False)
        #     )

    def forward(self, xin):
        B, C, H, W = xin.shape

        Q = self.WQ(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        K = self.WK(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        
        attention = K.transpose(-2, -1) @ Q / (self.att_emb_dim ** 0.5)
        attention = F.softmax(attention, dim=-1).squeeze()

        output = xin.view(B, C, H * W) @ attention
        output = output.view(B, C, H, W)

        # output = self.WV(xin).view(B, self.num_att_heads, self.att_out_dim, -1)
        # output = output @ attention
        # output = output.view(B, self.num_att_heads * self.att_out_dim, H, W)

        # output = self.linear_mixing(output)
        return output


class MHA_Vision_Transformer_module(nn.Module):
    def __init__(self, ch_dimensions: list, num_att_heads: int, pixel_shuffle_factor: int):
        """
        Args:
            ch_dimension: [token_dim, att_emb_dim, att_out_dim, mlp_middle_dim]
            num_att_heads: number of attention heads in each transformer module
        """
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Conv2d(ch_dimensions[0], ch_dimensions[3], kernel_size=1, bias=False),
            nn.Conv2d(ch_dimensions[3], ch_dimensions[0], kernel_size=1, bias=False)
        )
        
        self.pixel_shuffle_factor = pixel_shuffle_factor

        self.MH_Attention_module = Vision_Multi_Head_Attention(token_dim=ch_dimensions[0]*pixel_shuffle_factor**2, att_emb_dim=ch_dimensions[1], att_out_dim=ch_dimensions[2], num_att_heads=num_att_heads)

        

    def forward(self, x):

        x = self.mlp_head(x)

        z = pixel_unshuffler(x, downscale_factor=self.pixel_shuffle_factor)
        z = self.MH_Attention_module(z)
        z = pixel_shuffler(z, upscale_factor=self.pixel_shuffle_factor, im_size=x.shape[2:])

        return z + x
    

class MHA_Vision_Transformer_Block(nn.Module):
    def __init__(self, ch_dimensions: list, num_att_heads: int, pixel_shuffle_factor: int, num_modules: int):
        """
        Args:
            ch_dimension: [token_dim, att_emb_dim, att_out_dim, mlp_middle_dim]
            num_att_heads: number of attention heads in each transformer module
            num_modules: number of transformer modules
        """
        super().__init__()

        self.MHA_Vision_Transformer_modules = nn.ModuleList([
            MHA_Vision_Transformer_module(ch_dimensions, num_att_heads, pixel_shuffle_factor) for _ in range(num_modules)
        ])  

    def forward(self, xin):
        mean_feature =  torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin  - mean_feature
        x = torch.zeros_like(xin)
        for module in self.MHA_Vision_Transformer_modules:
            x = x + module(xin - x)
        return x + mean_feature


class MHA_Vision_Transformer_Denoiser(nn.Module):
    def __init__(self, dim_list, num_att_heads=2, num_modules=3, pixel_shuffle_factor = 2, im_color: str = 'color'):
        super().__init__()
        """
            Args:
            dim_list = [patch_dim, token_dim, att_emb_dim, att_out_dim, mlp_middle_dim]
            num_att_heads: number of attention heads in each transformer module
            num_modules: number of transformer modules
            pixel_shuffle_factor: number of pixels to be shuffled in each dimension
        """

        self.color_ch = 3 if im_color == 'color' else 1

        self.patch_dim, self.token_dim, self.att_emb_dim, self.att_out_dim, self.mlp_middle_dim = dim_list
        

        self.feature_extractor = Conv_Feature_Extraction(kernel_dim=self.patch_dim, token_dim=self.token_dim, num_modules=1, color_ch=self.color_ch)
        self.Transformer_block = MHA_Vision_Transformer_Block(ch_dimensions=dim_list[1:], num_att_heads=num_att_heads, pixel_shuffle_factor = pixel_shuffle_factor, num_modules=num_modules)
        self.reconstructor = Conv_Patch_Reconstruction(patch_dim=self.patch_dim, token_dim=self.token_dim, num_modules=1, color_ch=self.color_ch)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.Transformer_block(x)
        x = self.reconstructor(x)
        return x


# =============================================================================================================================
# Convolutional modules for denoising
# =============================================================================================================================

class Conv_Refinement_Module(nn.Module):
    def __init__(self, token_dim):
        super().__init__()

        self.conv_module = nn.Sequential(
            nn.Conv2d(token_dim, 4*token_dim, kernel_size=1, bias=False),
            nn.Conv2d(4*token_dim, 4*token_dim, kernel_size=7, bias=False, padding='same', padding_mode='replicate', groups=4*token_dim),
            nn.ReLU(),
            nn.Conv2d(4*token_dim, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_module(x)
        return x
    
class Conv_Refinement_Block(nn.Module):
    def __init__(self, token_dim, num_modules):
        super().__init__()

        self.res_modules = nn.ModuleList([
            Conv_Refinement_Module(token_dim) for _ in range(num_modules)
        ])

    def forward(self, xin, num_iter=1):
        mean = torch.mean(xin, dim = [2,3], keepdim=True)
        xin = xin - mean
        x = torch.zeros_like(xin)
        for module in self.res_modules:
            for _ in range(num_iter):
                x = x + module(xin - x)
        return x + mean
    
class Fully_Conv_Denoiser(nn.Module):
    def __init__(self, dim_list, num_modules=3, im_color: str = 'color'):
        super().__init__()

        self.color_ch = 3 if im_color == 'color' else 1
        self.patch_dim, self.token_dim = dim_list
        
        self.feature_extractor = Conv_Feature_Extraction(kernel_dim=self.patch_dim, token_dim=self.token_dim, num_modules=2, color_ch=self.color_ch)
        self.refinement_block  = Conv_Refinement_Block(token_dim=self.token_dim, num_modules=num_modules)
        self.reconstructor     = Fully_Conv_Reconstruction(kernel_dim=self.patch_dim, token_dim=self.token_dim, num_modules=2, color_ch=self.color_ch)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.refinement_block(x)
        x = self.reconstructor(x)
        return x




