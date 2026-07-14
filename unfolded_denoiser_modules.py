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
  


# denoising filter very optimal
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
    def __init__(self, patch_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.token_dim = token_dim

        self.analysis_op = nn.Sequential(
            nn.Conv2d(color_ch, 128, kernel_size=patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=5, num_modules=num_modules),
            nn.Conv2d(128, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.analysis_op(x)
        return x




class Conv_Patch_Reconstructor(nn.Module):
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

        self.feature_extractor = Conv_Feature_Extraction(patch_dim=self.patch_dim, token_dim=self.token_dim, num_modules=2, color_ch=self.color_ch)
        self.rnn = Denoising_RNN_Block(train_iter=train_iter, token_dim=self.token_dim)
        self.reconstructor = Conv_Patch_Reconstructor(patch_dim=self.patch_dim, token_dim=self.token_dim, num_modules=2, color_ch=self.color_ch)

    def forward(self, x, algo_params: AlgoParams):
        x = self.feature_extractor(x)
        x = self.rnn(x, algo_params)
        x = self.reconstructor(x)
        return x








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
    



