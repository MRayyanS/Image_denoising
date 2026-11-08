import math
from symtable import Class
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path


from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ==================================================================================================
# some useful functions
# ==================================================================================================


def activation_function(x, tau, activation='relu'):
    if activation == 'soft_threshold':
        return torch.sign(x) * F.relu(torch.abs(x) - tau)
    elif activation == 'leaky_relu':
        return F.leaky_relu(x - tau, negative_slope=0.125)
    else:
        return F.relu(x - tau)



def normalize_image(image):
    """
    Normalizes an image to the range [-1, 1], given input in the range [0, 1]
    """
    return 2*image - 1.0


def renormalize_image(image):
    """
    Renormalizes an image to the range [0, 1], given input in the range [-1, 1]
    """
    image = (image + 1) / 2
    # image = (image - torch.min(image))/(torch.max(image) - torch.min(image) + 1e-6)
    image = torch.clamp(image, 0, 1)
    return image


class LayerNorm2d(nn.Module):
    """Channel-first Layer Normalization for [B, C, H, W] tensors."""
    def __init__(self, num_channels: int, affine=True):
        super().__init__()

        self.affine = affine
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        if self.affine:
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance per pixel across channels
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        
        # Normalize and apply scale & shift
        x_norm = (x - mean) / torch.sqrt(var + 1e-6)

        if self.affine:
            return self.weight * x_norm + self.bias
        else:
            return self.weight * x_norm


# ==================================================================================================
# function to extract patch and denoise fixed size patches
# ==================================================================================================


# def denoise_image_patchwise(model, image, patch_size=128, stride=64, batch_size=16, num_iter=None):
#     """
#     Denoises an arbitrary-sized image using overlapping patches and weighted aggregation.
    
#     Args:
#         model: PyTorch model expecting input of shape (B, C, patch_size, patch_size)
#         image: Input tensor of shape (1, C, H, W) or (C, H, W)
#         patch_size: Patch height/width (128)
#         stride: Step size between patch centers (e.g., 64 for 50% overlap)
#         batch_size: Number of patches to pass through the model simultaneously
#     """
#     if image.dim() == 3:
#         image = image.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
        
#     B, C, H, W = image.shape
#     device = image.device
    
#     # 1. Pad image if smaller than patch_size or to fit exact striding
#     pad_h = max(0, patch_size - H) + (stride - (H - patch_size) % stride) % stride
#     pad_w = max(0, patch_size - W) + (stride - (W - patch_size) % stride) % stride
    
#     image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
#     _, _, H_pad, W_pad = image_padded.shape
    
#     # 2. Extract patches using unfold -> (B, C * patch_size * patch_size, N_patches)
#     patches = F.unfold(image_padded, kernel_size=patch_size, stride=stride)
#     num_patches = patches.shape[-1]
    
#     # Reshape patches to feed into model -> (N_patches, C, patch_size, patch_size)
#     patches = patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)
    
#     # 4. Process patches in batches
#     denoised_patches = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(0, num_patches, batch_size):
#             batch_in = patches[i : i + batch_size]
#             batch_out = model(batch_in, num_iter=num_iter) if num_iter is not None else model(batch_in)

#             denoised_patches.append(batch_out)
            
#     denoised_patches = torch.cat(denoised_patches, dim=0)
    
#     # Reshape for folding -> (B, C * patch_size * patch_size, N_patches)
#     denoised_patches = denoised_patches.reshape(1, num_patches, -1).permute(0, 2, 1)
    
#     # Create matching weight tensor for fold normalization
#     weight_patches = torch.ones_like(denoised_patches)
    
#     # 5. Aggregate weighted patches back into full image shape
#     output_padded = F.fold(
#         denoised_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride
#     )
#     weight_map = F.fold(
#         weight_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride
#     )
    
#     # Normalize by accumulated weights to compute running average
#     denoised_image = output_padded / (weight_map + 1e-8)
    
#     # Crop back to original image dimensions
#     return denoised_image[:, :, :H, :W]


# def denoise_image_tta(model, image, patch_size=128, stride=64, batch_size=16, num_iter=None):
#     """
#     Wraps denoise_image_patchwise with D4 Test-Time Augmentation (8 transforms:
#     4 rotations x 2 flip states), averages the results, and returns the final denoised image.
    
#     Args:
#         model: The denoiser model.
#         image: Input noisy image tensor (shape: [C, H, W] or [B, C, H, W]).
#         patch_size, stride, batch_size, num_iter: Arguments for denoise_image_patchwise.
        
#     Returns:
#         Averaged denoised image tensor matching the input shape/device.
#     """
#     # Handle batch dimension robustly
#     is_batched = (image.ndim == 4)
#     if not is_batched:
#         image = image.unsqueeze(0)  # Shape becomes [1, C, H, W]
    
#     device = image.device
#     b, c, h, w = image.shape
#     assert b == 1, "TTA wrapper is designed for single-image batch validation/testing."

#     accumulated_denoised = torch.zeros_like(image)
#     num_transforms = 8

#     for i in range(num_transforms):
#         # 1. Apply forward transformation
#         augmented = image.clone()
        
#         # Rotations: 0, 90, 180, 270 degrees (k = 0, 1, 2, 3)
#         k_rot = i % 4
#         if k_rot > 0:
#             augmented = torch.rot90(augmented, k=k_rot, dims=[-2, -1])
            
#         # Flips: indices 4-7 apply a horizontal flip
#         do_flip = (i >= 4)
#         if do_flip:
#             augmented = torch.flip(augmented, dims=[-1])
            
#         # 2. Denoise the augmented patchwise image
#         # Un-squeeze the batch dimension if your base function expects [C, H, W]
#         denoised_aug = denoise_image_patchwise(
#             model, 
#             augmented.squeeze(0), 
#             patch_size=patch_size, 
#             stride=stride, 
#             batch_size=batch_size, 
#             num_iter=num_iter
#         )
        
#         # Ensure output is [1, C, H, W]
#         if denoised_aug.ndim == 3:
#             denoised_aug = denoised_aug.unsqueeze(0)
            
#         # 3. Invert the transformation (Reverse order of forward pass)
#         # First, undo the flip if it was applied
#         if do_flip:
#             denoised_aug = torch.flip(denoised_aug, dims=[-1])
            
#         # Second, undo the rotation by applying 4 - k rotations
#         if k_rot > 0:
#             denoised_aug = torch.rot90(denoised_aug, k=(4 - k_rot), dims=[-2, -1])
            
#         # 4. Accumulate results
#         accumulated_denoised += denoised_aug

#     # Average across all 8 variations
#     final_denoised = accumulated_denoised / num_transforms

#     # Restore original dimension structure
#     if not is_batched:
#         final_denoised = final_denoised.squeeze(0)

#     return final_denoised


# def evaluate_model(model, image_folder, noise_range, num_images=None, num_iter=None):
#     """
#     Evaluates the model on a set of images and returns the denoised images.
#     """

#     device = next(model.parameters()).device

#     extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
#     image_paths = []
#     for ext in extensions:
#         image_paths.extend(Path(image_folder).glob(f'**/{ext}'))
#     image_paths = sorted([str(p) for p in image_paths])

#     if len(image_paths) == 0:
#         raise FileNotFoundError(f"No valid image files found in test path '{image_folder}'.")

#     if num_images is not None and len(image_paths) > num_images:
#         print(f"\nFound {len(image_paths)} images. Randomly sampling {num_images} for evaluation.")
#         image_paths = random.sample(image_paths, num_images)

#     if len(noise_range) == 1 or noise_range[1] == noise_range[0]:
#         noise_seq = noise_range[0]*np.ones(len(image_paths))
#     else:
#         noise_seq = np.random.uniform(noise_range[0], noise_range[1], len(image_paths))
            

#     model.eval()
#     psnr_values = []
#     with torch.no_grad():
#         for idx, img_path in enumerate(image_paths):

#             img = Image.open(img_path)
#             img = img.convert('L') if model.attributes['im_color'] == 'gray' else img.convert('RGB')
#             clean_im = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1, C, H, W]

#             # Add Gaussian noise
#             noise = torch.randn_like(clean_im) * (noise_seq[idx]/255.0)
#             noisy_im = torch.clamp(clean_im + noise, 0.0, 1.0)

#             # Model inference
#             test_patchwise = False
#             test_patch_size = 64
#             if model.patchsize_sensitive == True:
#                 test_patchwise = True
#                 test_patch_size = model.im_size
    
#             if test_patchwise:                                
#                 denoised_im = denoise_image_patchwise(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=8, num_iter=num_iter)
#             else:
#                 denoised_im = model(noisy_im, num_iter)

#             psnr = compute_PSNR(clean_im, denoised_im)
#             psnr_val = float(psnr.item() if torch.is_tensor(psnr) else psnr)

#             psnr_values.append(psnr_val)

#     return psnr_values


# ==================================================================================================
# general modules for denoising
# ==================================================================================================



#---------------------------------------------------------------------------------------------------
class Conv_Patch_Feature_Extraction(nn.Module):
    def __init__(self, kernel_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.patch_dim = kernel_dim

        self.analysis_op = nn.Sequential(
            nn.Conv2d(color_ch* kernel_dim**2, token_dim, kernel_size=1, bias=False), nn.ReLU(),
            Depth_Sep_Conv_Block(token_dim=token_dim, kernel_dim=5, num_modules=num_modules)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.unfold(x, kernel_size=self.patch_dim, padding=self.patch_dim//2, stride=1)  # Shape: (batch_size, channels*patch_dim^2, num_patches)
        x = x.view(B, -1, H, W)  # Reshape to (batch_size, channels*patch_dim^2, H, W)

        x = self.analysis_op(x)
        return x


class Conv_Patch_Reconstruction(nn.Module):
    def __init__(self, patch_dim=7, token_dim=128, num_modules=3, color_ch=3):
        super().__init__()

        self.patch_dim = patch_dim
        self.synthesis_op = nn.Sequential(
            Depth_Sep_Conv_Block(token_dim=token_dim, kernel_dim=patch_dim, num_modules=num_modules),
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
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
class ExpComp_Module(nn.Module):
    def __init__(self, ch_dim: list, kernel_dim=5):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(ch_dim[0], ch_dim[1], kernel_size=kernel_dim, bias=False, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(ch_dim[1], ch_dim[0], kernel_size=1, bias=False)
        )

    def forward(self, x):        
        return self.conv_layer(x)


class ExpComp_Block(nn.Module):
    def __init__(self, ch_dim: list, kernel_dim=5, num_modules=5):
        super().__init__()

        self.expcomp_modules = nn.ModuleList([
            ExpComp_Module(ch_dim, kernel_dim) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.expcomp_modules:
            x = module(x)
        return x
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
class Depth_Sep_Conv(nn.Module):
    def __init__(self, token_dim=128, kernel_dim=5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=kernel_dim, padding='same', bias=False, padding_mode='replicate', groups=token_dim),
            nn.ReLU(),
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
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
def pixel_unshuffler(x, downscale_factor):
    pad_h = (downscale_factor - (x.shape[2] % downscale_factor)) % downscale_factor
    pad_w = (downscale_factor - (x.shape[3] % downscale_factor)) % downscale_factor
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    x = F.pixel_unshuffle(x, downscale_factor=downscale_factor)

    return x


def pixel_shuffler(x, upscale_factor, im_size):
    x = F.pixel_shuffle(x, upscale_factor=upscale_factor)
    return x[:, :, :im_size[0], :im_size[1]]
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
def self_att_spatial(X, temparature):
    """
        x is a tensor of shape [B, emb_dim, H, W]
    """
    B,C,H,W = X.size()
    X = X.view(B,C,H*W)
    attention = temparature * X.transpose(-2, -1) @ X
    attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-2)
    X = X @ attention
    X = X.view(B,C,H,W)
    return X


def self_transposed_attention(X, temparature):
    """
        X is a matrix of shape [B, N, emb_dim]
    """
    attention = temparature * X @ X.transpose(-2, -1)
    attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-1)
    return attention


def cross_att_denoising(K,Q,temparature):
    """
        x is a matrix of shape [B, emb_dim, N]
    """
    attention = temparature * K.transpose(-2, -1) @ Q
    attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-2)
    return attention
#---------------------------------------------------------------------------------------------------




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
    def __init__(self, algo='new_ISTA', activation='leaky_relu', token_dim=128):
        super().__init__()

        self.algo = algo
        self.activation = activation

        self.gd_stepsize = nn.Parameter(torch.ones(1))

        self.tau_mlp = nn.Sequential(
            nn.Conv2d(token_dim, 32, kernel_size=1, bias=False),
            nn.Conv2d(32, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, bias=False)
        )
        

    def forward(self, phi_op, xin, num_iter):
        
        tau = self.tau_mlp(xin)

        x = torch.zeros_like(xin)  # Initialize x with zeros, same shape as xin
        if self.algo == 'ISTA':
            for iter in range(num_iter):
                x = x - self.gd_stepsize * (phi_op.phiadj_phi(x) - xin)
                x = activation_function(x, tau, activation=self.activation)

        elif self.algo == 'new_ISTA':
            for iter in range(num_iter):
                x = x + activation_function(x - self.gd_stepsize*(phi_op.phiadj_phi(x) + x - xin), tau, activation=self.activation)

        return x
    


class LSC_denoiser(nn.Module):
    def __init__(self, attributes: dict):
        super().__init__()

        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = False

        self.attributes = attributes

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1
        self.patch_dim  = attributes['patch_dim']
        self.token_dim  = attributes['token_dim']

        self.phi_op = Phi_op(in_channels=self.token_dim, out_channels=self.color_ch*self.patch_dim**2)

        self.train_iter = attributes['train_iter']

        self.feature_extractor = Patch_feature_extraction(patch_dim=self.patch_dim)
        self.rnn = SparseCoding_RNN(algo=attributes['algo'], activation=attributes['activation'], token_dim=self.token_dim)
        self.reconstructor = Patch_reconstruction(patch_dim=self.patch_dim)


    def forward(self, xin, num_iter=None):
        xin = normalize_image(xin)
        x, x_mean = self.feature_extractor(self.phi_op, xin)
        x = self.rnn(self.phi_op, x, num_iter) if num_iter is not None else self.rnn(self.phi_op, x, self.train_iter)
        x = self.reconstructor(self.phi_op, x, x_mean)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x


    def evaluate(self, image_folder, noise_range, num_images, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values





# =============================================================================================================================
# Convolutional modules for denoising
# =============================================================================================================================


class Fully_Conv_Feature_Extraction(nn.Module):
    def __init__(self, kernel_dim=7, token_dim=128, num_ExpComp_modules=3, color_ch=3):
        super().__init__()

        self.token_dim = token_dim

        self.analysis_op = nn.Sequential(
            nn.Conv2d(color_ch, 4*token_dim, kernel_size=kernel_dim, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(4*token_dim, 8, kernel_size=1, bias=False),
            ExpComp_Block(ch_dim=[8, 4*8], kernel_dim=kernel_dim, num_modules=num_ExpComp_modules),
            nn.Conv2d(8, token_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.analysis_op(x)
        return x
    

class Fully_Conv_Reconstruction(nn.Module):
    def __init__(self, kernel_dim=7, token_dim=128, num_DepthSep_modules=3, color_ch=3):
        super().__init__()

        self.synthesis_op = nn.Sequential(
            Depth_Sep_Conv_Block(token_dim=token_dim, kernel_dim=kernel_dim, num_modules=num_DepthSep_modules),
            nn.Conv2d(token_dim, 4*token_dim, kernel_size=1, bias=False),
            nn.Conv2d(4*token_dim, color_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.synthesis_op(x)
        return x


class Conv_Refinement_Module(nn.Module):
    def __init__(self, token_dim, low_emb_dim):
        super().__init__()

        self.normalizer = LayerNorm2d(token_dim, affine=False)

        self.token_to_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(token_dim, low_emb_dim, kernel_size=1, bias=False, padding='same', padding_mode='replicate')
        )

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(low_emb_dim, low_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(low_emb_dim, low_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )

        self.emb_to_token = nn.Sequential(
            nn.Conv2d(low_emb_dim, token_dim, kernel_size=1, bias=False),
            nn.Conv2d(token_dim, token_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate', groups=token_dim)
        )       

    def forward(self, x):
        x = self.normalizer(x)
        x = self.token_to_emb(x)
        x = x + self.conv_mixer( x )
        x = self.emb_to_token( x )

        return x


class Conv_Refinement_Block(nn.Module):
    def __init__(self, token_dim, low_emb_dim, num_modules):
        super().__init__()

        self.refinement_modules = nn.ModuleList([
            Conv_Refinement_Module(token_dim, low_emb_dim) for _ in range(num_modules)
        ])

    def forward(self, xin, num_iter):
        mean = torch.mean(xin, dim = [2,3], keepdim=True)
        xin = xin - mean

        x = torch.zeros_like(xin)
        for i, module in enumerate(self.refinement_modules):
            for ietr in range(num_iter):
                x = x + module(x + xin)
        
        x = x + mean
        return x


class Fully_Conv_Denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()

        self.attributes = attributes

        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = False

        self.num_modules = attributes['num_modules']
        self.train_iter = attributes['train_iter']

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1
        self.patch_dim, self.token_dim, self.low_emb_dim = attributes['patch_dim'], attributes['token_dim'], attributes['low_emb_dim']
        
        self.refinement_block  = Conv_Refinement_Block(token_dim=self.token_dim, low_emb_dim=self.low_emb_dim, num_modules=self.num_modules)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=self.patch_dim, num_modules=0),
            nn.Conv2d(128, self.token_dim, kernel_size=1, bias=False)
        )

        self.reconstructor     = nn.Sequential(
            nn.Conv2d(self.token_dim, self.color_ch, kernel_size=3, bias=False, padding='same', padding_mode='replicate')
        )
    

    def forward(self, xin, num_iter=None):
        xin = normalize_image(xin)

        x = self.feature_extractor(xin)
        x = self.refinement_block(x, num_iter) if num_iter is not None else self.refinement_block(x, self.train_iter)
        x = self.reconstructor(x)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x


    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values


# ==================================================================================================
# Transformer modules for Image denoising
# ==================================================================================================

class Vision_Multi_Head_Attention(nn.Module):
    def __init__(self, token_dim, att_emb_dim, att_out_dim, num_att_heads: int):
        super().__init__()
            
        self.token_dim, self.att_emb_dim, self.num_att_heads = token_dim, att_emb_dim, num_att_heads

        self.WQ = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WK = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)

        # self.WV = nn.Sequential(
        #     nn.Conv2d(self.token_dim, self.att_out_dim * self.num_att_heads, kernel_size=1, bias=False)
        #     )


        # averaging in the embedded space
        self.linear_mixing = nn.Sequential(
            nn.Conv2d(self.att_emb_dim * self.num_att_heads, self.token_dim, kernel_size=1, bias=False)
            )

        # # averaging in the token space
        # self.linear_mixing = nn.Sequential(
        #     nn.Conv2d(self.token_dim * self.num_att_heads, self.num_att_heads*self.att_emb_dim, kernel_size=1, bias=False),
        #     nn.Conv2d(self.num_att_heads*self.att_emb_dim, self.token_dim, kernel_size=1, bias=False)
        #     )


    def forward(self, xin, xstart=None):
        B, C, H, W = xin.shape

        if xstart is not None:
            Q = self.WQ(xstart).view(B, self.num_att_heads, self.att_emb_dim, -1)
        else:
            Q = self.WQ(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        K = self.WK(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        
        attention = K.transpose(-2, -1) @ Q / (self.att_emb_dim ** 0.5)
        # attention = K.transpose(-2, -1) @ K / (self.att_emb_dim ** 0.5)

        attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-2)

        # averaging in the embedded space
        output = K @ attention
        output = self.linear_mixing(output.view(B, self.num_att_heads*self.att_emb_dim, H, W))

        # output = self.WV(xin).view(B, self.num_att_heads, self.att_out_dim, -1)
        # output = output @ attention
        # output = self.linear_mixing(output.view(B, self.num_att_heads * self.att_out_dim, H, W))

        return output


class attention_module_shuffle_before_QK(nn.Module):
    def __init__(self, in_tensor_dim, att_emb_dim, num_att_heads, pixel_shuffle_factor, attention_type):
        super().__init__()

        self.in_tensor_dim = in_tensor_dim
        self.att_emb_dim = att_emb_dim
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.num_att_heads = num_att_heads
        self.attention_type = attention_type

        self.Q_map = nn.Conv2d(in_tensor_dim*pixel_shuffle_factor**2, att_emb_dim*self.num_att_heads, kernel_size=1, bias=False)
        self.K_map = nn.Conv2d(in_tensor_dim*pixel_shuffle_factor**2, att_emb_dim*self.num_att_heads, kernel_size=1, bias=False)

        self.mixing_heads = nn.Conv2d(att_emb_dim*self.num_att_heads, in_tensor_dim*pixel_shuffle_factor**2, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x = pixel_unshuffler(x, downscale_factor=self.pixel_shuffle_factor)

        Q = self.Q_map(x)
        x = self.K_map(x)

        B, C1, H1, W1 = x.shape
        Q = Q.view(B, self.num_att_heads, self.att_emb_dim, -1)
        x = x.view(B, self.num_att_heads, self.att_emb_dim, -1)

        if self.attention_type == 'spatial':
            attention = x.transpose(-2, -1) @ Q
            attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-2)
            x = x @ attention # shape = [B, num_att_heads, att_emb_dim, HW']

        elif self.attention_type == 'channel':
            attention = Q @ x.transpose(-2, -1)
            attention = torch.sign(attention) * F.softmax(torch.abs(attention), dim=-1)
            x = attention @ x # shape = [B, num_att_heads, att_emb_dim, HW']

        x = x.view(B, C1, H1, W1)
        x = self.mixing_heads(x)
        x = pixel_shuffler(x, upscale_factor=self.pixel_shuffle_factor, im_size=[H, W])

        return x


class attention_module_shuffle_after_QK(nn.Module):
    def __init__(self, in_tensor_dim, att_emb_dim, num_att_heads, pixel_shuffle_factor, attention_type):
        super().__init__()

        self.in_tensor_dim = in_tensor_dim
        self.att_emb_dim = att_emb_dim
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.num_att_heads = num_att_heads
        self.attention_type = attention_type

        self.Q_map = nn.Conv2d(in_tensor_dim, att_emb_dim*self.num_att_heads, kernel_size=1, bias=False)
        self.K_map = nn.Conv2d(in_tensor_dim, att_emb_dim*self.num_att_heads, kernel_size=1, bias=False)

        self.mixing_heads = nn.Conv2d(att_emb_dim*self.num_att_heads, in_tensor_dim, kernel_size=1, bias=False)


    def forward(self, x):
        B, C, H, W = x.shape

        Q = self.Q_map(x)
        x = self.K_map(x)

        # x is the placeholder name for K, output, understandable from the context
        Q = pixel_unshuffler(Q, downscale_factor=self.pixel_shuffle_factor)
        x = pixel_unshuffler(x, downscale_factor=self.pixel_shuffle_factor)

        # shape of Q, x now = [B, att_heads, C, HW']
        B, C1, H1, W1 = x.shape
        Q = Q.view(B, self.num_att_heads, self.att_emb_dim*self.pixel_shuffle_factor**2, -1)
        x = x.view(B, self.num_att_heads, self.att_emb_dim*self.pixel_shuffle_factor**2, -1)

        if self.attention_type == 'spatial':
            attention = x.transpose(-2, -1) @ Q
            attention = torch.sgn(attention) * F.softmax(torch.abs(attention), dim=-2)
            x = x @ attention

        elif self.attention_type == 'channel':
            attention = Q @ x.transpose(-2, -1)
            # attention = (2*torch.sigmoid(attention) - 1) * F.softmax(torch.abs(attention), dim=-1)
            attention = torch.sgn(attention) * F.softmax(torch.abs(attention), dim=-1)

            x = attention @ x

        x = x.view(B, C1, H1, W1)
        x = pixel_shuffler(x, upscale_factor=self.pixel_shuffle_factor, im_size=[H, W])
        x = self.mixing_heads(x)

        return x


class MHA_Vision_Transformer_module(nn.Module):
    def __init__(self, in_ch_dim, conv_middle_dim, attention_config):
        """
        Args:
            attention_config = {'att_emb_dim': att_emb_dim, 'num_att_heads': num_att_heads, 'pixel_shuffle_factor': pixel_shuffle_factor, 'attention_type': attention_type, 'projection_type': projection_type}
        """
        super().__init__()

        self.att_normalizer  = LayerNorm2d(in_ch_dim, affine=False)
        self.conv_normalizer = LayerNorm2d(in_ch_dim, affine=False)
        
        if attention_config['projection_type'] == 'shuffle_before_QK':
            self.MH_Attention_module = attention_module_shuffle_before_QK(in_tensor_dim=in_ch_dim, att_emb_dim=attention_config['att_emb_dim'], num_att_heads=attention_config['num_att_heads'], pixel_shuffle_factor=attention_config['pixel_shuffle_factor'], attention_type=attention_config['attention_type'])
        elif attention_config['projection_type'] == 'shuffle_after_QK':
            self.MH_Attention_module = attention_module_shuffle_after_QK(in_tensor_dim=in_ch_dim, att_emb_dim=attention_config['att_emb_dim'], num_att_heads=attention_config['num_att_heads'], pixel_shuffle_factor=attention_config['pixel_shuffle_factor'], attention_type=attention_config['attention_type'])

        
        self.conv_token_to_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch_dim, conv_middle_dim, kernel_size=1, bias=False),
        )

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(conv_middle_dim, conv_middle_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(conv_middle_dim, conv_middle_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )

        self.conv_emb_to_token = nn.Conv2d(conv_middle_dim, in_ch_dim, kernel_size=1, bias=False)



    def forward(self, x):
        x = x + self.MH_Attention_module( self.att_normalizer(x) )

        y = self.conv_normalizer(x)
        y = self.conv_token_to_emb(y)
        y = y + self.conv_mixer(y)
        x = x + self.conv_emb_to_token( y )

        return x
    

class MHA_Vision_Transformer_Block(nn.Module):
    def __init__(self, in_ch_dim, conv_middle_dim, num_modules: int, attention_config):
        """
        Args:
            
            num_modules: number of transformer modules
            attention_config: configuration for the attention module has attributes: num_att_heads, pixel_shuffle_factor, attention_type, projection_map
        """
        super().__init__()

        self.MHA_Vision_Transformer_modules = nn.ModuleList([
            MHA_Vision_Transformer_module(in_ch_dim=in_ch_dim, conv_middle_dim=conv_middle_dim, attention_config=attention_config) for _ in range(num_modules)
        ])  

    def forward(self, xin, num_iter):
        mean_feature =  torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin  - mean_feature
        x = torch.zeros_like(xin)
        for module in self.MHA_Vision_Transformer_modules:
            for iter in range(num_iter):
                x = module(x + xin)

        x = x + mean_feature
        return x


class MHA_Vision_Transformer_Denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()
        """
            attributes = {
            'denoiser_type': denoiser_type,
            'op_mode': 'noise_est',
            'im_color': im_color,
            'attention_config': attention_config, # attention_config = {'att_emb_dim': att_emb_dim, 'num_att_heads': num_att_heads, 'pixel_shuffle_factor': pixel_shuffle_factor, 'attention_type': attention_type, 'projection_type': projection_type}
            'num_modules': num_modules,
            'train_iter': train_iter,
            'patch_dim': patch_dim,
            'token_dim': token_dim,
            'conv_middle_dim': conv_middle_dim,
            }
        """

        self.attributes = attributes
        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = False

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1

        self.attention_config = attributes['attention_config']

        self.num_modules = attributes['num_modules']
        self.train_iter = attributes['train_iter']

        self.patch_dim = attributes['patch_dim']
        self.token_dim = attributes['token_dim']
        self.conv_middle_dim = attributes['conv_middle_dim']

        # modules of the denoiser
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(128, self.token_dim, kernel_size=1, bias=False)
        )

        self.Transformer_refinement_block = MHA_Vision_Transformer_Block(in_ch_dim=self.token_dim, conv_middle_dim = self.conv_middle_dim, attention_config=self.attention_config, num_modules=self.num_modules)

        self.reconstructor     = nn.Sequential(
            nn.Conv2d(self.token_dim, 2*self.token_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2*self.token_dim, self.color_ch, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )

    def forward(self, xin, num_iter=None):
        xin = normalize_image(xin)

        x = self.feature_extractor(xin)
        x = self.Transformer_refinement_block(x, num_iter) if num_iter is not None else self.Transformer_refinement_block(x, self.train_iter)
        x = self.reconstructor(x)
        
        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)

        return x


    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
            num_iter = self.train_iter if num_iter is None else num_iter
            psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
            return psnr_values



# =============================================================================================================================
#  U-Net type denoiser
# =============================================================================================================================


class UNet_layer_refinement_block(nn.Module):
    def __init__(self, token_dim, low_emb_dim, train_iter):
        super().__init__()

        self.layer_train_iter = train_iter

        self.normalizer = LayerNorm2d(token_dim, affine=False)

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(token_dim, low_emb_dim, kernel_size=1, bias=False),
            nn.Conv2d(low_emb_dim, low_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(low_emb_dim, low_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.Conv2d(low_emb_dim, token_dim, kernel_size=1, bias=False),
            # nn.Conv2d(token_dim, token_dim, kernel_size=7, bias=False, padding='same', padding_mode='replicate', groups=token_dim)
        )


    def forward(self, xin, num_iter=None):
        num_iter = self.layer_train_iter if num_iter is None else num_iter

        mean_feature = torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin - mean_feature
        x = torch.zeros_like(xin)
        for iter in range(num_iter):
            x = x + self.conv_mixer( self.normalizer(x + xin) )

        x = x + mean_feature
        return x


class UNet_Layers(nn.Module):
    def __init__(self, in_ch_dim, low_emb_dim_list, train_iter_list, num_layers):
        super().__init__()

        self.UNet_layers = num_layers

        self.layer_refinement = UNet_layer_refinement_block(token_dim=in_ch_dim, low_emb_dim=low_emb_dim_list[0], train_iter=train_iter_list[0])

        self.conv_compress_channels = nn.Conv2d(in_ch_dim, in_ch_dim//4, kernel_size=3, bias=False, padding='same', padding_mode='replicate')
        self.conv_expand_channels   = nn.Conv2d(in_ch_dim//4, in_ch_dim, kernel_size=3, bias=False, padding='same', padding_mode='replicate')

        if num_layers > 1:
            self.lower_UNet = UNet_Layers(in_ch_dim=in_ch_dim, low_emb_dim_list=low_emb_dim_list[1:], train_iter_list=train_iter_list[1:], num_layers=num_layers-1)
        else:
            self.lower_UNet = UNet_layer_refinement_block(token_dim=in_ch_dim, low_emb_dim=low_emb_dim_list[0], train_iter=train_iter_list[0])


    def forward(self, x, num_iter):
        B, C, H, W = x.shape

        z = self.layer_refinement(x, num_iter[0])

        y = self.conv_compress_channels(z)
        y = pixel_unshuffler(y, downscale_factor=2)

        y = y + self.lower_UNet(y, num_iter[1:])

        y = pixel_shuffler(y, upscale_factor=2, im_size=[H, W])
        y = z + self.conv_expand_channels(y)

        x = x + z + y

        return x


class UNet_denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()

        self.attributes = attributes

        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = False

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1

        self.token_dim = attributes['token_dim']
        self.patch_dim = attributes['patch_dim']
        self.num_layers = attributes['num_layers']
        self.train_iter_list = attributes['train_iter_list']

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=self.patch_dim, num_modules=0),
            nn.Conv2d(128, self.token_dim, kernel_size=1, bias=False)
        )

        self.UNet_layers = UNet_Layers(in_ch_dim=self.token_dim, low_emb_dim_list=attributes['low_emb_dim_list'], train_iter_list=attributes['train_iter_list'], num_layers=self.num_layers)

        self.reconstructor = nn.Sequential(
            nn.Conv2d(self.token_dim, 2*self.token_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2*self.token_dim, self.color_ch, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, xin, num_iter=None):
        if num_iter is None or len(num_iter) != self.num_layers:
            num_iter = self.train_iter_list
        elif len(num_iter) == 1:
            num_iter = self.num_layers*[num_iter]

        xin = normalize_image(xin)
        x = self.feature_extractor(xin)
        x = self.UNet_layers(x, num_iter=num_iter)
        x = self.reconstructor(x)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x


    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values



# =============================================================================================================================
# bilinear transform modules
# =============================================================================================================================


class bilinear_transform(nn.Module):
    """
    Applies a spatial bilinear transform F_out = U * F_in * V.
    
    Args:
        in_dim (list) : Input dimension  - [in_h, in_w]
        out_dim (list): Output dimension - [out_m, out_n]
        in_h (int): Input height H
        in_w (int): Input width W
        out_m (int): Output height m
        out_n (int): Output width n
    """
    def __init__(self, in_dim: list, out_dim: list):
        super().__init__()

        # Define U (m x H) and V (W x n) as trainable parameters
        self.U = nn.Parameter(torch.Tensor(out_dim[0], in_dim[0]))
        self.V = nn.Parameter(torch.Tensor(in_dim[1], out_dim[1]))

        # Default initialization: Kaiming (He) Uniform
        self.init_kaiming()

    def init_kaiming(self):
        """Initializes U and V using Kaiming (He) Uniform initialization."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    @torch.no_grad()
    def init_fourier(self):
        """Re-initializes U and V with normalized Discrete Cosine Transform (DCT-II) bases."""

        out_m, in_h = self.U.shape
        in_w, out_n = self.V.shape

        # Construct orthogonal DCT basis matrix for U (out_m x in_h)
        m_idx = torch.arange(out_m, dtype=torch.float32).unsqueeze(1)
        h_idx = torch.arange(in_h, dtype=torch.float32).unsqueeze(0)
        u_basis = torch.cos(math.pi * m_idx * (h_idx + 0.5) / in_h)
        u_basis *= math.sqrt(2.0 / in_h)

        # Construct orthogonal DCT basis matrix for V (in_w x out_n)
        w_idx = torch.arange(in_w, dtype=torch.float32).unsqueeze(1)
        n_idx = torch.arange(out_n, dtype=torch.float32).unsqueeze(0)
        v_basis = torch.cos(math.pi * n_idx * (w_idx + 0.5) / in_w)
        v_basis *= math.sqrt(2.0 / in_w)

        # Copy data into parameters without breaking computation graph tracking
        self.U.copy_(u_basis)
        self.V.copy_(v_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Output tensor of shape [B, C, m, n]
        """

        # U @ x @ V:
        # Step 1: (out_m, in_h) @ (B, C, in_h, in_w) -> (B, C, out_m, in_w)
        # Step 2: (B, C, out_m, in_w) @ (in_w, out_n) -> (B, C, out_m, out_n)

        x = self.U @ x @ self.V  # shape = [B, C, out_m, out_n]
        # x = torch.einsum('ik, bckw, wj -> bcij', self.U, x, self.V)

        return x


    def adjoint(self, x):
        x = self.U.transpose(0,1) @ x @ self.V.transpose(0,1)
        return x
    

class BLT_refinement_module(nn.Module):
    def __init__(self, in_dim, transform_dim, pixel_shuffle_factor=2, large_context_type='BLT_conv'):
        super().__init__()

        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.large_context_type = large_context_type

        self.normalizer   = LayerNorm2d(in_dim[0], affine=False)

        self.token_to_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim[0], transform_dim[0], kernel_size=1, bias=False),
        )

        self.emb_to_token = nn.Sequential(
            nn.Conv2d(transform_dim[0], in_dim[0], kernel_size=1, bias=False),
            nn.Conv2d(in_dim[0], in_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate', groups=in_dim[0])
        )

        self.BLT     = bilinear_transform(in_dim=[dim//pixel_shuffle_factor for dim in in_dim[1:3]], out_dim=transform_dim[1:3])
        # self.inv_bili_transform = bilinear_transform_temp(in_dim=transform_dim[1:3], out_dim=[dim//pixel_shuffle_factor for dim in in_dim[1:3]])

        if large_context_type == 'BLT_conv':
            self.large_context_module = nn.Sequential(                    
                nn.Conv2d(transform_dim[0], 2*transform_dim[0], kernel_size=3, bias=False, padding='same', padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(2*transform_dim[0], transform_dim[0], kernel_size=3, bias=False, padding='same', padding_mode='replicate')
            )
        else:
            self.temparature = nn.Parameter(torch.randn(1))

        self.conv_local = nn.Sequential(
            nn.Conv2d(transform_dim[0], transform_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(transform_dim[0], transform_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, z):
        B,C,H,W = z.shape

        z = self.normalizer(z)

        z = self.token_to_emb(z)

        y = z + self.conv_local(z)

        z = pixel_unshuffler(y, downscale_factor=self.pixel_shuffle_factor)
        z = self.BLT(z)
        # z = F.relu(z)

        if self.large_context_type == 'BLT_conv':
            z = pixel_shuffler(z, upscale_factor=self.pixel_shuffle_factor, im_size=[H,W])
            z = z + self.large_context_module(z)
            z = pixel_unshuffler(z, downscale_factor=self.pixel_shuffle_factor)

        else:
            z = z + self_att_spatial(z, self.temparature)

        z = self.BLT.adjoint(z)
        z = pixel_shuffler(z, upscale_factor=self.pixel_shuffle_factor, im_size=[H,W])

        z = y + z 
        z = self.emb_to_token(z)

        return z
    

class BLT_refinement_Block(nn.Module):
    def __init__(self, in_dim, transform_dim, pixel_shuffle_factor=2, num_modules=3, large_context_type='BLT_conv'):
        super().__init__()
        self.num_modules = num_modules
        self.refinement_modules = nn.ModuleList([
            BLT_refinement_module(in_dim, transform_dim, pixel_shuffle_factor=pixel_shuffle_factor, large_context_type=large_context_type) for _ in range(num_modules)
        ])
    
    def forward(self, xin, num_iter):
        mean_feature = torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin - mean_feature
        
        x = torch.zeros_like(xin)
        for i, module in enumerate(self.refinement_modules):
            for iter in range(num_iter):
                x = x + module(xin + x)

        x = x + mean_feature
        return x
    

class BLT_Denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()

        self.attributes = attributes
        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = True

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1

        self.num_modules = attributes['num_modules']
        self.train_iter = attributes['train_iter']

        self.im_size = attributes['token_dim_list'][1]

        self.patch_dim = attributes['patch_dim']
        self.token_dim_list = attributes['token_dim_list']
        self.transform_dim_list = attributes['transform_dim_list']

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=self.patch_dim, num_modules=0),
            nn.Conv2d(128, self.token_dim_list[0], kernel_size=1, bias=False)
        )
        
        self.refinement_block  = BLT_refinement_Block(in_dim=self.token_dim_list, transform_dim=self.transform_dim_list, pixel_shuffle_factor=attributes['pixel_shuffle_factor'], num_modules=self.num_modules, large_context_type=attributes['large_context_type'])

        self.reconstructor     = nn.Sequential(
            nn.Conv2d(self.token_dim_list[0], 2*self.token_dim_list[0], kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2*self.token_dim_list[0], self.color_ch, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, xin, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        xin = normalize_image(xin)
        x = self.feature_extractor(xin)
        x = self.refinement_block(x, num_iter) if num_iter is not None else self.refinement_block(x, self.train_iter)
        x = self.reconstructor(x)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x

    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values





class transform_3D(nn.Module):
    """
    Applies a spatial bilinear transform F_out = U * F_in * V.
    
    Args:
        in_dim (list) : Input dimension  - [in_c, in_h, in_w]
        out_dim (list): Output dimension - [out_c, out_m, out_n]
        in_h (int): Input height H
        in_w (int): Input width W
        out_m (int): Output height m
        out_n (int): Output width n
    """
    def __init__(self, in_dim: list, out_dim: list, operation_mode = 'transform'):
        super().__init__()

        self.operation_mode = operation_mode  # 'compression' or 'expansion'

        # Define U (m x H) and V (W x n) as trainable parameters
        self.U = nn.Parameter(torch.Tensor(out_dim[1], in_dim[1]))
        self.V = nn.Parameter(torch.Tensor(in_dim[2], out_dim[2]))

        self.F = nn.Conv2d(in_dim[0], out_dim[0], kernel_size=1, bias=False)

        # Default initialization: Kaiming (He) Uniform
        self.init_kaiming()
        # self.init_fourier()

    def init_kaiming(self):
        """Initializes U and V using Kaiming (He) Uniform initialization."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Output tensor of shape [B, C, m, n]
        """
        
        if self.operation_mode == 'transform':
            # for canonical attention better to first project along the channel dimension during compression (doing transforms)
            x = self.F(x)  # Shape: [B, C_out, H, W]

        # U @ x @ V:
        # Step 1: (out_m, in_h) @ (B, C, in_h, in_w) -> (B, C, out_m, in_w)
        # Step 2: (B, C, out_m, in_w) @ (in_w, out_n) -> (B, C, out_m, out_n)
        x = torch.einsum('ik, bckw, wj -> bcij', self.U, x, self.V)
        
        if self.operation_mode == 'inv_transform':
            # for canonical attention better to expand later along the channel dimension during expansion (inv transforms)
            x = self.F(x)  # Shape: [B, C_out, H, W]

        return x




# =============================================================================================================================
# pixel-shuufle and transposed attention
# =============================================================================================================================

class SingleHead_Transposed_Attention(nn.Module):
    def __init__(self, token_dim = [32, 32], att_emb_dim: list = [8, 8]):
        super().__init__()
        """
            token_dim, att_emb_dim must be a tuple since each token is the entire spatial feature map
        """

        self.token_dim, self.att_emb_dim = token_dim, att_emb_dim
        
        self.Q_transform = bilinear_transform(in_dim=self.token_dim, out_dim=self.att_emb_dim)
        self.K_transform = bilinear_transform(in_dim=self.token_dim, out_dim=self.att_emb_dim)
        self.V_transform = bilinear_transform(in_dim=self.att_emb_dim, out_dim=self.token_dim)

        self.temperature = nn.Parameter(torch.randn(1))

        # self.token_to_emb_transform = bilinear_transform(in_dim=self.token_dim, out_dim=self.att_emb_dim)
        # self.emb_to_token_transform = bilinear_transform(in_dim = self.att_emb_dim, out_dim=self.token_dim)

    def forward(self, x):

        xQ = self.Q_transform(x)
        x = self.K_transform(x) 

        B,C,H,W = x.shape

        xQ = xQ.view(B,C,-1).transpose(-2,-1)
        x = x.view(B,C,-1).transpose(-2,-1)
        x = x @ cross_att_denoising(xQ, x, self.temperature)
        
        x = self.V_transform( x.transpose(-2, -1).view(B, C, H, W) )

        return x
        

class  MultiHead_Transposed_Attention(nn.Module):
    def __init__(self, tensor_dim = [16, 256, 256], att_emb_dim: list = [8, 8], num_att_heads: int = 1, pixel_shuffle_factor: int = 4):
        super().__init__()
        """
            token_dim, att_emb_dim must be a tuple since each token is the entire spatial feature map
        """

        self.num_att_heads = num_att_heads
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.att_token_dim = np.array(tensor_dim[1:], dtype=int)//pixel_shuffle_factor

        self.attention_heads = nn.ModuleList([
            SingleHead_Transposed_Attention(token_dim=self.att_token_dim, att_emb_dim=att_emb_dim) for _ in range(self.num_att_heads)
        ])

    def forward(self, x):
        # x is a matrix of shape [B, emb_dim, N]

        B,C,H,W = x.shape
        x = pixel_unshuffler(x, downscale_factor=self.pixel_shuffle_factor)

        x_out = 0
        for i, att_head in enumerate(self.attention_heads):
            x_out = x_out + att_head(x)
            
        x_out = x_out/self.num_att_heads

        x_out = pixel_shuffler(x_out, upscale_factor=self.pixel_shuffle_factor, im_size=[H,W])
    
        return x_out


class MHTA_transformer_module(nn.Module):
    def __init__(self, tensor_dim:list = [64, 256, 256], att_emb_dim:list=[8,8], pixel_shuffle_factor: int = 4, num_att_heads: int = 1, conv_emb_dim: int = 64):
        super().__init__()
        """
            tensor_dim = [ch_dim, H, W]
            att_emb_dim = [att_emb_H, att_emb_W]
        """

        self.att_normalizer  = LayerNorm2d(tensor_dim[0], affine=False)
        self.conv_normalizer = LayerNorm2d(tensor_dim[0], affine=False)
        
        self.MHTA_module = MultiHead_Transposed_Attention(tensor_dim=tensor_dim, att_emb_dim=att_emb_dim, num_att_heads=num_att_heads, pixel_shuffle_factor=pixel_shuffle_factor)

        self.conv_token_to_emb = nn.Sequential(
            nn.Conv2d(tensor_dim[0], conv_emb_dim, kernel_size=1, bias=False),
        )

        self.conv_emb_to_token = nn.Sequential(
            nn.Conv2d(conv_emb_dim, tensor_dim[0], kernel_size=1, bias=False),
        )

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(conv_emb_dim, conv_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(conv_emb_dim, conv_emb_dim, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, x):

        x = x + self.MHTA_module( self.att_normalizer(x))

        y = self.conv_normalizer(x)
        y = self.conv_token_to_emb(y)
        y = y + self.conv_mixer( y )
        x = x + self.conv_emb_to_token( y )

        return x
    

class MHTA_Transformer_Block(nn.Module):
    def __init__(self, tensor_dim:list = [16, 256,256] , att_emb_dim:list = [8,8], pixel_shuffle_factor: int = 4, num_att_heads: int = 1, conv_emb_dim: int = 128, num_transformer_modules: int = 3):
        super().__init__()
        """
            tensor_dim = [ch_dim, H, W]
            att_emb_dim = [att_emb_H, att_emb_W]
        """

        self.MHTA_transformer_modules = nn.ModuleList([
            MHTA_transformer_module(tensor_dim=tensor_dim, att_emb_dim=att_emb_dim, pixel_shuffle_factor=pixel_shuffle_factor, num_att_heads=num_att_heads, conv_emb_dim=conv_emb_dim) for _ in range(num_transformer_modules)
        ])

    def forward(self, xin, num_iter=1):
        mean_feature = torch.mean(xin, dim=[-2,-1], keepdim=True)
        xin = xin - mean_feature
        x = torch.zeros_like(xin)
        for i, module in enumerate(self.MHTA_transformer_modules):
            for iter in range(num_iter):
                x  = module(x + xin)
        return x + mean_feature
    

class MHTA_Transformer_Denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()

        self.attributes = attributes
        self.denoiser_type = attributes['denoiser_type']
        self.patchsize_sensitive = True

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1
    
        self.patch_dim = attributes['patch_dim']
        self.tensor_dim = attributes['tensor_dim']
        self.im_size = attributes['tensor_dim'][1]

        self.num_modules = attributes['num_modules']
        self.train_iter = attributes['train_iter']

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=self.patch_dim, num_modules=0),
            nn.Conv2d(128, self.tensor_dim[0], kernel_size=1, bias=False)
        )

        self.refinement_block  = MHTA_Transformer_Block(tensor_dim=self.tensor_dim, att_emb_dim=attributes['att_emb_dim'], pixel_shuffle_factor=attributes['pixel_shuffle_factor'], num_att_heads=attributes['num_att_heads'], conv_emb_dim=attributes['conv_emb_dim'], num_transformer_modules=self.num_modules)

        self.reconstructor     = nn.Sequential(
            nn.Conv2d(self.tensor_dim[0], 2*self.tensor_dim[0], kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2*self.tensor_dim[0], self.color_ch, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )
        
        
    def forward(self, xin, num_iter=None):
        xin = normalize_image(xin)
        x = self.feature_extractor(xin)
        x = self.refinement_block(x, num_iter) if num_iter is not None else self.refinement_block(x, self.train_iter)
        x = self.reconstructor(x)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x


    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values




# =============================================================================================================================
# Temporary model to experiment
# =============================================================================================================================



class bilinear_transform_temp(nn.Module):
    """
    Applies a spatial bilinear transform F_out = U * F_in * V.
    
    Args:
        in_dim (list) : Input dimension  - [in_h, in_w]
        out_dim (list): Output dimension - [out_m, out_n]
        in_h (int): Input height H
        in_w (int): Input width W
        out_m (int): Output height m
        out_n (int): Output width n
    """
    def __init__(self, in_dim: list, out_dim: list):
        super().__init__()

        # Define U (m x H) and V (W x n) as trainable parameters
        self.U = nn.Parameter(torch.Tensor(out_dim[0], in_dim[0]))
        self.V = nn.Parameter(torch.Tensor(in_dim[1], out_dim[1]))

        # Default initialization: Kaiming (He) Uniform
        self.init_kaiming()

    def init_kaiming(self):
        """Initializes U and V using Kaiming (He) Uniform initialization."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    @torch.no_grad()
    def init_fourier(self):
        """Re-initializes U and V with normalized Discrete Cosine Transform (DCT-II) bases."""

        out_m, in_h = self.U.shape
        in_w, out_n = self.V.shape

        # Construct orthogonal DCT basis matrix for U (out_m x in_h)
        m_idx = torch.arange(out_m, dtype=torch.float32).unsqueeze(1)
        h_idx = torch.arange(in_h, dtype=torch.float32).unsqueeze(0)
        u_basis = torch.cos(math.pi * m_idx * (h_idx + 0.5) / in_h)
        u_basis *= math.sqrt(2.0 / in_h)

        # Construct orthogonal DCT basis matrix for V (in_w x out_n)
        w_idx = torch.arange(in_w, dtype=torch.float32).unsqueeze(1)
        n_idx = torch.arange(out_n, dtype=torch.float32).unsqueeze(0)
        v_basis = torch.cos(math.pi * n_idx * (w_idx + 0.5) / in_w)
        v_basis *= math.sqrt(2.0 / in_w)

        # Copy data into parameters without breaking computation graph tracking
        self.U.copy_(u_basis)
        self.V.copy_(v_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Output tensor of shape [B, C, m, n]
        """

        # U @ x @ V:
        # Step 1: (out_m, in_h) @ (B, C, in_h, in_w) -> (B, C, out_m, in_w)
        # Step 2: (B, C, out_m, in_w) @ (in_w, out_n) -> (B, C, out_m, out_n)

        x = self.U @ x @ self.V  # shape = [B, C, out_m, out_n]
        # x = torch.einsum('ik, bckw, wj -> bcij', self.U, x, self.V)

        return x
    

class temp_BLT_refinement_module(nn.Module):
    def __init__(self, in_dim, transform_dim, pixel_shuffle_factor=2, large_context_type='BLT_conv'):
        super().__init__()

        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.large_context_type = large_context_type

        self.normalizer   = LayerNorm2d(in_dim[0], affine=False)

        self.token_to_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim[0], transform_dim[0], kernel_size=1, bias=False),
        )

        self.emb_to_token = nn.Sequential(
            nn.Conv2d(transform_dim[0], in_dim[0], kernel_size=1, bias=False),
            nn.Conv2d(in_dim[0], in_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate', groups=in_dim[0])
        )

        self.bili_transform     = bilinear_transform_temp(in_dim=[dim//pixel_shuffle_factor for dim in in_dim[1:3]], out_dim=transform_dim[1:3])
        self.inv_bili_transform = bilinear_transform_temp(in_dim=transform_dim[1:3], out_dim=[dim//pixel_shuffle_factor for dim in in_dim[1:3]])

        if large_context_type == 'BLT_conv':
            self.large_context_module = nn.Sequential(                    
                nn.Conv2d(transform_dim[0], 2*transform_dim[0], kernel_size=3, bias=False, padding='same', padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(2*transform_dim[0], transform_dim[0], kernel_size=3, bias=False, padding='same', padding_mode='replicate')
            )
        else:
            self.temparature = nn.Parameter(torch.randn(1))

        self.conv_local = nn.Sequential(
            nn.Conv2d(transform_dim[0], transform_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(transform_dim[0], transform_dim[0], kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, z):
        B,C,H,W = z.shape

        z = self.normalizer(z)
        z = self.token_to_emb(z)

        y = z + self.conv_local(z)

        z = pixel_unshuffler(y, downscale_factor=self.pixel_shuffle_factor)
        z = self.bili_transform(z)

        if self.large_context_type == 'BLT_conv':
            z = pixel_shuffler(z, upscale_factor=self.pixel_shuffle_factor, im_size=[H,W])
            z = z + self.large_context_module(z)
            z = pixel_unshuffler(z, downscale_factor=self.pixel_shuffle_factor)

        else:
            z = z + self_att_spatial(z, self.temparature)

        z = self.inv_bili_transform(z)
        z = pixel_shuffler(z, upscale_factor=self.pixel_shuffle_factor, im_size=[H,W])

        z = y + z 
        z = self.emb_to_token(z)
        return z
    

class temp_BLT_refinement_Block(nn.Module):
    def __init__(self, in_dim, transform_dim, pixel_shuffle_factor=2, num_modules=3, large_context_type='BLT_conv'):
        super().__init__()
        self.num_modules = num_modules
        self.refinement_modules = nn.ModuleList([
            temp_BLT_refinement_module(in_dim, transform_dim, pixel_shuffle_factor=pixel_shuffle_factor, large_context_type=large_context_type) for _ in range(num_modules)
        ])
    
    def forward(self, xin, num_iter):
        mean_feature = torch.mean(xin, dim=[2,3], keepdim=True)
        xin = xin - mean_feature
        
        x = torch.zeros_like(xin)
        for i, module in enumerate(self.refinement_modules):
            for iter in range(num_iter):
                x = x + module(xin + x)

        x = x + mean_feature
        return x
    

class Temp_Denoiser(nn.Module):
    def __init__(self, attributes):
        super().__init__()

        self.attributes = attributes
        self.patchsize_sensitive = True
        self.denoiser_type = attributes['denoiser_type']

        self.color_ch = 3 if attributes['im_color'] == 'color' else 1

        self.num_modules = attributes['num_modules']
        self.train_iter = attributes['train_iter']

        self.im_size = attributes['token_dim_list'][1]

        self.patch_dim = attributes['patch_dim']
        self.token_dim_list = attributes['token_dim_list']
        self.transform_dim_list = attributes['transform_dim_list']

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.color_ch, 128, kernel_size=self.patch_dim, bias=False, padding='same', padding_mode='replicate'),
            Depth_Sep_Conv_Block(token_dim=128, kernel_dim=self.patch_dim, num_modules=0),
            nn.Conv2d(128, self.token_dim_list[0], kernel_size=1, bias=False)
        )
        
        self.refinement_block  = temp_BLT_refinement_Block(in_dim=self.token_dim_list, transform_dim=self.transform_dim_list, pixel_shuffle_factor=attributes['pixel_shuffle_factor'], num_modules=self.num_modules, large_context_type=attributes['large_context_type'])

        self.reconstructor     = nn.Sequential(
            nn.Conv2d(self.token_dim_list[0], 2*self.token_dim_list[0], kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2*self.token_dim_list[0], self.color_ch, kernel_size=5, bias=False, padding='same', padding_mode='replicate')
        )


    def forward(self, xin, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        xin = normalize_image(xin)
        x = self.feature_extractor(xin)
        x = self.refinement_block(x, num_iter) if num_iter is not None else self.refinement_block(x, self.train_iter)
        x = self.reconstructor(x)

        x = xin - x if self.attributes['op_mode'] == 'noise_est' else x
        x = renormalize_image(x)
        return x

    def evaluate(self, image_folder, noise_range, num_images=None, num_iter=None):
        num_iter = self.train_iter if num_iter is None else num_iter
        psnr_values = evaluate_model(self, image_folder, noise_range, num_images=num_images, num_iter=num_iter)
        return psnr_values



