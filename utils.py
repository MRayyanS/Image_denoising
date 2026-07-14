import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys


# function to load checkpoint
def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

# ============================================================================
# Build Model - CNN based
# ============================================================================

# canonical vanilla Conv2d block and modules like in VGGnet
class ConvModule(nn.Module):
    def __init__(self, in_ch):
        super(ConvModule, self).__init__()

        self.convmodule = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU()
        )

    def forward(self, x):
        x = self.convmodule(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, num_blocks):
        super(ConvBlock, self).__init__()

        self.conv_modules = nn.ModuleList([
            ConvModule(in_ch) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        return x


# residual blocks like original ResNet paper
class VanillaResModule(nn.Module):
    def __init__(self, in_ch):
        super(VanillaResModule, self).__init__()

        self.VanResModule = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        x = self.VanResModule(x) + x
        return x

class VanillaResBlock(nn.Module):
    def __init__(self, in_ch, num_modules):
        super(VanillaResBlock, self).__init__()

        self.resModules = nn.ModuleList([
            VanillaResModule(in_ch) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.resModules:
            x = module(x)
        return x


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_loss_curves(train_loss_history, val_loss_history, batches_per_epoch):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    train_x = np.arange(len(train_loss_history))
    val_x = np.arange(1, len(val_loss_history) + 1) * batches_per_epoch - 1
    
    # Plotting
    ax.plot(train_x, train_loss_history, label='Training Loss', alpha=0.7, linewidth=0.8)
    ax.plot(val_x, val_loss_history, label='Validation Loss', 
            marker='o', markersize=5, linewidth=2, color='red')
    
    ax.set_xlabel('Mini-batch Number', fontsize=12, labelpad=5)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- SECONDARY X-AXIS FOR EPOCHS ---
    ax2 = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
    ax2.spines['bottom'].set_position(('outward', 35)) 
    
    # Generate ticks every 50 epochs
    # We use step=50 in arange, and multiply by batches_per_epoch to find the x-position
    total_epochs = len(val_loss_history)
    epoch_indices = np.arange(0, total_epochs + 1, 50)
    
    # Ensure the last epoch is always included if it's not a multiple of 50
    if total_epochs not in epoch_indices:
        epoch_indices = np.append(epoch_indices, total_epochs)
        
    epoch_ticks = epoch_indices * batches_per_epoch
    
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels([str(i) for i in epoch_indices])
    ax2.set_xlabel('Epoch (Every 50)', fontsize=12)

    # Add vertical lines only at the 10-epoch marks to keep the plot clean
    for tick in epoch_ticks:
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    print("\n✓ Loss curve plot saved with 10-epoch intervals.")
    plt.show()



# fucntion ot plot PSNR values
def plot_psnr_CDF(psnr_list: list, title: str, save_path):
    plt.figure(figsize=(10, 6)) 
    plt.hist(psnr_list, bins='auto', cumulative=True, density=True, 
         histtype='step', color='green', linewidth=1.5)

    plt.title(f'{title}', fontsize=14, fontweight='bold')
    plt.xlabel('PSNR (dB)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')





# ============================================================================
# Print the parameters of the model
# ============================================================================

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




# ============================================================================
# Print the log into a file
# ============================================================================

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensures real-time writing to the file
    def flush(self):
        for f in self.files:
            f.flush()











# ============================================================================
# Guassian noise and blur for data augmentation
# ============================================================================

# Custom Gaussian Noise transform with dynamic std
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1, p=0.95):
        self.mean = mean
        self.std = std
        self.p = p
        self.current_std = std
    
    def set_std(self, std):
        self.current_std = std
    
    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.current_std + self.mean
            return tensor + noise
        return tensor
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.current_std}, p={self.p})'



# Custom Gaussian Blur transform with dynamic sigma
class AddGaussianBlur:
    def __init__(self, kernel_size=3, sigma=1.0, p=0.95):

        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.sigma = sigma
        self.p = p
        self.current_sigma = sigma
    
    def set_sigma(self, sigma):
        self.current_sigma = sigma
    
    def __call__(self, img):
        if torch.rand(1).item() < self.p and self.current_sigma > 0:
            return TF.gaussian_blur(img, self.kernel_size, [self.current_sigma])
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.current_sigma}, p={self.p})'



# ============================================================================
# Some functions for image processing
# ============================================================================

def compute_PSNR(noisy_im, clean_im, max_val=1.0, eps=1e-10):
    """
    PSNR(noisy_im, clean_im) = 10log10(peak signal power / noise power)
    peak signal power = maximum pixel value in clean image (=1, since pixel values are normalized to [0, 1])
    noise power = mean_[i,j] | clean_im[i,j] - noisy_im[i,j] |^2 
    """

    noise_power = torch.mean((clean_im - noisy_im) ** 2)
    if noise_power == 0:
        return torch.tensor(float('100'), device=clean_im.device)
    return 10.0 * torch.log10((max_val ** 2) / (noise_power + eps))




