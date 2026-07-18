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


import os
from PIL import Image


import sys





# ============================================================================
# Loading checkpoint from last training
# ============================================================================
# function to load checkpoint
def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint





# ============================================================================
# Dataloaders for Image Denoising
# ============================================================================

class DenoiserDataset(Dataset):
    """
    Dataset for Image Denoising on BSDS500.
    Supports both 'color' (RGB) and 'gray' (Grayscale) formats.
    Returns: (noisy_image, clean_image)
    """
    def __init__(self, root_dir, sigma=25, transform=None, im_color='color'):
        self.root_dir = root_dir
        # Get all image files
        self.img_names = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
        self.img_names.sort()
        self.sigma = sigma / 255.0  # Normalize noise level to [0, 1] range
        self.transform = transform
        
        # Dynamically determine PIL conversion mode: 'RGB' for 3 channels, 'L' for 1 channel
        self.mode = 'RGB' if im_color == 'color' else 'L'

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. Load Clean Image using the selected color mode
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        clean_img = Image.open(img_path).convert(self.mode)

        # 2. Apply Transform (e.g., RandomCrop for training)
        if self.transform:
            clean_img = self.transform(clean_img)
        else:
            clean_img = transforms.ToTensor()(clean_img)

        # 3. Create Noisy Version (Input)
        # torch.randn_like now automatically maps to 1 channel or 3 channels 
        # depending on the clean_img tensor dimensions
        noise = torch.randn_like(clean_img) * self.sigma
        noisy_img = clean_img + noise
        
        # Clamp to ensure pixels stay in valid [0, 1] range
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        # Normalize to [-1, 1] range
        noisy_img = (noisy_img - 0.5) / 0.5
        clean_img = (clean_img - 0.5) / 0.5

        return noisy_img, clean_img

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=16, patch_size=128, sigma=25, im_color='color'):
    """
    Returns train, validation, and test dataloaders.
    """
    
    # Training Transform: Uses RandomCrop to allow batching
    train_transform = transforms.Compose([
        transforms.RandomCrop(patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])

    # Validation Transform: Keeps ORIGINAL RESOLUTION
    val_transform = transforms.Compose([
        # transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # transoform for test set is same as val set since we want to evaluate on original resolution
    test_transform = val_transform

    train_ds = DenoiserDataset(train_dir, sigma=sigma, transform=train_transform, im_color=im_color)
    val_ds   = DenoiserDataset(val_dir, sigma=sigma, transform=val_transform, im_color=im_color)
    test_ds  = DenoiserDataset(test_dir, sigma=sigma, transform=test_transform, im_color=im_color)

    # Note: val_loader MUST have batch_size=1 to handle original resolutions
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader






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




