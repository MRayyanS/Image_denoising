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
from pathlib import Path


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

import random

class PairedSpatialTransform:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        # Expects a PIL Image, returns a PyTorch Tensor
        return img

    def apply_paired(self, img):
        # 1. Synchronized Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.patch_size, self.patch_size))
        img = TF.crop(img, i, j, h, w)

        # 2. Synchronized Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)

        # 3. Synchronized Vertical Flip
        if random.random() > 0.85:
            img = TF.vflip(img)

        # 4. Strict Orthogonal Rotations (No interpolation artifacts)
        if random.random() > 0.75:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)

        return transforms.ToTensor()(img)


class DenoiserDataset(Dataset):
    """
    Dataset for Image Denoising on BSDS500.
    Supports both 'color' (RGB) and 'gray' (Grayscale) formats.
    Returns: (noisy_image, clean_image)
    """
    def __init__(self, root_dir, sigma_range=[5, 50], transform=None, im_color='color'):
        self.root_dir = root_dir
        self.img_names = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
        self.img_names.sort()

        # self.sigma = sigma / 255.0  # Normalize noise level to [0, 1] range
        self.sigma_min = sigma_range[0] / 255.0
        self.sigma_max = sigma_range[1] / 255.0

        self.transform = transform
        self.mode = 'RGB' if im_color == 'color' else 'L'

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. Load Clean Image
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        clean_img = Image.open(img_path).convert(self.mode)

        # 2. Apply Custom Paired or Standard Transform
        if self.transform:
            if isinstance(self.transform, PairedSpatialTransform):
                clean_img = self.transform.apply_paired(clean_img)
            else:
                clean_img = self.transform(clean_img)
        else:
            clean_img = transforms.ToTensor()(clean_img)

        # 3. Create Noisy Version (Input) after spatial modification
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean_img) * sigma
        noisy_img = clean_img + noise
        
        # Clamp to ensure pixels stay in valid [0, 1] range
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        # # Normalize to [-1, 1] range
        # noisy_img = 2*(noisy_img - 0.5)
        # clean_img = 2*(clean_img - 0.5)

        return noisy_img, clean_img, sigma



def im_denoising_dataloader(image_dir=None, batch_size=16, transform=None, sigma_range=[5,50], im_color='color'):
    """
    Returns train, validation, and test dataloaders.
    """

    image_ds = DenoiserDataset(image_dir, sigma_range=sigma_range, transform=transform, im_color=im_color)
    
    image_loader = DataLoader(image_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return image_loader





# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_loss_curves(train_loss_history, val_loss_history, batches_per_epoch, output_path='training_plots/loss_curves.png'):
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Loss curve plot saved with 10-epoch intervals.")




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
    plt.close()





# ============================================================================
# Save image denoising visuals
# ============================================================================

def denoise_image_patchwise(model, image, patch_size=128, stride=64, batch_size=16, num_iter=None):
    """
    Denoises an arbitrary-sized image using overlapping patches and weighted aggregation.
    
    Args:
        model: PyTorch model expecting input of shape (B, C, patch_size, patch_size)
        image: Input tensor of shape (1, C, H, W) or (C, H, W)
        patch_size: Patch height/width (128)
        stride: Step size between patch centers (e.g., 64 for 50% overlap)
        batch_size: Number of patches to pass through the model simultaneously
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
        
    B, C, H, W = image.shape
    device = image.device
    
    # 1. Pad image if smaller than patch_size or to fit exact striding
    pad_h = max(0, patch_size - H) + (stride - (H - patch_size) % stride) % stride
    pad_w = max(0, patch_size - W) + (stride - (W - patch_size) % stride) % stride
    
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
    _, _, H_pad, W_pad = image_padded.shape
    
    # 2. Extract patches using unfold -> (B, C * patch_size * patch_size, N_patches)
    patches = F.unfold(image_padded, kernel_size=patch_size, stride=stride)
    num_patches = patches.shape[-1]
    
    # Reshape patches to feed into model -> (N_patches, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)
    
    # 4. Process patches in batches
    denoised_patches = []
    model.eval()
    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_in = patches[i : i + batch_size]
            batch_out = model(batch_in, num_iter=num_iter) if num_iter is not None else model(batch_in)

            denoised_patches.append(batch_out)
            
    denoised_patches = torch.cat(denoised_patches, dim=0)
    
    # Reshape for folding -> (B, C * patch_size * patch_size, N_patches)
    denoised_patches = denoised_patches.reshape(1, num_patches, -1).permute(0, 2, 1)
    
    # Create matching weight tensor for fold normalization
    weight_patches = torch.ones_like(denoised_patches)
    
    # 5. Aggregate weighted patches back into full image shape
    output_padded = F.fold(
        denoised_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride
    )
    weight_map = F.fold(
        weight_patches, output_size=(H_pad, W_pad), kernel_size=patch_size, stride=stride
    )
    
    # Normalize by accumulated weights to compute running average
    denoised_image = output_padded / (weight_map + 1e-8)
    
    # Crop back to original image dimensions
    return denoised_image[:, :, :H, :W]



def denoise_image_tta(model, image, patch_size=128, stride=64, batch_size=16, num_iter=None):
    """
    Wraps denoise_image_patchwise with D4 Test-Time Augmentation (8 transforms:
    4 rotations x 2 flip states), averages the results, and returns the final denoised image.
    
    Args:
        model: The denoiser model.
        image: Input noisy image tensor (shape: [C, H, W] or [B, C, H, W]).
        patch_size, stride, batch_size, num_iter: Arguments for denoise_image_patchwise.
        
    Returns:
        Averaged denoised image tensor matching the input shape/device.
    """
    # Handle batch dimension robustly
    is_batched = (image.ndim == 4)
    if not is_batched:
        image = image.unsqueeze(0)  # Shape becomes [1, C, H, W]
    
    device = image.device
    b, c, h, w = image.shape
    assert b == 1, "TTA wrapper is designed for single-image batch validation/testing."

    accumulated_denoised = torch.zeros_like(image)
    num_transforms = 8

    for i in range(num_transforms):
        # 1. Apply forward transformation
        augmented = image.clone()
        
        # Rotations: 0, 90, 180, 270 degrees (k = 0, 1, 2, 3)
        k_rot = i % 4
        if k_rot > 0:
            augmented = torch.rot90(augmented, k=k_rot, dims=[-2, -1])
            
        # Flips: indices 4-7 apply a horizontal flip
        do_flip = (i >= 4)
        if do_flip:
            augmented = torch.flip(augmented, dims=[-1])
            
        # 2. Denoise the augmented patchwise image
        # Un-squeeze the batch dimension if your base function expects [C, H, W]
        denoised_aug = denoise_image_patchwise(
            model, 
            augmented.squeeze(0), 
            patch_size=patch_size, 
            stride=stride, 
            batch_size=batch_size, 
            num_iter=num_iter
        )
        
        # Ensure output is [1, C, H, W]
        if denoised_aug.ndim == 3:
            denoised_aug = denoised_aug.unsqueeze(0)
            
        # 3. Invert the transformation (Reverse order of forward pass)
        # First, undo the flip if it was applied
        if do_flip:
            denoised_aug = torch.flip(denoised_aug, dims=[-1])
            
        # Second, undo the rotation by applying 4 - k rotations
        if k_rot > 0:
            denoised_aug = torch.rot90(denoised_aug, k=(4 - k_rot), dims=[-2, -1])
            
        # 4. Accumulate results
        accumulated_denoised += denoised_aug

    # Average across all 8 variations
    final_denoised = accumulated_denoised / num_transforms

    # Restore original dimension structure
    if not is_batched:
        final_denoised = final_denoised.squeeze(0)

    return final_denoised



def evaluate_model(model, image_folder, noise_range, num_images=None, num_iter=None):
    """
    Evaluates the model on a set of images and returns the denoised images.
    """

    device = next(model.parameters()).device

    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_folder).glob(f'**/{ext}'))
    image_paths = sorted([str(p) for p in image_paths])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No valid image files found in test path '{image_folder}'.")

    if num_images is not None and len(image_paths) > num_images:
        print(f"\nFound {len(image_paths)} images. Randomly sampling {num_images} for evaluation.")
        image_paths = random.sample(image_paths, num_images)

    if len(noise_range) == 1 or noise_range[1] == noise_range[0]:
        noise_seq = noise_range[0]*np.ones(len(image_paths))
    else:
        noise_seq = np.random.uniform(noise_range[0], noise_range[1], len(image_paths))
            

    model.eval()
    psnr_values = []
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):

            img = Image.open(img_path)
            img = img.convert('L') if model.attributes['im_color'] == 'gray' else img.convert('RGB')
            clean_im = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1, C, H, W]

            # Add Gaussian noise
            noise = torch.randn_like(clean_im) * (noise_seq[idx]/255.0)
            noisy_im = torch.clamp(clean_im + noise, 0.0, 1.0)

            # Model inference
            test_patchwise = False
            test_patch_size = 64
            if model.patchsize_sensitive == True:
                test_patchwise = True
                test_patch_size = model.im_size
    
            if test_patchwise:                                
                denoised_im = denoise_image_patchwise(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=8, num_iter=num_iter)
            else:
                denoised_im = model(noisy_im, num_iter)

            psnr = compute_PSNR(clean_im, denoised_im)
            psnr_val = float(psnr.item() if torch.is_tensor(psnr) else psnr)

            psnr_values.append(psnr_val)

    return psnr_values



def visualise_denoiser_performance(model, image_tuple, noise_sigma, device, output_path):
    """Selects a random image from a random batch and plots clean, noisy, and denoised images."""
    model.eval()

    clean_im = image_tuple[0].to(device)
    noisy_im = image_tuple[1].to(device)
    
    with torch.no_grad():
        test_patchwise = False
        test_patch_size = 64
        if model.patchsize_sensitive == True:
            test_patchwise = True
            test_patch_size = model.im_size

        if test_patchwise:                                
            denoised_im = denoise_image_patchwise(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=8)
        else:
            denoised_im = model(noisy_im)

    noisy_psnr = compute_PSNR(clean_im, noisy_im)
    denoised_psnr = compute_PSNR(clean_im, denoised_im)
    
    # Convert tensors to numpy format for matplotlib imshow
    def to_numpy(tensor):
        img = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
        if img.shape[0] == 3:  # Color image (C, H, W) -> (H, W, C)
            img = img.transpose(1, 2, 0)
        elif img.shape[0] == 1:  # Grayscale image (1, H, W) -> (H, W)
            img = img.squeeze(0)
        return img

    clean_np = to_numpy(clean_im)
    noisy_np = to_numpy(noisy_im)
    denoised_np = to_numpy(denoised_im)

    # Plot the side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    axes[0].imshow(clean_np, cmap='gray' if clean_np.ndim == 2 else None)
    axes[0].set_title(f"Clean Image")
    axes[0].axis('off')

    axes[1].imshow(denoised_np, cmap='gray' if denoised_np.ndim == 2 else None)
    axes[1].set_title(f"Denoised Image (PSNR: {denoised_psnr:.2f} dB)")
    axes[1].axis('off')

    axes[2].imshow(noisy_np, cmap='gray' if noisy_np.ndim == 2 else None)
    axes[2].set_title(fr"Noisy Image ($\sigma = $ {noise_sigma:.2f}, PSNR: {noisy_psnr:.2f} dB)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved qualitative comparison image to: {output_path}")



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


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.window = self._create_window(window_size, sigma, channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.exp(torch.Tensor([-(x - window_size // 2)**2 / float(2 * sigma**2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel):
        _1d_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # Move window to the same device and dtype as images
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device).type_as(img1)

        channels = img1.size(1)
        
        # Calculate means using gaussian window
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM formula components
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-8)
        
        # Return SSIM loss (1 - mean(SSIM))
        return 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)


class L1L2Loss(nn.Module):
    def __init__(self, regularizer=0.1):
        super().__init__()

        self.L1loss = nn.L1Loss()
        self.L2loss = nn.MSELoss()

        self.regularizer = regularizer


    def forward(self, img1, img2):
        return self.L1loss(img1, img2) + self.regularizer * self.L2loss(img1, img2)
