import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import random

import os
from PIL import Image
import copy


from utils import *
from model_architectures import *


# only to supress warning messages
import warnings
# This ignores the NumPy/Torchvision compatibility warning without needing the np attribute
warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Essential global variables
# ============================================================================
np.random.seed(999)
torch.manual_seed(999)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# torch.mps.synchronize()


# ============================================================================
# LOAD appropriate DATASET and create train/val split
# ============================================================================

from dataset_loaders import get_dataloaders

# Your specific paths
TRAIN_PATH = 'datasets/BSDS500/train'
VAL_PATH   = 'datasets/BSD68'
TEST_PATH  = 'datasets/BSD68'

# Set some hyperpaprameters
batch_size = 2
patch_size = 128
noise_sigma = 25
im_color = 'gray' # 'color' or 'gray'

# options: 'im_out', 'noise_out'
denoiser_learning_mode = 'im_out'
# denoiser_learning_mode = 'noise_out'

# initialize model from a pre-learned model
pre_learned_init = True  # False or True


# create dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    train_dir=TRAIN_PATH, 
    val_dir=VAL_PATH, 
    test_dir=TEST_PATH,
    batch_size=batch_size, 
    patch_size=patch_size, 
    sigma=noise_sigma,
    im_color=im_color
)


# ============================================================================
# Build train loop, validation, and test functions
# ============================================================================


def train(model, epoch: int, learning_mode, train_loss_history: list):
    model.train()
    epoch_loss = 0.0

    for i, (noisy_im, clean_im) in enumerate(train_loader):
        noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

        # extract patches to compute patch-based loss
        B, C, H, W = clean_im.size()

        # # extract overlapping patches of clean image
        # clean_patches = F.unfold(clean_im, kernel_size=7, padding=7//2, stride=1)  # Shape: (batch_size, channels*patch_dim^2, num_patches)
        # clean_patches = clean_patches.view(B, -1, H, W)  # Reshape to (batch_size, channels*patch_dim^2, H, W)

        denoised_im = noisy_im - model(noisy_im) if learning_mode == 'noise_out' else model(noisy_im)

        # denoised_patches = F.unfold(denoised_im, kernel_size=7, padding=7//2, stride=1)  # Shape: (batch_size, channels*patch_dim^2, num_patches)
        # denoised_patches = denoised_patches.view(B, -1, H, W)  # Reshape to (batch_size, channels*patch_dim^2, H, W)

        # Calculate Loss = MSE(clean_im, denoised_im)
        # loss = criterion(clean_patches, denoised_patches)
        loss = criterion(clean_im, denoised_im)

        # backprop and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        detached_loss = loss.detach().cpu() 
        train_loss_history.append(detached_loss)
        epoch_loss += (detached_loss - epoch_loss) / (i + 1)
        
    
    # 2. Convert to a standard Python float ONLY when printing at the end
    print(f'Epoch = {epoch}')
    print(f'Train Loss: {epoch_loss:.8f}')


# function for validation
def validate(model, learning_mode: str):
    model.eval()
    
    val_loss = torch.tensor(0.0, device=device)
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im in val_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            denoised_im = noisy_im - model(noisy_im) if learning_mode == 'noise_out' else model(noisy_im)
            loss = criterion(clean_im, denoised_im)

            # recover images from normalized range to [0, 1]
            clean_im = (clean_im + 1) / 2
            denoised_im = (denoised_im + 1) / 2

            psnr = compute_PSNR(clean_im, denoised_im)

            # 3. Accumulated as tensors without blocking
            val_loss += loss
            PSNR_values.append(psnr)

        avg_loss = (val_loss / len(val_loader))
        
        # Convert the whole list to CPU/floats in one asynchronous batch operation
        PSNR_values = torch.stack(PSNR_values).cpu().tolist()
            
    return avg_loss.cpu(), PSNR_values


# function for testing
def test(model, learning_mode):
    model.eval()
    print(f'Testing with the best model on the test data')
    
    test_loss = 0.0
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im in test_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            denoised_im = model(noisy_im) if learning_mode == 'im_out' else noisy_im - model(noisy_im)
            loss = criterion(clean_im, denoised_im)

            # recover images from normalized range to [0, 1]
            clean_im = (clean_im + 1) / 2
            denoised_im = (denoised_im + 1) / 2

            psnr = compute_PSNR(clean_im, denoised_im)

            test_loss += loss.item()
            PSNR_values.append(psnr.item())
            
    avg_loss = test_loss / len(test_loader)
    return avg_loss, PSNR_values


# ============================================================================
# Training loop
# ============================================================================

if __name__ == '__main__':
    
    # define the model to be trained
    patch_dim = 9
    token_dim = 256

    middle_ch = 4

    mlp_middle_dim = 32
    num_att_heads = 1
    att_emb_dim = 6
    att_out_dim = token_dim

    num_modules = 3
    unrolled_iter = 1*np.ones(num_modules, dtype=int)
    next_update_epoch = 25

    training_model = Conv_only_denoiser(patch_dim=patch_dim, token_dim=token_dim, middle_ch=middle_ch, num_modules=num_modules, unrolled_iter=unrolled_iter, im_color=im_color).to(device)

    # load a pre-learned model if any
    if pre_learned_init == True:
        prev_checkpoint = load_checkpoint('last_model_trained.pth' , device)
        training_model.load_state_dict(prev_checkpoint['model_state_dict'])

    # count and print the number of parameters
    count_parameters(training_model)

    # define loss criterion to train the model
    criterion = nn.HuberLoss()  # nn.MSELoss()

    # printing some basic info
    print(f'Starting training... using device: {device}')
    print(f"✓ Batch size: {batch_size}, Training batches per epoch: {len(train_loader)}, Total images:{len(train_loader)*batch_size}")
    
    num_epochs = 250
    learning_rate = 0.00125   #0.00125
    
    # optimizer = optim.Adam(training_model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, min_lr=1e-5)

    optimizer = optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_loss_history  = []
    val_loss_history    = []
    
    best_mean_psnr = 0.0
    best_epoch = -1
    batches_per_epoch = len(train_loader)
    best_model_state_dict = copy.deepcopy(training_model.state_dict())
    
    for epoch in range(num_epochs):
        # 1. Train the model - and update the train loss and confusion matrix
        train(training_model, epoch, denoiser_learning_mode, train_loss_history)

        # 2. evealuate on validation data and compute val_loss, val_acc, and confusion_matrix
        val_loss, PSNR_values = validate(training_model, denoiser_learning_mode)

        val_loss_history.append(val_loss)
        mean_psnr = np.mean(PSNR_values)
        print(f'Valid Loss: {val_loss:.8f} \n')
        print(f'Validation PSNR -- mean: {mean_psnr:.2f} dB, median: {np.median(PSNR_values):.2f} dB \n')

        scheduler.step()

        if  mean_psnr > best_mean_psnr:
            best_mean_psnr = mean_psnr
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(training_model.state_dict())
            print(f'Best model weights saved with mean PSNR: {best_mean_psnr:.2f} dB at epoch: {best_epoch}')
        
        print(f'Best mean PSNR: {best_mean_psnr:.2f} dB, best PSNR at epoch = {best_epoch}')
        print('-' * 60)
        

    print(f'Training completed!')
    print('-' * 80)
    
    # Load the best model weights before evaluation and saving
    best_model = Conv_only_denoiser(patch_dim=patch_dim, token_dim=token_dim, middle_ch=middle_ch, num_modules=num_modules, unrolled_iter=unrolled_iter, im_color=im_color).to(device)

    best_model.load_state_dict(best_model_state_dict)

    # Evaluate the best model on the test set
    test_loss, test_PSNRvalues = test(best_model, denoiser_learning_mode)
    
    print('-' * 80)
    print(f'Best Val PSNR: {best_mean_psnr:.2f} dB, at epoch: {best_epoch}')
    print(f'Final Test PSNR: {np.mean(test_PSNRvalues):.2f} dB')
    print('-' * 80)


    # ============================================================================
    # Saving everything
    # ============================================================================

    # Define the path for the results file
    results_path = 'last_model_trained.pth'

    # Create a dictionary containing all the data you want to preserve
    training_results = {
        'epoch_trained': epoch + 1,
        'learning_rate': learning_rate,
        'token_dim': token_dim,
        'patch_dim': patch_dim,
        'middle_ch': middle_ch,
        'num_modules': num_modules,
        'unrolled_iter': unrolled_iter,
        'im_color': im_color,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'best_val_psnr': best_mean_psnr,
    }

    # Save everything into one file
    torch.save(training_results, results_path)

    print(f"\n✓ All training results and best model saved to: {results_path}")

    # plotting the loss curves and PSNR distribution
    plot_loss_curves(train_loss_history, val_loss_history, batches_per_epoch)



