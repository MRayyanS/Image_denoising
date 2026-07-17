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
seed_num = 1000
np.random.seed(seed_num)
torch.manual_seed(seed_num)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# torch.mps.synchronize()



# ============================================================================
# LOAD appropriate DATASET and create train/val split
# ============================================================================

from dataset_loaders import get_dataloaders

# Your specific paths
TRAIN_PATH = 'datasets/BSDS500/train'
VAL_PATH   = 'datasets/BSD68'
TEST_PATH  = 'datasets/Set12'

# Set some hyperpaprameters
batch_size = 1
patch_size = 128
noise_sigma = 25
im_color = 'gray' # 'color' or 'gray'

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

def train(model : Any, algo_params: AlgoParams, epoch: int, learning_mode, train_loss_history: list):
    model.train()
    epoch_loss = 0.0

    for i, (noisy_im, clean_im) in enumerate(train_loader):
        noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)
        
        denoised_im = model(noisy_im, algo_params) if learning_mode == 'im_out' else noisy_im - model(noisy_im, algo_params)
        
        # Calculate Loss = MSE(clean_im, denoised_im)
        MSEloss = criterion(clean_im, denoised_im)
        loss  = MSEloss

        # backprop and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # normalize columns of phi if required
        model.phi_op.normalize_phi() if algo_params.normalize_phi and algo_params.denoiser_type == 'lsc' else None

        # ------------------------------------        
        train_loss_history.append(loss.item())
        epoch_loss += (loss.item() - epoch_loss)/(i+1)
    
    print(f'Epoch = {epoch}')
    print(f'Train Loss: {epoch_loss:.8f}')


# function for validation
def validate(model : Any, algo_params: AlgoParams, learning_mode: str):
    model.eval()
    
    val_loss = 0.0
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im in val_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            denoised_im = model(noisy_im, algo_params) if learning_mode == 'im_out' else noisy_im - model(noisy_im, algo_params)
            loss = criterion(clean_im, denoised_im)

            # recover images from normalized range to [0, 1]
            clean_im = (clean_im + 1) / 2
            denoised_im = (denoised_im + 1) / 2

            psnr = compute_PSNR(clean_im, denoised_im)

            val_loss += loss.item()
            PSNR_values.append(psnr.item())

        avg_loss = val_loss / len(val_loader)
            
    return avg_loss, PSNR_values


# function for testing
def test(model: Any, algo_params: AlgoParams, learning_mode: str):
    model.eval()
    print(f'Testing with the best model on the test data')
    
    test_loss = 0.0
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im in test_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            denoised_im = model(noisy_im, algo_params) if learning_mode == 'im_out' else noisy_im - model(noisy_im, algo_params)
            loss = criterion(clean_im, denoised_im)

            # recover images from normalized range to [0, 1]
            clean_im = (clean_im + 1) / 2
            denoised_im = (denoised_im + 1) / 2

            psnr = compute_PSNR(clean_im, denoised_im)

            test_loss += loss.item()
            PSNR_values.append(psnr.item())
            
    avg_loss = test_loss / len(test_loader)
    return avg_loss, PSNR_values



@torch.no_grad()
def profile_unrolled_denoising_model(model: Any, algo_params, sample_batch_loader, device):
    """
    Profiles the relative ratio of the remaining features' L1-norm to the 
    total L1-norm across individual pixel/patch locations.
    """
    model.eval()
    
    # 1. Extract the dictionary matrix Phi
    phi_tensor = model.phi_op.phi.weight.squeeze(-1).squeeze(-1) # [out_ch, token_dim]
    gram = phi_tensor.t() @ phi_tensor
    gram_cpu = gram.cpu()
    eigenvalues_cpu = torch.linalg.eigvalsh(gram_cpu)
    max_ev = eigenvalues_cpu[-1].item()
    
    # 2. Accumulators for localized ratio statistics
    all_ratios_means = []
    all_ratios_vars = []
    
    all_dominant_indices = []
    mean_residual_val = 0.0
    
    total_elements = 0
    dead_elements = 0
    batch_count = 0
    
    for noisy_im, clean_im in sample_batch_loader:
        noisy_im = noisy_im.to(device)
        
        # Pass through feature extractor
        xin, x_mean = model.feature_extractor(model.phi_op, noisy_im)
        
        # Track threshold and residual 
        mean_residual_val += torch.abs(xin).mean().item()
        
        # Get final sparse coefficients [B, token_dim, H, W]
        y_latent = model.rnn(model.phi_op, xin, algo_params)
        
        # Compute absolute values
        abs_y = torch.abs(y_latent)
        
        # Find maximum absolute feature value along the channel dimension (dim=1)
        max_feature, max_feature_idx = torch.max(abs_y, dim=1)
        all_dominant_indices.append(max_feature_idx.detach().cpu())
        
        # Compute total L1-norm per patch coordinate [B, H, W]
        total_l1norm = abs_y.sum(dim=1)
        
        # Isolate the remaining features' L1-norm
        l1norm_rest_features = total_l1norm - max_feature
        
        # Calculate the localized ratio map at every pixel location
        # Add a tiny epsilon to prevent division by zero in case of completely zeroed patches
        eps = 1e-8
        ratio_map = l1norm_rest_features / (total_l1norm + eps)
        
        # Flatten across spatial dimensions to analyze pixel-wise variations
        flat_ratios = ratio_map.view(-1)
        
        all_ratios_means.append(flat_ratios.mean().item())
        all_ratios_vars.append(flat_ratios.var().item())
        
        # Calculate overall sparsity parameters
        total_elements += y_latent.numel()
        dead_elements += (y_latent == 0).sum().item()
        
        batch_count += 1
        break # Evaluate more batches if needed by removing this line

    sparsity_ratio = (dead_elements / total_elements) * 100
    
    # Process dominant channel index distribution
    concat_indices = torch.cat([idx.view(-1) for idx in all_dominant_indices])
    mode_res = concat_indices.mode()
    dominant_index_mode = mode_res.values.item()
    mode_frequency = (concat_indices == dominant_index_mode).float().mean().item() * 100

    print("\n" + "="*60)
    print("   LOCALIZED COEFFICIENT ENERGY RATIO STATISTICS REPORT   ")
    print("="*60)
    print(f"Index Domination Check:")
    print(f"  -> Global Dominant Channel Index (Mode): {dominant_index_mode}")
    print(f"  -> Frequency of Mode across all patches: {mode_frequency:.2f}%")
    print("-"*60)
    print(f"Ratio Statistics: (||y_rest||_1 / ||y_all||_1) per Pixel Location:")
    print(f"  -> Mean Energy Ratio in Remaining Space: {np.mean(all_ratios_means):.6f}")
    print(f"  -> Spatial Variance of Energy Ratio:     {np.mean(all_ratios_vars):.6f}")
    print("-"*60)
    print(f"Global Network Properties:")
    print(f"  -> Max Eigenvalue (λ_max) of Phi^T*Phi:  {max_ev:.4f}")
    print(f"  -> Sparsity Ratio:                       {sparsity_ratio:.2f}%")
    print("="*60 + "\n")




# ============================================================================
# Training loop
# ============================================================================

# options: 'im_out', 'noise_out'
denoiser_learning_mode = 'im_out'

# to initialize with a pre-learned model or start from scratch
pre_learned_init = False


if __name__ == '__main__':

    # --- LOG FILE DIRECTORY  ---
    log_filename = "training_log.txt"
    log_file = open(log_filename, "w", encoding="utf-8")
    
    # Backup original stdout to restore it later if needed
    original_stdout = sys.stdout 
    # Prints to both terminal and the file
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        token_dim = 256
        patch_dim = 7
        train_iter = 3

        algo_params = AlgoParams(
            denoiser_type = 'lsc', # options = 'lsc', 'rnn'
            algo='new_ISTA', # options = 'FLIPS', 'ISTA' 
            num_iter=train_iter,
            activation='leaky_relu',  # 'relu',
            normalize_phi=True
            )

        # define the training model
        if algo_params.denoiser_type == 'lsc':
            training_model = LSC_denoiser(patch_dim=patch_dim, token_dim=token_dim, train_iter=train_iter, im_color=im_color).to(device)
        elif algo_params.denoiser_type == 'rnn':
            training_model = RNN_denoiser(patch_dim=patch_dim, token_dim=token_dim, train_iter=train_iter, im_color=im_color).to(device)
        
        # count and print the number of parameters
        count_parameters(training_model)

        # load a pre-learned model if any
        if pre_learned_init == True:
            prev_checkpoint = load_checkpoint('last_model_trained.pth' , device)
            training_model.load_state_dict(prev_checkpoint['model_state_dict'])
        training_model.rnn.train_iter = train_iter

        # define loss criterion to train the model
        criterion = nn.HuberLoss() # nn.MSELoss() , nn.HuberLoss()

        # printing some basic info
        print(f'Starting training... using device: {device}')
        print(f"✓ Batch size: {batch_size}, Training batches per epoch: {len(train_loader)}, Total images:{len(train_loader)*batch_size}")
        
        num_epochs = 10
        learning_rate = 0.00125 # 0.00125

        optimizer = optim.AdamW(training_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        train_loss_history  = []
        val_loss_history    = []
        
        best_mean_psnr = 0.0
        best_epoch = -1
        batches_per_epoch = len(train_loader)
        best_model_state_dict = copy.deepcopy(training_model.state_dict())
        
        for epoch in range(num_epochs):

            # 1. Train the model - and update the train loss and confusion matrix
            train(training_model, algo_params, epoch, denoiser_learning_mode, train_loss_history)

            # 2. evealuate on validation data and compute val_loss, val_acc, and confusion_matrix
            val_loss, PSNR_values = validate(training_model, algo_params, denoiser_learning_mode)

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

            if algo_params.denoiser_type == 'lsc':
                profile_unrolled_denoising_model(training_model, algo_params, val_loader, device)
            
            
        print(f'Training completed!')
        print('-' * 80)
        

        # Load the best model weights before evaluation and saving
        if algo_params.denoiser_type == 'lsc':
            best_model = LSC_denoiser(patch_dim=patch_dim, token_dim=token_dim, train_iter=train_iter, im_color=im_color).to(device)
        elif algo_params.denoiser_type == 'rnn':
            best_model = RNN_denoiser(patch_dim=patch_dim, token_dim=token_dim, train_iter=train_iter, im_color=im_color).to(device)
        
        best_model.load_state_dict(best_model_state_dict)

        # Evaluate the best model on the test set
        test_loss, test_PSNRvalues = test(best_model, algo_params, denoiser_learning_mode)
        
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
            'learning_mode': denoiser_learning_mode,
            'epoch_trained': epoch + 1,
            'learning_rate': learning_rate,
            'token_dim': token_dim,
            'train_iter': train_iter,
            'patch_dim': patch_dim,
            'im_color': im_color,
            'algo_params': algo_params.__dict__,
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


    finally:
        # --- CLEAN UP ---
        # This guarantees the file closes properly even if the script crashes midway
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n✓ Complete log history written to {log_filename}")


