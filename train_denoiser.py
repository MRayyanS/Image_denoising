import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.amp import autocast
# from torch.mps import amp

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import random

import os
from PIL import Image
import copy

import sys

from utils import *


from architecture_modules import *
# from LSC_modules import *

# only to supress warning messages
import warnings
# This ignores the NumPy/Torchvision compatibility warning without needing the np attribute
warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Essential global variables
# ============================================================================
seed = 10000000
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# torch.mps.synchronize()


# ============================================================================
# LOAD appropriate DATASET and create train/val split
# ============================================================================

# Your specific paths

# TRAIN_PATH = 'data/imagenet1k_256'
TRAIN_PATH = 'data/PascalVoc07_256/last500'

VAL_PATH = 'data/BSD68'
# VAL_PATH   = 'data/Set12/Set12_256'
TEST_PATH  = 'data/Set12/Set12_256'

# Set some hyperpaprameters
batch_size = 1
im_size = 128
sigma_range = [15,15.1]
im_color = 'gray' # 'color' or 'gray'


# initialize model from a pre-learned model
pre_learned_init = 'model_params'  # 'everything' , 'model_params' , 'none'

train_transform = PairedSpatialTransform(im_size)
train_loader = im_denoising_dataloader(image_dir=TRAIN_PATH, batch_size=batch_size, transform = train_transform, sigma_range=sigma_range, im_color=im_color)

val_sigma_range = [15,15.1]
val_transform = transforms.ToTensor()
val_loader  = im_denoising_dataloader(image_dir=VAL_PATH, batch_size=1, transform = val_transform, sigma_range=val_sigma_range, im_color=im_color)
test_loader = im_denoising_dataloader(image_dir=TEST_PATH, batch_size=1, transform = val_transform, sigma_range=val_sigma_range, im_color=im_color)


# ============================================================================
# Build train loop, validation, and test functions
# ============================================================================


def train(model, epoch: int, train_loss_history: list):
    model.train()
    epoch_loss = 0.0

    for i, (noisy_im, clean_im, _) in enumerate(train_loader):
        noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

        # extract patches to compute patch-based loss
        B, C, H, W = clean_im.size()

        # # -----------------------------------------------------------------
        # # PROFILER BLOCK: Run on batch index 5 (after 5 warmup steps)
        # # -----------------------------------------------------------------
        # if epoch == 0 and i == 5:
        #     print("\n=== RUNNING PROFILER ===")

        #     with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU],
        #         record_shapes=True
        #     ) as prof:
        #         # with torch.autocast(device_type="mps", dtype=torch.float16):
        #         optimizer.zero_grad()
        #         outputs = model(noisy_im)
        #         denoised_im = noisy_im - outputs if learning_mode == 'noise_out' else outputs
        #         loss = criterion(clean_im, denoised_im)
        #         loss.backward()
        #         optimizer.step()

        #     # Print top operations sorted by total CPU dispatch time
        #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        #     print("=== PROFILING COMPLETE — EXITING SCRIPT ===")
        #     import sys; sys.exit(0)
        # # -----------------------------------------------------------------

        denoised_im = model(noisy_im)
        loss = criterion(clean_im, denoised_im)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Place this inside your training loop (e.g., right after optimizer.step())
        # print(f"Current MPS Memory: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")
        # print(f"Driver Allocated Memory: {torch.mps.driver_allocated_memory() / (1024**2):.2f} MB")

        # normalize columns of phi if required
        model.phi_op.normalize_phi() if model.attributes['denoiser_type'] == 'lsc' and model.attributes['normalize_phi'] else None

        detached_loss = loss.item() 
        train_loss_history.append(detached_loss)
        epoch_loss += (detached_loss - epoch_loss) / (i + 1)

        # torch.mps.empty_cache()
    
    # 2. Convert to a standard Python float ONLY when printing at the end
    print(f'Epoch = {epoch}')
    print(f'Train Loss: {epoch_loss:.8f}')


# function for validation
def validate(model):
    model.eval()
    
    val_loss = torch.tensor(0.0, device=device)
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im, noise_sigma in val_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            test_patchwise = False
            test_patch_size = im_size
            if model.patchsize_sensitive == True:
                test_patchwise = True
                test_patch_size = model.im_size

            if test_patchwise:                                
                denoised_im = denoise_image_patchwise(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=4)
                # denoised_im = denoise_image_tta(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=8)
            else:
                denoised_im = model(noisy_im)
            
            loss = criterion(clean_im, denoised_im)
            psnr = compute_PSNR(clean_im, denoised_im)

            # # Place this inside your training loop (e.g., right after optimizer.step())
            # print(f"Current MPS Memory: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB")
            # print(f"Driver Allocated Memory: {torch.mps.driver_allocated_memory() / (1024**2):.2f} MB")

            # 3. Accumulated as tensors without blocking
            val_loss += loss
            PSNR_values.append(psnr)

        avg_loss = (val_loss / len(val_loader))
        
        # Convert the whole list to CPU/floats in one asynchronous batch operation
        PSNR_values = torch.stack(PSNR_values).cpu().tolist()

        # show denoising for the last image in the test set
        image_path = 'plots_and_images/visualize_denoiser_performance.png'
        visualise_denoiser_performance(model, [clean_im, noisy_im], 255*noise_sigma.item(), device, output_path=image_path)
            
    return avg_loss.cpu(), PSNR_values


# function for testing
def test(model):
    model.eval()
    print(f'Testing with the best model on the test data')
    
    test_loss = 0.0
    PSNR_values = []
    
    with torch.no_grad():
        for noisy_im, clean_im, noise_sigma in test_loader:
            noisy_im, clean_im = noisy_im.to(device), clean_im.to(device)

            test_patchwise = False
            test_patch_size = 64
            if model.patchsize_sensitive == True:
                test_patchwise = True
                test_patch_size = model.im_size

            if test_patchwise:                                
                denoised_im = denoise_image_patchwise(model, noisy_im, patch_size=test_patch_size, stride=test_patch_size//2, batch_size=8)
            else:
                denoised_im = model(noisy_im)
            
            loss = criterion(clean_im, denoised_im)
            psnr = compute_PSNR(clean_im, denoised_im)

            test_loss += loss.item()
            PSNR_values.append(psnr.item())
            
    avg_loss = test_loss / len(test_loader)

    # show denoising for the last image in the test set
    image_path = 'plots_and_images/visualize_denoiser_performance.png'
    visualise_denoiser_performance(model, [clean_im, noisy_im], 255*noise_sigma.item(), device, output_path=image_path)
    return avg_loss, PSNR_values


# ============================================================================
# Training loop
# ============================================================================

if __name__ == '__main__':

     # --- LOG FILE DIRECTORY  ---
    log_filename = "training_results/training_log.txt"
    log_file = open(log_filename, "w", encoding="utf-8")
    
    # Backup original stdout to restore it later if needed
    original_stdout = sys.stdout 
    # Prints to both terminal and the file
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # load previous check-point if any
        if pre_learned_init in ('everything', 'model_params'):
            model_path = input("Enter the path to the model checkpoint file (e.g., 'last_model_trained.pth'): ").strip()
            if not model_path:
                model_path = 'last_model_trained.pth'
            prev_checkpoint = load_checkpoint(model_path , device)


        # define denoiser type
        denoiser_type = 'conv'   # types: 'conv', 'transformer', 'BLT', 'UNet', 'MHTA_transformer', 'tinkering'

        
        if denoiser_type == 'lsc':
            
            model_attributes = {
                'denoiser_type': 'lsc',
                'im_color': im_color,
                'op_mode': 'im_est', # 'noise_est', 'im_est'
                'train_noise': sigma_range,
                'algo': 'new_ISTA',
                'activation': 'relu',
                'normalize_phi': True,
                'train_iter': 5,
                'token_dim': 256,
                'patch_dim': 9,
                }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes
            
            training_model = LSC_denoiser(attributes = model_attributes).to(device)
            best_model     = LSC_denoiser(attributes = model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None

        
        if denoiser_type == 'conv':

            model_attributes = {
                'im_color': im_color,
                'op_mode': 'noise_est',  # 'noise_est', 'im_est'
                'train_noise': sigma_range,
                'denoiser_type': 'conv',
                'patch_dim': 7,
                'token_dim': 64,
                'low_emb_dim': 16,
                'num_modules': 4,
                'train_iter': 3,
            }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes
            
            training_model = Fully_Conv_Denoiser( attributes = model_attributes).to(device)
            best_model     = Fully_Conv_Denoiser( attributes = model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None


        elif denoiser_type == 'transformer':
            
            token_dim = 64
            attention_type = 'spatial' # 'spatial', 'channel'
        
            attention_config = {
                'att_emb_dim': 16,
                'num_att_heads': 1,
                'pixel_shuffle_factor': 4 if attention_type == 'spatial' else 8,
                'attention_type': attention_type,
                'projection_type': 'shuffle_after_QK', # 'shuffle_before_QK', 'shuffle_after_QK'
                }

            model_attributes = {
                'denoiser_type': 'transformer',
                'op_mode': 'noise_est',
                'noise_range': sigma_range,
                'im_color': im_color,
                'attention_config': attention_config, # attention_config = {'att_emb_dim': att_emb_dim, 'num_att_heads': num_att_heads, 'pixel_shuffle_factor': pixel_shuffle_factor, 'attention_type': attention_type, 'projection_type': projection_type}
                'num_modules': 3,
                'train_iter': 3,
                'patch_dim': 7,
                'token_dim': token_dim,
                'conv_middle_dim': 16,
                }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes

            training_model = MHA_Vision_Transformer_Denoiser(attributes=model_attributes).to(device)
            best_model     = MHA_Vision_Transformer_Denoiser(attributes=model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None


        elif denoiser_type == 'UNet':

            model_attributes = {
                'im_color': im_color,
                'op_mode': 'noise_est',  # 'noise_est', 'im_est'
                'train_noise': sigma_range,
                'denoiser_type': 'UNet',

                'patch_dim': 7,
                'token_dim': 64,

                'num_layers': 1,
                'low_emb_dim_list': [16, 16, 16],
                'train_iter_list': [1, 1, 1],
            }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes
            
            training_model = UNet_denoiser(attributes=model_attributes).to(device)
            best_model     = UNet_denoiser(attributes=model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None
            

        elif denoiser_type == 'BLT':

            model_attributes = {
                'im_color': im_color,
                'op_mode': 'noise_est',  # 'noise_est', 'im_est'
                'train_noise': sigma_range,
                'denoiser_type': 'BLT',

                'patch_dim': 7,
                'token_dim_list': [64, im_size, im_size],  # token_dim_list[0] is the number of channels
                'transform_dim_list': [16, 32, 32],
                
                'num_modules': 3,
                'train_iter': 3,

                'pixel_shuffle_factor': 2,
                'large_context_type': 'BLT_attention',   # 'BLT_conv', 'BLT_attention'
            }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes
            
            training_model = BLT_Denoiser(attributes=model_attributes).to(device)
            best_model     = BLT_Denoiser(attributes=model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None


        elif denoiser_type == 'MHTA_transformer':
                    
            model_attributes = {
            'denoiser_type': 'MHTA_transformer',
            'im_color': im_color,
            'op_mode': 'noise_est',  # 'noise_est', 'im_est'
            'train_noise': sigma_range,
            
            'patch_dim': 7,
            'tensor_dim': [64, im_size, im_size],

            'num_modules': 1,
            'train_iter': 1,
        
            'att_emb_dim': [32, 32],
            'pixel_shuffle_factor': 4,
            'num_att_heads': 1, 
            'conv_emb_dim': 32
            }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes
            
            training_model = MHTA_Transformer_Denoiser(attributes = model_attributes).to(device)
            best_model     = MHTA_Transformer_Denoiser(attributes = model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None
            

        elif denoiser_type == 'temp':

            model_attributes = {
                'im_color': im_color,
                'op_mode': 'noise_est',  # 'noise_est', 'im_est'
                'train_noise': sigma_range,
                'denoiser_type': 'BLT',

                'patch_dim': 7,
                'token_dim_list': [64, im_size, im_size],  # token_dim_list[0] is the number of channels
                'transform_dim_list': [24, 32, 32],
                
                'num_modules': 3,
                'train_iter': 1,

                'pixel_shuffle_factor': 2,
                'large_context_type': 'BLT_attention',   # 'BLT_conv', 'BLT_attention'
            }

            # load attributes from previous checkpoint if needed
            model_attributes = prev_checkpoint.get('model_attributes', model_attributes) if pre_learned_init == 'everything' else model_attributes

            training_model = Temp_Denoiser(attributes = model_attributes).to(device)
            best_model     = Temp_Denoiser(attributes = model_attributes).to(device)

            # load a pre-learned model if any
            training_model.load_state_dict(prev_checkpoint['model_state_dict']) if pre_learned_init in ('everything', 'model_params') else None


        # count and print the number of parameters
        count_parameters(training_model)

        # define loss criterion to train the model
        # criterion = nn.MSELoss()
        criterion = L1L2Loss(regularizer=0.25)

        # criterion = SSIMLoss(window_size=11, sigma=1.5, channel=1)

        # printing some basic info
        print(f'Starting training... using device: {device}')
        print(f"✓ Batch size: {batch_size}, Training batches per epoch: {len(train_loader)}, Total images:{len(train_loader)*batch_size}")
        
        num_epochs = 150
        learning_rate = 0.0005   #0.00125
        
        optimizer = optim.AdamW(training_model.parameters(), lr=learning_rate, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-5)

        train_loss_history  = []
        val_loss_history    = []
        
        best_mean_psnr = 0.0
        best_epoch = -1
        batches_per_epoch = len(train_loader)
        best_model_state_dict = copy.deepcopy(training_model.state_dict())
        
        for epoch in range(num_epochs):
            # 1. Train the model - and update the train loss and confusion matrix
            train(training_model, epoch, train_loss_history)

            # 2. evealuate on validation data and compute val_loss, val_acc, and confusion_matrix
            val_loss, PSNR_values = validate(training_model)

            val_loss_history.append(val_loss)
            mean_psnr = np.mean(PSNR_values)
            print(f'Valid Loss: {val_loss:.8f} \n')
            print(f'Validation PSNR -- mean: {mean_psnr:.2f} dB, median: {np.median(PSNR_values):.2f} dB \n')

            scheduler.step()

            if  mean_psnr > best_mean_psnr:
                best_mean_psnr = mean_psnr
                best_epoch = epoch
                best_model_state_dict = copy.deepcopy(training_model.state_dict())

                # save the current best model weights
                training_results ={ 'model_attributes': model_attributes, 'model_state_dict': best_model_state_dict }
                torch.save(training_results, 'last_model_trained.pth')                
                print(f'Best model weights saved with mean PSNR: {best_mean_psnr:.2f} dB at epoch: {best_epoch}')


            if mean_psnr < best_mean_psnr - 1.0:
                print(f'Warning: Validation PSNR is below 26 dB at epoch {epoch} loading best model weights from epoch {best_epoch} and continuing training...')
                training_model.load_state_dict(best_model_state_dict)

            
            print(f'Best mean PSNR: {best_mean_psnr:.2f} dB, best PSNR at epoch = {best_epoch}')
            print('-' * 60)

            # save_visual_comparison(training_model, test_loader, noise_sigma = sigma_range[1], device=device, output_path='denoising_sample_visualization.png')
            

        print(f'Training completed!')
        print('-' * 80)
        

        # load best model weights for final evaluation on test set
        best_model.load_state_dict(best_model_state_dict)

        # Evaluate the best model on the test set
        test_loss, test_PSNRvalues = test(best_model)
        
        print('-' * 80)
        print(f'Best Val PSNR: {best_mean_psnr:.2f} dB, at epoch: {best_epoch}')
        print(f'Final Test PSNR: {np.mean(test_PSNRvalues):.2f} dB')
        print('-' * 80)


        # ============================================================================
        # Saving everything
        # ============================================================================

        # Create a dictionary containing all the data you want to preserve
        training_results = {
            'epoch_trained': epoch + 1,
            'model_attributes': model_attributes,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'best_val_psnr': best_mean_psnr,
        }

        # Save everything into one file
        torch.save(training_results, 'last_model_trained.pth')
        torch.save(training_results, 'last_training_results.pth')

        print(f"\n✓ All training results and best model saved")

        # plotting the loss curves and PSNR distribution
        plot_loss_curves(train_loss_history, val_loss_history, batches_per_epoch, output_path='plots_and_images/loss_curves.png')
        

    finally:
        # --- CLEAN UP ---
        # This guarantees the file closes properly even if the script crashes midway
        sys.stdout = original_stdout
        log_file.close()
        print(f"\n✓ Complete log history written to {log_filename}")



