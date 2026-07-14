import argparse
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset_loaders import get_dataloaders
from model_architectures import *
from unfolded_denoiser_modules import *
from utils import *

# Default paths
TRAIN_PATH = 'datasets/BSDS500/train'
VAL_PATH   = 'datasets/BSDS500/val'
TEST_PATH  = 'datasets/BSD68'

test_model_path = 'last_model_trained.pth'
default_sigma = 25

def eval_fixed_iter(model, test_loader, algo_params, device):
    """Evaluates the model for a fixed number. of unrolled iterations and a fixed noise level."""

    model.eval()
    psnr_values = []
    with torch.no_grad():
        for noisy_im, clean_im in test_loader:
            noisy_im = noisy_im.to(device)
            clean_im = clean_im.to(device)

            # Forward pass with explicit number of iterations
            denoised_im = model(noisy_im, algo_params)

            # recover images from normalized range to [0, 1]
            clean_im = (clean_im + 1) / 2
            denoised_im = (denoised_im + 1) / 2

            psnr_values.append(compute_PSNR(denoised_im, clean_im).item())
    return psnr_values



def eval_fixed_noise_varying_iters(model, test_loader, fixed_sigma, algo_params, iter_list, device):
    """Evaluates the model over a range of iterations for a fixed noise level."""
    print(fr"\n--- Evaluating varying iterations at fixed noise level --- $\sigma = $ {fixed_sigma}")
    psnr_per_iter = []

    for iters in iter_list:
        model.eval()
        psnr_values = []
        with torch.no_grad():
            for noisy_im, clean_im in test_loader:
                noisy_im = noisy_im.to(device)
                clean_im = clean_im.to(device)

                # Forward pass with explicit number of iterations
                algo_params.num_iter = iters
                denoised_im = model(noisy_im, algo_params)

                # recover images from normalized range to [0, 1]
                clean_im = (clean_im + 1) / 2
                denoised_im = (denoised_im + 1) / 2


                psnr_values.append(compute_PSNR(denoised_im, clean_im).item())

        avg_psnr = sum(psnr_values) / max(1, len(psnr_values))
        psnr_per_iter.append(avg_psnr)
        print(f"Iterations: {iters:<3} | Avg PSNR: {avg_psnr:.2f} dB")

    return psnr_per_iter


def eval_varying_noise_fixed_iter(model, noise_list, algo_params, args, im_color, device):
    """Evaluates the model over a range of noise levels for a fixed number of iterations."""
    print("\n--- Evaluating varying noise levels (fixed iterations) ---")
    print(f"Fixed iterations: {algo_params.num_iter}")
    psnr_per_noise = []

    for sigma in noise_list:
        # Load dataloader specific to the current noise level (sigma)
        _, _, test_loader = get_dataloaders(
            train_dir=None,
            val_dir=None,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            patch_size=128,
            sigma=sigma,
            im_color=im_color
        )

        model.eval()
        psnr_values = []
        with torch.no_grad():
            for noisy_im, clean_im in test_loader:
                noisy_im = noisy_im.to(device)
                clean_im = clean_im.to(device)

                # Forward pass with fixed iterations
                denoised_im = model(noisy_im, algo_params)

                # recover images from normalized range to [0, 1]
                clean_im = (clean_im + 1) / 2
                denoised_im = (denoised_im + 1) / 2

                psnr_values.append(compute_PSNR(denoised_im, clean_im).item())

        avg_psnr = sum(psnr_values) / max(1, len(psnr_values))
        psnr_per_noise.append(avg_psnr)
        print(rf"Noise Sigma ($\sigma$): {sigma:<3} | Avg PSNR: {avg_psnr:.2f} dB")

    return psnr_per_noise


def save_visual_comparison(model, test_loader, noise_sigma, algo_params, device, output_path):
    """Selects a random image from a random batch and plots clean, noisy, and denoised images."""
    model.eval()
    
    # Grab a random batch from the test loader
    batches = list(test_loader)
    noisy_batch, clean_batch = random.choice(batches)
    
    # Select a random sample from the chosen batch
    idx = random.randint(0, noisy_batch.size(0) - 1)
    noisy_im = noisy_batch[idx].unsqueeze(0).to(device)
    clean_im = clean_batch[idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        denoised_im = model(noisy_im, algo_params)

    # recover images from normalized range to [0, 1]
    clean_im = (clean_im + 1) / 2
    noisy_im = (noisy_im + 1) / 2
    denoised_im = (denoised_im + 1) / 2

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean_np, cmap='gray' if clean_np.ndim == 2 else None)
    axes[0].set_title(f"Clean Image")
    axes[0].axis('off')

    axes[1].imshow(noisy_np, cmap='gray' if noisy_np.ndim == 2 else None)
    axes[1].set_title(fr"Noisy Image, $\sigma = $ {noise_sigma}, PSNR: {noisy_psnr:.2f} dB")
    axes[1].axis('off')

    axes[2].imshow(denoised_np, cmap='gray' if denoised_np.ndim == 2 else None)
    axes[2].set_title(f"Denoised Image ({algo_params.num_iter} Iterations), PSNR: {denoised_psnr}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved qualitative comparison image to: {output_path}")



def get_subspace_stats(image, patch_size):
    patches = F.unfold(image, kernel_size=patch_size, padding=patch_size//2, stride=1)
    mean_patch = torch.mean(patches, dim=1)
    patches_centered = patches - mean_patch.unsqueeze(1)
    cov_matrix = torch.bmm(patches, patches.transpose(1, 2)) / patches.size(2)
    return cov_matrix.squeeze()


@torch.no_grad()
def cov_matrix_comparison(model: Any, sample_batch_loader, device):
    """
    Profiles the relative ratio of the remaining features' L1-norm to the 
    total L1-norm across individual pixel/patch locations.
    """
    model.eval()
    model.to(device)
    for noisy_im, clean_im in sample_batch_loader:
        noisy_im = noisy_im.to(device)
        clean_im = clean_im.to(device)

        clean_cov_matrix = get_subspace_stats(clean_im, patch_size=7)
        clean_singular_values = torch.linalg.svdvals(clean_cov_matrix).numpy()

        noisy_cov_matrix = get_subspace_stats(noisy_im, patch_size=7)
        noisy_singular_values = torch.linalg.svdvals(noisy_cov_matrix).numpy()

        matrix_phi = model.phi_op.phi.weight.squeeze(-1).squeeze(-1)
        phi_cov_matrix = matrix_phi @ matrix_phi.t()
        phi_singular_values = torch.linalg.svdvals(phi_cov_matrix).numpy()

        plt.figure(figsize=(8, 7))
        plt.plot(clean_singular_values / clean_singular_values.max() , marker='o', linestyle='-', color='b', markersize=4, label='Clean Singular Values')
        plt.plot(noisy_singular_values / noisy_singular_values.max() , marker='s', linestyle='-', color='r', markersize=4, label='Noisy Singular Values')
        plt.plot(phi_singular_values / phi_singular_values.max() , marker='x', linestyle='-', color='g', markersize=4, label='Phi Singular Values')

        plt.title(fr'Singular Values of Image features and $\Phi$', fontsize=13)
        plt.xlabel('Index', fontsize=11)
        plt.ylabel(r'Singular Value ($\sigma$)', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('singular_values_comparison.png', dpi=300)
        
        break # Evaluate more batches if needed by removing this line

    


def plot_im_from_matrix(matrix, title):
    # Optional: Normalize by the maximum value to scale between [-1, 1] or [0, 1]
    max_val = np.max(np.abs(matrix))
    if max_val > 0:
        matrix_normalized = matrix / max_val
    else:
        matrix_normalized = matrix

    plt.figure(figsize=(8, 7))
    # 'viridis' or 'plasma' work great; use 'coolwarm' if you expect negative correlations
    im = plt.imshow(matrix_normalized, cmap='viridis', interpolation='nearest')
    
    plt.colorbar(im, label='Normalized Correlation Intensity')
    plt.title(title, fontsize=12)
    
    
    plt.tight_layout()
    plt.savefig('title.png', dpi=300)
    
    print("✓ plot title saved ")




def main(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint and extract hyperparameters
    checkpoint = load_checkpoint(args.model_path, device)
    token_dim = checkpoint.get('token_dim', 128)
    patch_dim = checkpoint.get('patch_dim', 7)
    train_iter = checkpoint.get('train_iter', 1)
    patch_dim = checkpoint.get('patch_dim', 7)
    im_color = checkpoint.get('im_color', 'color')

    algo_params = AlgoParams(**checkpoint.get('algo_params', {}))

    print(f"\n--- Loaded Algo Hyperparameters ---")
    print(f'Algorithm: {algo_params.algo}')
    print(f'Number of Iterations: {algo_params.num_iter}')
    print(f'Activation Function: {algo_params.activation}')

    # Instantiate your specific RNN-style denoiser model
    model = LSC_denoiser(patch_dim=patch_dim, token_dim=token_dim, train_iter=train_iter, im_color=im_color)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Experiment 1: Fixed Noise Level, Varying Iterations
    # ---------------------------------------------------------
    _, _, test_loader_fixed_noise = get_dataloaders(
        train_dir=None,
        val_dir=None,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        patch_size=128,
        sigma=args.noise_sigma,
        im_color=im_color
    )

    # Plotting to investigate the phi matrix
    print(f'Using device: {device}')
    cov_matrix_comparison(model, test_loader_fixed_noise, 'cpu')


    model.to(device)
    psnr_vs_iters = eval_fixed_noise_varying_iters(
        model=model, 
        test_loader=test_loader_fixed_noise,
        fixed_sigma = args.noise_sigma,
        algo_params=algo_params,
        iter_list=args.iter_list, 
        device=device
    )

    # ---------------------------------------------------------
    # Experiment 2: Varying Noise Levels, Fixed Iterations
    # ---------------------------------------------------------
    fixed_iter = args.fixed_iter if args.fixed_iter is not None else train_iter
    algo_params.num_iter = fixed_iter
    psnr_vs_noise = eval_varying_noise_fixed_iter(
        model=model, 
        noise_list=args.noise_list,
        algo_params=algo_params,
        args=args, 
        im_color=im_color, 
        device=device
    )
    
    # ---------------------------------------------------------
    # Figure 1: Plotting both curves side by side
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: PSNR vs Iterations
    ax1.plot(args.iter_list, psnr_vs_iters, marker='o', color='b', linestyle='-', linewidth=2)
    ax1.set_title(fr"PSNR vs. Number of Iterations (Fixed $\sigma$ = {args.noise_sigma})")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Average PSNR (dB)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Subplot 2: PSNR vs Noise Level
    ax2.plot(args.noise_list, psnr_vs_noise, marker='s', color='r', linestyle='-', linewidth=2)
    ax2.set_title(f"PSNR vs. Noise Level (Fixed Iterations = {fixed_iter})")
    ax2.set_xlabel(fr"Noise Sigma ($\sigma$)")
    ax2.set_ylabel("Average PSNR (dB)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, 'psnr_performance_curves.png')
    plt.savefig(curves_path, dpi=300)
    print(f"\nSaved combined performance plots to: {curves_path}")

    # ---------------------------------------------------------
    # Figure 2: Quantitative visual sample
    # ---------------------------------------------------------
    visual_path = os.path.join(args.output_dir, 'denoising_sample_visualization.png')
    save_visual_comparison(
        model=model, 
        test_loader=test_loader_fixed_noise,
        noise_sigma=args.noise_sigma,
        algo_params=algo_params,
        device=device, 
        output_path=visual_path
    )

    # ---------------------------------------------------------
    # Figure 3: CDF plots
    # ---------------------------------------------------------    
    # add functionality to plot the CDF plots

    cdf_psnr_values = eval_fixed_iter(model, test_loader_fixed_noise, algo_params=algo_params, device=device)
    cdf_path = os.path.join(args.output_dir, 'psnr_CDF.pdf')
    plot_psnr_CDF(cdf_psnr_values, title='CDF plot for PSNR values',save_path=cdf_path)

    plt.show()



if __name__ == '__main__':
    def parse_int_list(arg):
        return [int(x) for x in arg.split(',')]

    parser = argparse.ArgumentParser(description='Evaluate RNN-style denoiser across multiple settings.')
    parser.add_argument('--model-path', type=str, default=test_model_path,
                        help='Path to the saved model checkpoint.')
    parser.add_argument('--test-dir', type=str, default=TEST_PATH, help='Test directory.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation.')
    
    # Defaults for curves
    parser.add_argument('--noise-sigma', type=int, default=default_sigma, help='Fixed noise level for the varying iterations plot.')
    parser.add_argument('--fixed-iter', type=int, default=None, help='Fixed iteration depth for varying noise plot (defaults to checkpoint value).')
    
    # Target value sequences
    parser.add_argument('--iter-list', type=parse_int_list, default='1,2,3,4,5',
                        help='Comma-separated sequence of iterations to test (e.g., 1,2,5,10).')
    parser.add_argument('--noise-list', type=parse_int_list, default='15,20,25',
                        help=fr'Comma-separated sequence of noise levels ($\sigma$) to test (e.g., 15,20,25,35,50).')
    
    parser.add_argument('--output-dir', type=str, default='eval_results', help='Directory to save generated images.')
    args = parser.parse_args()

    main(args)



