import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import transforms

# Import custom modules based on your repository structure
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *
from architecture_modules import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple LSC denoiser models across interactive noise ranges and plot comparative PSNR."
    )
    parser.add_argument(
        '--test_path', 
        type=str, 
        default='data/BSD68',
        help="Path to the test dataset directory."
    )
    parser.add_argument(
        '--max_images', 
        type=int, 
        default=100,
        help="Maximum number of test images to evaluate (default: 1000)."
    )
    parser.add_argument(
        '--output_plot', 
        type=str, 
        default='plots_and_images/multi_model_noise_vs_psnr.png',
        help="Output filename for the comparative scatterplot."
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='',
        help="Device to run evaluation on ('cuda', 'mps', 'cpu'). Auto-selects if empty."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup PyTorch device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================================
    # 1. Interactive Input Loop for Models and Noise Levels
    # ============================================================================
    models_to_test = []
    
    print("\n=== Interactive Model & Noise Evaluation Setup ===")
    while True:
        model_path = input("Enter the path to the model checkpoint file (e.g., 'last_model_trained.pth'): ").strip()
        if not model_path:
            model_path = 'last_model_trained.pth'
            
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at '{model_path}'. Please try again.")
            continue

        model_label = input(f"Enter a custom label/name for this model [Default: {Path(model_path).stem}]: ").strip()
        if not model_label:
            model_label = Path(model_path).stem

        sigma_input = input("Enter the noise-sigma sequence to test model, for eg., 5 10 15 20: ").strip()
        try:
            if sigma_input:
                sigma_seq = list(map(float, sigma_input.split()))

            else:
                sigma_range = input("Enter the noise-sigma range, for eg., 20 25: ").strip()
                sigma_range = list(map(float, sigma_range.split()))
                num_sigma_points = input("Enter the number of noise values to evaluate model for: ").strip()
                num_sigma_points = int(num_sigma_points)
                sigma_seq = np.linspace(sigma_range[0], sigma_range[1], num_sigma_points)

        except ValueError:
            print("X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X Invalid range format. Defaulting to [15.0, 25.0, 50.0] X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X")
            sigma_seq = [15.0, 25.0, 50.0]

        # Load the model immediately to verify it
        print(f"Loading checkpoint from '{model_path}'...")
        checkpoint = load_checkpoint(model_path, device)

        model_attributes = checkpoint.get('model_attributes')

        denoiser_type = model_attributes['denoiser_type']
        if denoiser_type == 'lsc':
            model = LSC_denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'conv':
            model = Fully_Conv_Denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'transformer':
            model = MHA_Vision_Transformer_Denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'UNet':
            model = UNet_denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'BLT':
            model = BLT_Denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'MHTA_transformer':
            model = MHTA_Transformer_Denoiser(attributes=model_attributes).to(device)
        elif denoiser_type == 'temp':
            model = Temp_Denoiser(attributes=model_attributes).to(device)



        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✓ Model '{model_label}' of type {model.denoiser_type} successfully loaded.\n")
        # count and print the number of parameters
        count_parameters(model)

        models_to_test.append({
            'name': model_label,
            'model': model,
            'sigma_seq': sigma_seq
        })

        # Ask if the user wants to add another model
        cont = input("Is that the only set of models we want to test? (Type 'yes' to finish, or 'no' to add another model): ").strip().lower()
        if cont in ['yes', 'y']:
            break
        print("-" * 50)

    # ============================================================================
    # 2. Evaluation Loop across All Models and Their Noise Ranges
    # ============================================================================

    image_folder = args.test_path
    num_images = args.max_images

    val_transform = transforms.ToTensor()
    results = {m['name']: {'sigmas': [], 'psnrs': [], 'mean_psnr': [], 'psnr_std': []} for m in models_to_test}

    print(f"\nStarting evaluation for {len(models_to_test)} model(s) across test images...")

    with torch.no_grad():
        for m_dict in models_to_test:

            model = m_dict['model']
            m_name = m_dict['name']
            print(f'Evaluating model {model.denoiser_type}')
            for current_sigma in m_dict['sigma_seq']:
                psnr_values = model.evaluate(image_folder,[current_sigma,], num_images=num_images)
                mean_psnr = np.mean(psnr_values)
                psnr_std = np.sqrt(np.mean((psnr_values - mean_psnr)**2))
                results[m_name]['sigmas'].append(current_sigma)
                results[m_name]['psnrs'].append(psnr_values)
                results[m_name]['mean_psnr'].append(mean_psnr)
                results[m_name]['psnr_std'].append(psnr_std)



# ============================================================================
    # 3. Plotting Mean PSNR Curve with Standard Deviation Discs
    # ============================================================================

    plt.figure(figsize=(20, 8))
    
    for m_name in results.keys():
        sigmas = results[m_name]['sigmas']
        mean_psnrs = results[m_name]['mean_psnr']
        psnr_stds = results[m_name]['psnr_std']
        
        # Scale standard deviation values into marker sizes (s parameter in scatter represents area)
        # Adjust the multiplier (e.g., 300) to make the discs larger or smaller
        disc_sizes = [val * 200 for val in psnr_stds]

        # Plot the curve and markers for mean PSNR (let matplotlib auto-assign distinct colors)
        line = plt.plot(
            sigmas, mean_psnrs, 
            marker='o', 
            linewidth=0.5, 
            markersize=3,
            label=f"{m_name}"
        )
        line_color = line[0].get_color()

        # Overlay transparent disc scaled to the standard deviation at each point
        plt.scatter(
            sigmas, mean_psnrs, 
            s=disc_sizes, 
            color=line_color, 
            alpha=0.3, 
            edgecolors='none'
        )

    plt.title('Denoiser Zoo: Noise Sigma vs. PSNR', fontsize=13, fontweight='bold')
    plt.xlabel(r'Noise Sigma ($\sigma$)', fontsize=11)
    plt.ylabel('mean PSNR (dB)', fontsize=11)

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9)

    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparative plot successfully saved to '{args.output_plot}'.")


    



    # ============================================================================
    # 4. Generate Combined Scatterplot: noise_sigma vs psnr
    # ============================================================================
    scatterplot = False
    if scatterplot:
        plt.figure(figsize=(10, 6))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for i, m_dict in enumerate(models_to_test):
            m_name = m_dict['name']
            sigmas = results[m_name]['sigmas']
            psnrs = results[m_name]['psnrs']
            mean_psnr = np.mean(psnrs)
            marker = markers[i % len(markers)]

            plt.scatter(
                sigmas, psnrs, 
                alpha=0.25, 
                marker=marker, 
                edgecolors='k', 
                linewidth=0.3, 
                label=f"{m_name} (Mean: {mean_psnr:.2f} dB)"
            )

        plt.title('Denoiser zoo: Noise Sigma vs. PSNR Comparison Across Multiple Models', fontsize=13, fontweight='bold')
        plt.xlabel(r'Noise Sigma ($\sigma$)', fontsize=11)
        plt.ylabel('PSNR (dB)', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', framealpha=0.9)

        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Comparative plot successfully saved to '{args.output_plot}'.")


if __name__ == '__main__':
    main()