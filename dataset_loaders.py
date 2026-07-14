import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
    

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


