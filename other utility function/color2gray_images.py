import cv2
import os
from pathlib import Path

def convert_bsd68_to_grayscale(source_dir, output_dir):
    # Create the output directory if it doesn't exist
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    print(f"Starting conversion from {source_path}...")
    
    count = 0
    for img_file in source_path.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            # Read the color image
            img_color = cv2.imread(str(img_file))
            
            if img_color is not None:
                # Convert to Gray (uses the formula: Y = 0.299R + 0.587G + 0.114B)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                
                # Save the new image
                save_name = output_path / img_file.name
                cv2.imwrite(str(save_name), img_gray)
                count += 1
            else:
                print(f"Could not read: {img_file.name}")

    print(f"Done! Processed {count} images.")

# --- Configuration ---
SOURCE_FOLDER = '/Users/mrayyan/Documents/Research Projects/Image_denoising/datasets/McMaster/color'
DESTINATION_FOLDER = '/Users/mrayyan/Documents/Research Projects/Image_denoising/datasets/McMaster/gray'

if __name__ == "__main__":
    convert_bsd68_to_grayscale(SOURCE_FOLDER, DESTINATION_FOLDER)