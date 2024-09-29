import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid

def plot_images(image_name):
    base_dir = os.getcwd()
    severity_levels = ["severity_lv1", "severity_lv2", "severity_lv3", "severity_lv4"]
    subfolders = ["Brightness", "Contrast", "Defocus_Blur", "Elastic", "Gaussian_Noise", 
                  "Impulse_Noise", "jpeg_comp", "Motion_Blur", "Pixelate", "Shot_Noise", "Zoom_Blur"]

    # Define a transformation to convert PIL images to tensors
    to_tensor = transforms.ToTensor()
    image_list = []

    for j, subfolder in enumerate(subfolders):
        print(subfolder)
        for i, severity in enumerate(severity_levels):
            if '_' in subfolder.lower():
                img_path = os.path.join(base_dir, severity, subfolder, f"{image_name}_{subfolder.lower().split('_')[0]}_.jpg")
            else:
                img_path = os.path.join(base_dir, severity, subfolder, f"{image_name}_{subfolder.lower()}_.jpg")
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                image_list.append(to_tensor(img))  # Convert image to tensor and append to list
            else:
                print(f"Image not found: {img_path}")
                
    # Convert list of tensors to a single tensor
    image_tensor = torch.stack(image_list, 0)

    # Create a grid of images
    grid_img = make_grid(image_tensor, nrow=len(severity_levels))  # Setting nrow to the number of severity levels

    # Display the grid
    # Save the grid
    save_path = os.path.join(base_dir, f"{image_name}_grid.jpg")
    plt.figure(figsize=(100, 40))
    plt.imshow(grid_img.permute(1, 2, 0))  # Adjust channel dimensions for visualization
    plt.axis('off')
    plt.tight_layout()  # Makes sure the saved image doesn't have unnecessary padding
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save without extra space
    print(f"Saved grid image to: {save_path}")
    plt.close()  # Close the figure to free up memory

# Usage:
plot_images("d_r_47")


