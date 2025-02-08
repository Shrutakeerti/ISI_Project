import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np

# Define paths
original_ct_scan_dir = r"D:\only paste\LIDC-IDRI-slices\LIDC-IDRI-0001\nodule-0\images"  # Change to an actual patient folder
patch_dir = r"D:\only paste\train_data\images"  # Directory containing extracted patches

# Get original CT scan images and patches
original_images = sorted(os.listdir(original_ct_scan_dir))  # Sorted to ensure correct order
patch_images = sorted(os.listdir(patch_dir))  # Extracted patches

# Select a random CT slice
random_ct_idx = random.randint(0, len(original_images) - 1)
original_ct_path = os.path.join(original_ct_scan_dir, original_images[random_ct_idx])

# Load the original CT slice
original_ct_img = cv2.imread(original_ct_path, cv2.IMREAD_GRAYSCALE)

# Load corresponding extracted patches (randomly pick a few)
num_patches_to_show = 5  # Number of patches to visualize
random_patch_indices = random.sample(range(len(patch_images)), num_patches_to_show)
patches = [cv2.imread(os.path.join(patch_dir, patch_images[idx]), cv2.IMREAD_GRAYSCALE) for idx in random_patch_indices]

# Plot original CT image and extracted patches
fig, axes = plt.subplots(1, num_patches_to_show + 1, figsize=(15, 5))

# Plot the original CT scan
axes[0].imshow(original_ct_img, cmap="gray")
axes[0].set_title(f"Original CT Slice {random_ct_idx}")
axes[0].axis("off")

# Plot extracted patches
for i, patch in enumerate(patches):
    axes[i + 1].imshow(patch, cmap="gray")
    axes[i + 1].set_title(f"Patch {random_patch_indices[i]}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()
