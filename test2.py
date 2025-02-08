import numpy as np
import cv2
import os
from patchify import patchify

# Define input and output directories
image_dir = 'small_dataset_for_training/images/'
mask_dir = 'small_dataset_for_training/masks/'

output_image_dir = 'patches/images/'
output_mask_dir = 'patches/masks/'

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# List images and masks
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])

print(f"Found {len(image_files)} images and {len(mask_files)} masks in the dataset.")

# Process images
for img_idx, img_name in enumerate(image_files):
    img_path = os.path.join(image_dir, img_name)
    large_image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read JPG as BGR

    if large_image is None:
        print(f"❌ Error loading image: {img_path}")
        continue
    print(f"✅ Loaded image: {img_path}, Shape: {large_image.shape}")

    patches_img = patchify(large_image, (256, 256, 3), step=256)  # 3 channels for color
    print(f"Patches for {img_name}: {patches_img.shape}")

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0]  # Extract patch
            patch_path = os.path.join(output_image_dir, f'image_{img_idx}_{i}{j}.jpg')
            cv2.imwrite(patch_path, single_patch_img)
            print(f"📝 Saved patch: {patch_path}")

# Process masks
for mask_idx, mask_name in enumerate(mask_files):
    mask_path = os.path.join(mask_dir, mask_name)
    large_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale

    if large_mask is None:
        print(f"❌ Error loading mask: {mask_path}")
        continue
    print(f"✅ Loaded mask: {mask_path}, Shape: {large_mask.shape}")

    patches_mask = patchify(large_mask, (256, 256), step=256)  # Grayscale mask
    print(f"Patches for {mask_name}: {patches_mask.shape}")

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, 0]  # Extract patch

            # Normalize mask to 0-255
            single_patch_mask = (single_patch_mask / 255.0) * 255
            single_patch_mask = single_patch_mask.astype(np.uint8)

            patch_path = os.path.join(output_mask_dir, f'mask_{mask_idx}_{i}{j}.jpg')
            cv2.imwrite(patch_path, single_patch_mask)
            print(f"📝 Saved mask patch: {patch_path}")

print("✅ Patching complete! 🚀")

