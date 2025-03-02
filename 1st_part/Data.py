import os
import numpy as np
import nibabel as nib  # For handling medical images
from sklearn.model_selection import train_test_split
import cv2  # For image processing
import h5py

# Define dataset path
dataset_path = "/home/guest_miu/Dataset_init/LIDC-IDRI-slices"

# Define function to load slices and masks
def load_patient_data(patient_folder):
    patient_data = []
    for nodule_folder in os.listdir(patient_folder):
        nodule_path = os.path.join(patient_folder, nodule_folder)
        if not os.path.isdir(nodule_path):
            continue
        
        images_path = os.path.join(nodule_path, "images")
        masks_paths = [
            os.path.join(nodule_path, f"mask-{i}") for i in range(4)
        ]
        
        # Load image slices and masks
        image_slices = sorted([os.path.join(images_path, f) for f in os.listdir(images_path)])
        mask_slices = [
            sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)]) 
            for mask_path in masks_paths
        ]
        
        patient_data.append((image_slices, mask_slices))
    return patient_data

# Load all patient data
all_data = []
for patient in os.listdir(dataset_path):
    patient_folder = os.path.join(dataset_path, patient)
    if os.path.isdir(patient_folder):
        all_data.extend(load_patient_data(patient_folder))

# Split dataset into train and test sets
train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=42)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

def extract_patches(images, masks, patch_size=64, stride=32):
    patches = []
    mask_patches = []

    for img_path, mask_paths in zip(images, zip(*masks)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        masks_combined = sum([cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths])

        h, w = img.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                mask_patch = masks_combined[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                mask_patches.append(mask_patch)

    return np.array(patches), np.array(mask_patches)

# Example for training data
train_patches = []
train_mask_patches = []

for image_slices, mask_slices in train_data:
    patches, mask_patches = extract_patches(image_slices, mask_slices)
    train_patches.extend(patches)
    train_mask_patches.extend(mask_patches)

train_patches = np.array(train_patches)
train_mask_patches = np.array(train_mask_patches)
print(f"Extracted {len(train_patches)} patches from training data.")

# Split patches into train and validation
train_patches, val_patches, train_masks, val_masks = train_test_split(
    train_patches, train_mask_patches, test_size=0.2, random_state=42
)

print(f"Training patches: {len(train_patches)}, Validation patches: {len(val_patches)}")

# Extract patches for test data
test_patches = []
test_mask_patches = []

for image_slices, mask_slices in test_data:
    patches, mask_patches = extract_patches(image_slices, mask_slices)
    test_patches.extend(patches)
    test_mask_patches.extend(mask_patches)

test_patches = np.array(test_patches)
test_mask_patches = np.array(test_mask_patches)

# Save data to HDF5 file
def save_to_hdf5(file_path, images, masks):
    with h5py.File(file_path, "w") as f:
        f.create_dataset("images", data=images, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")

# Save train, validation, and test sets
save_to_hdf5("train_data.h5", train_patches, train_masks)
save_to_hdf5("val_data.h5", val_patches, val_masks)
save_to_hdf5("test_data.h5", test_patches, test_mask_patches)

print("Data saved to HDF5 files.")
