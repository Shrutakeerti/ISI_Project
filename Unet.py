import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define paths to the saved patches
train_image_dir = r"D:\ISI 2nd_test\train_data\images"
train_mask_dir = r"D:\ISI 2nd_test\train_data\masks"
val_image_dir = r"D:\ISI 2nd_test\val_data\images"
val_mask_dir = r"D:\ISI 2nd_test\val_data\masks"
test_image_dir = r"D:\ISI 2nd_test\test_data\images"
test_mask_dir = r"D:\ISI 2nd_test\test_data\masks"

# Load images and masks
def load_patches(image_dir, mask_dir, target_size=(32, 32)):
    images = []
    masks = []
    
    for image_file in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace("image", "mask"))
        
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
        mask = load_img(mask_path, color_mode="grayscale", target_size=target_size)
        
        images.append(img_to_array(image) / 255.0)  # Normalize images
        masks.append(img_to_array(mask) / 255.0)    # Normalize masks
    
    return np.array(images), np.array(masks)

# Load training, validation, and test data
target_size = (32, 32)
train_images, train_masks = load_patches(train_image_dir, train_mask_dir, target_size=target_size)
val_images, val_masks = load_patches(val_image_dir, val_mask_dir, target_size=target_size)
test_images, test_masks = load_patches(test_image_dir, test_mask_dir, target_size=target_size)

print(f"Train data: {train_images.shape}, Train masks: {train_masks.shape}")
print(f"Validation data: {val_images.shape}, Validation masks: {val_masks.shape}")
print(f"Test data: {test_images.shape}, Test masks: {test_masks.shape}")

# Data Augmentation
data_gen_args = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
data_gen = ImageDataGenerator(**data_gen_args)

# Define U-Net model
def unet_model(input_size=(32, 32, 1)):
    inputs = layers.Input(input_size)
    
    # Contracting path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.Dropout(0.3)(c5)

    # Expansive path
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile the model
unet = unet_model()
unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = unet.fit(
    data_gen.flow(train_images, train_masks, batch_size=8),
    validation_data=(val_images, val_masks),
    epochs=20
)

# Evaluate the model on the test set
test_loss, test_accuracy = unet.evaluate(test_images, test_masks)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the trained model
unet.save("optimized_unet_model.h5")
