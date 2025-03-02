import h5py

def load_hdf5_data(file_path):
    with h5py.File(file_path, "r") as f:
        images = f["images"][:]
        masks = f["masks"][:]
    return images, masks


train_images, train_masks = load_hdf5_data("/home/guest_miu/Dataset_init/train_data.h5")
val_images, val_masks = load_hdf5_data("/home/guest_miu/Dataset_init/val_data.h5")
test_images, test_masks = load_hdf5_data("/home/guest_miu/Dataset_init/test_data.h5")


train_images = train_images.astype("float32") / 255.0
val_images = val_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


train_masks = train_masks.astype("float32")
val_masks = val_masks.astype("float32")
test_masks = test_masks.astype("float32")

import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape=(64, 64, 1)):  
    inputs = layers.Input(input_shape)

    
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (Expansive Path)
    u5 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = models.Model(inputs, outputs)
    return model


model = build_unet(input_shape=(64, 64, 1))  # Adjust the shape to your image dimensions
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_images, train_masks, 
    validation_data=(val_images, val_masks),
    batch_size=4, 
    epochs=8
)



test_loss, test_acc = model.evaluate(test_images, test_masks)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")



predictions = model.predict(test_images)

