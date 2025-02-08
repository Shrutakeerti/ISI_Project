import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import img_to_array

# Define the path to your test images and labels
test_images_path = r"D:\only paste\test_data\images"
test_labels_path = r"D:\only paste\test_data\masks"

# Load the model
model = load_model(r'D:\only paste\unet_model.h5')

# Function to load images
def load_images_from_directory(directory_path):
    images = []
    for file_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, file_name)
        if img_path.endswith('.png') or img_path.endswith('.jpg'):  # check for image files
            img = image.load_img(img_path, target_size=(256, 256))  # Use the same input size as during training
            img_array = img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)
    return np.array(images)

# Load the test images and labels
test_images = load_images_from_directory(test_images_path)
# For labels, ensure they are loaded properly and correspond to your model's output
test_labels = load_images_from_directory(test_labels_path)  # Adjust for your dataset's labels

# Make predictions
predictions = model.predict(test_images)

# Compare predictions with true labels (you may need to adjust this for your specific use case)
accuracy = accuracy_score(test_labels.flatten(), predictions.flatten())  # Use appropriate flattening
print(f"Model Accuracy: {accuracy * 100:.2f}%")
