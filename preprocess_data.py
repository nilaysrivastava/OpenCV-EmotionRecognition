import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

# Define paths
dataset_path = os.path.join("..", "data", "fer2013")

# Check if dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")

# Define emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to load images and labels
def load_images_and_labels(folder_path):
    images, labels = [], []
    for i, emotion in enumerate(emotions):
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: Missing folder '{emotion}' in {folder_path}")
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(i)
    return np.array(images, dtype="float32"), np.array(labels)

# Load and preprocess data
X_train, y_train = load_images_and_labels(train_folder)
X_test, y_test = load_images_and_labels(test_folder)

X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
X_train, X_test = np.expand_dims(X_train, -1), np.expand_dims(X_test, -1)
y_train, y_test = to_categorical(y_train, num_classes=7), to_categorical(y_test, num_classes=7)

# Save preprocessed data
np.savez(os.path.join("data", "preprocessed_data.npz"),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("✅ Data preprocessing complete!")
