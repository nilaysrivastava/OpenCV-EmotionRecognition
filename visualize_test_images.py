import matplotlib.pyplot as plt
import numpy as np

# Load the preprocessed data
data = np.load("data/preprocessed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Convert one-hot encoded labels back to categorical labels
y_test_labels = np.argmax(y_test, axis=1)

# Define emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Plot the first 5 test images with labels
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(X_test[i].squeeze(), cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(emotions[y_test_labels[i]])

plt.show(block=True)  # Ensure the window stays open
