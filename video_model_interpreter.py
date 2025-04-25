import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("animal_behavior_autoencoder_optimized.keras")

# Load one video sequence for testing
def load_video_sequence_test(folder_path, num_frames=4, img_size=48):
    frames = []
    for fname in sorted(os.listdir(folder_path))[:num_frames]:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            frames.append(img / 255.0)
    while len(frames) < num_frames:
        frames.append(np.zeros((img_size, img_size, 3)))
    return np.array(frames, dtype=np.float32)

# Path to a sample test folder (e.g., "processed_data/video_frames/fight_001")
test_folder = "processed_data/video_frames/AADBUVKA"
input_sequence = load_video_sequence_test(test_folder)
input_sequence = np.expand_dims(input_sequence, axis=0)  # shape: (1, T, H, W, C)

# Predict (reconstruction)
reconstructed = model.predict(input_sequence)

# Compute reconstruction error
reconstruction_error = np.mean((input_sequence - reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.6f}")

# Visualize input vs reconstruction (first frame)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(input_sequence[0][0])
plt.title("Original Frame")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[0][0])
plt.title("Reconstructed Frame")
plt.show()
