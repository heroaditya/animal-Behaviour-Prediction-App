import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# -------------------------------
# Settings
# -------------------------------
DATA_PATH = "processed_data/video_frames/"
IMG_SIZE = 48
FRAMES_PER_VIDEO = 4
BATCH_SIZE = 4
EPOCHS = 10

# -------------------------------
# Load frames from one video
# -------------------------------
def load_video_sequence(folder_path, num_frames=FRAMES_PER_VIDEO):
    frames = []
    for fname in sorted(os.listdir(folder_path))[:num_frames]:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            frames.append(img / 255.0)
    while len(frames) < num_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    return np.array(frames, dtype=np.float32)

# -------------------------------
# Dataset generator using tf.data
# -------------------------------
def create_dataset(folders, batch_size):
    def gen():
        for folder in folders:
            frames = load_video_sequence(os.path.join(DATA_PATH, folder))
            yield frames, frames

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Prepare datasets
# -------------------------------
video_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
train_folders, val_folders = train_test_split(video_folders, test_size=0.2, random_state=42)

train_dataset = create_dataset(train_folders, BATCH_SIZE)
val_dataset = create_dataset(val_folders, BATCH_SIZE)

steps_per_epoch = len(train_folders) // BATCH_SIZE
validation_steps = len(val_folders) // BATCH_SIZE

# -------------------------------
# ConvLSTM Autoencoder
# -------------------------------
def build_convlstm_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    x = ConvLSTM2D(16, (3, 3), padding="same", return_sequences=True, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((1, 2, 2))(x)

    x = ConvLSTM2D(16, (3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling3D((1, 2, 2))(x)

    x = Conv3D(3, (3, 3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_convlstm_autoencoder((FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))
model.summary()

# -------------------------------
# Callbacks
# -------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# -------------------------------
# Training
# -------------------------------
model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_scheduler]
)

# -------------------------------
# Save model
# -------------------------------
model.save("animal_behavior_autoencoder_optimized.keras")
