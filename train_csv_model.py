import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models, callbacks
import joblib

# Ensure 'models' directory exists
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('combined_data.csv')

# Drop unneeded cols and encode target
X = df.select_dtypes(include=[np.number]).drop(['behavior_code'], axis=1)
# Save the feature columns used during training
joblib.dump(X.columns.tolist(), 'models/csv_feature_columns.save')
le = LabelEncoder()
y = le.fit_transform(df['behavior_code'])
# Save label mapping for readability
label_map = dict(zip(le.transform(le.classes_), le.classes_))
joblib.dump(label_map, 'models/csv_label_map.save')

# Save LabelEncoder
joblib.dump(le, 'models/label_encoder.save')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
joblib.dump(scaler, 'models/csv_scaler.save')

# Model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32,
          callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

model.save('models/csv_model.h5')
