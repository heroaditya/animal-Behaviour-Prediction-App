# app.py - Unified Flask App with All Models Integrated
import os
import uuid
import traceback
import json
import numpy as np
import pandas as pd
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import Counter

# Flask Setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Dummy behavior categories
behavior_categories = {
    0: "Normal",
    1: "Aggressive",
    2: "Chasing",
    3: "Fighting",
    4: "Eating",
    5: "Resting"
}

# The class_indices you got from flow_from_directory
class_indices = {
    'Spider': 0, 'Parrot': 1, 'Scorpion': 2, 'Sea turtle': 3, 'Cattle': 4, 'Fox': 5, 'Hedgehog': 6,
    'Turtle': 7, 'Cheetah': 8, 'Snake': 9, 'Shark': 10, 'Horse': 11, 'Magpie': 12, 'Hamster': 13,
    'Woodpecker': 14, 'Eagle': 15, 'Penguin': 16, 'Butterfly': 17, 'Lion': 18, 'Otter': 19,
    'Raccoon': 20, 'Hippopotamus': 21, 'Bear': 22, 'Chicken': 23, 'Pig': 24, 'Owl': 25,
    'Caterpillar': 26, 'Koala': 27, 'Polar bear': 28, 'Squid': 29, 'Whale': 30, 'Harbor seal': 31,
    'Raven': 32, 'Mouse': 33, 'Tiger': 34, 'Lizard': 35, 'Ladybug': 36, 'Red panda': 37,
    'Kangaroo': 38, 'Starfish': 39, 'Worm': 40, 'Tortoise': 41, 'Ostrich': 42, 'Goldfish': 43,
    'Frog': 44, 'Swan': 45, 'Elephant': 46, 'Sheep': 47, 'Snail': 48, 'Zebra': 49,
    'Moths and butterflies': 50, 'Shrimp': 51, 'Fish': 52, 'Panda': 53, 'Lynx': 54, 'Duck': 55,
    'Jaguar': 56, 'Goose': 57, 'Goat': 58, 'Rabbit': 59, 'Giraffe': 60, 'Crab': 61, 'Tick': 62,
    'Monkey': 63, 'Bull': 64, 'Seahorse': 65, 'Centipede': 66, 'Mule': 67, 'Rhinoceros': 68,
    'Canary': 69, 'Camel': 70, 'Brown bear': 71, 'Sparrow': 72, 'Squirrel': 73, 'Leopard': 74,
    'Jellyfish': 75, 'Crocodile': 76, 'Deer': 77, 'Turkey': 78, 'Sea lion': 79
}

# Generate class_labels based on sorted index
class_labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_labels[idx] = label


emoji_map = {
    "Bear": "🐻",
    "Brown bear": "🐻‍❄️",
    "Bull": "🐂",
    "Butterfly": "🦋",
    "Camel": "🐫",
    "Canary": "🐤",
    "Caterpillar": "🐛",
    "Cattle": "🐄",
    "Centipede": "🐛",
    "Cheetah": "🐆",
    "Chicken": "🐔",
    "Crab": "🦀",
    "Crocodile": "🐊",
    "Deer": "🦌",
    "Duck": "🦆",
    "Eagle": "🦅",
    "Elephant": "🐘",
    "Fish": "🐟",
    "Fox": "🦊",
    "Frog": "🐸",
    "Giraffe": "🦒",
    "Goat": "🐐",
    "Goldfish": "🐠",
    "Goose": "🪿",
    "Hamster": "🐹",
    "Harbor seal": "🦭",
    "Hedgehog": "🦔",
    "Hippopotamus": "🦛",
    "Horse": "🐴",
    "Jaguar": "🐆",
    "Jellyfish": "🎐",  # closest available
    "Kangaroo": "🦘",
    "Koala": "🐨",
    "Ladybug": "🐞",
    "Leopard": "🐆",
    "Lion": "🦁",
    "Lizard": "🦎",
    "Lynx": "🐱",
    "Magpie": "🐦",
    "Monkey": "🐒",
    "Moths and butterflies": "🦋",
    "Mouse": "🐭",
    "Mule": "🐴",
    "Ostrich": "🦤",  # closest match
    "Otter": "🦦",
    "Owl": "🦉",
    "Panda": "🐼",
    "Parrot": "🦜",
    "Penguin": "🐧",
    "Pig": "🐷",
    "Polar bear": "🐻‍❄️",
    "Rabbit": "🐰",
    "Raccoon": "🦝",
    "Raven": "🐦",
    "Red panda": "🦊",  # closest
    "Rhinoceros": "🦏",
    "Scorpion": "🦂",
    "Sea lion": "🦭",
    "Sea turtle": "🐢",
    "Seahorse": "🐎",  # closest (or use emoji-less fallback)
    "Shark": "🦈",
    "Sheep": "🐑",
    "Shrimp": "🦐",
    "Snail": "🐌",
    "Snake": "🐍",
    "Sparrow": "🐦",
    "Spider": "🕷️",
    "Squid": "🦑",
    "Squirrel": "🐿️",
    "Starfish": "🌟",  # closest
    "Swan": "🦢",
    "Tick": "🦗",  # closest
    "Tiger": "🐯",
    "Tortoise": "🐢",
    "Turkey": "🦃",
    "Turtle": "🐢",
    "Whale": "🐋",
    "Woodpecker": "🐦",
    "Worm": "🪱",
    "Zebra": "🦓"
}
label_to_emoji = {label: emoji_map.get(label, "") for label in class_labels}

# Load all models at startup
try:
    animal_detection_model = load_model('C:/Users/91843/Desktop/wildlife-monitoring-system (1)/MyModel.keras')
    behavior_prediction_model = load_model('C:/Users/91843/Desktop/wildlife-monitoring-system (1)/animal_behavior_autoencoder_optimized.keras')
    csv_model = load_model("C:/Users/91843/Desktop/wildlife-monitoring-system (1)/models/csv_model.h5")
    csv_scaler = joblib.load("C:/Users/91843/Desktop/wildlife-monitoring-system (1)/models/csv_scaler.save")
    print("[INFO] All models loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load models:", str(e))
    # Log more details for each model loading step
    if "MyModel.keras" not in str(e):
        print("[ERROR] Failed to load animal_detection_model")
    if "animal_behavior_autoencoder_optimized.keras" not in str(e):
        print("[ERROR] Failed to load behavior_prediction_model")
    if "model/csv_model.h5" not in str(e):
        print("[ERROR] Failed to load csv_model")
    if "model/csv_scalar.save" not in str(e):
        print("[ERROR] Failed to load csv_scaler")
    animal_detection_model = None
    behavior_prediction_model = None
    csv_model = None
    csv_scaler = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    analysis_type = request.form.get('analysis_type')

    if file:
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        media_path = url_for('static', filename=f"uploads/{filename}")

        try:
            if analysis_type == 'animal_detection':
                result_data = run_animal_detection(filepath)
            elif analysis_type == 'behavior_analysis':
                result_data = run_behavior_analysis(filepath)
            elif analysis_type == 'csv_behavior':
                result_data = run_csv_prediction(filepath)
            elif analysis_type == 'video_behavior':
                result_data = run_video_behavior(filepath)
            else:
                flash("Unsupported analysis type.")
                return redirect(url_for('index'))

            result_data["media_path"] = media_path
            result_data["type"] = analysis_type 

            result_id = str(uuid.uuid4())
            result_path = f"static/results/{result_id}.json"
            with open(result_path, 'w') as f:
                json.dump(result_data, f)

            return redirect(url_for('view_results', result_id=result_id))

        except Exception as e:
            traceback.print_exc()
            flash(f"Error during analysis: {str(e)}")
            return redirect(url_for('index'))
        

@app.route('/results/<result_id>')
def view_results(result_id):
    result_path = f"static/results/{result_id}.json"
    if not os.path.exists(result_path):
        flash("Result not found.")
        return redirect(url_for('index'))
    with open(result_path) as f:
        result_data = json.load(f)
    return render_template('results.html', result=result_data)

# ------------ Model Runners ------------------

# ------------ Helper Function for MSE Interpretation ------------------
def interpret_mse(mse):
    if mse < 0.005:
        return "✅ Normal behavior detected.", "High", "Behavior is consistent with what the model has seen during training."
    elif mse < 0.01:
        return "⚠️ Slightly unusual behavior detected.", "Medium", "There might be a deviation from normal behavior patterns."
    else:
        return "🚨 Unusual or aggressive behavior detected!", "Low", "The model found significant differences from expected behavior."


def run_animal_detection(image_path):
    if animal_detection_model is None:
        raise RuntimeError("Animal detection model is not loaded.")

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = animal_detection_model.predict(img_array)
    pred_index = np.argmax(pred)

    label = class_labels[pred_index]
    emoji = label_to_emoji.get(label, "")
    confidence = float(pred[0][pred_index])

    filename = os.path.basename(image_path)

    return {
        "type": "animal_detection",
        "prediction": f"{label} {emoji}",
        "confidence": confidence,
        "media_path": url_for('static', filename='uploads/' + filename)
    }

def run_behavior_analysis(image_path):
    if behavior_prediction_model is None:
        raise RuntimeError("Behavior prediction model is not loaded.")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48)) / 255.0
    sequence = np.expand_dims(np.array([img] * 4), axis=0)
    reconstructed = behavior_prediction_model.predict(sequence)
    error = np.mean((sequence - reconstructed) ** 2)
    
    label, confidence, comment = interpret_mse(error)
    return {
        "type": "behavior_analysis",
        "reconstruction_mse": float(error),
        "interpretation": label,
        "confidence": confidence,
        "explanation": comment
    }

def run_csv_prediction(input_csv_path):
    try:
        # Load model components
        label_encoder = joblib.load('models/label_encoder.save')
        expected_features = joblib.load('models/csv_feature_columns.save')
        label_map = joblib.load('models/csv_label_map.save')  # {0: 0.0, 1: 1.0, ...}

        # Define readable behavior names
        BEHAVIOR_CLASSES = {
            0.0: 'Unknown/Complex/Exploratory Behaviour',
            1.0: 'Resting Behaviour',
            2.0: 'Foraging Behaviour',
            3.0: 'Travelling Behaviour',
            4.0: 'Diving Behaviour'
        }

        # Load input CSV
        input_df = pd.read_csv(input_csv_path, low_memory=False)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        missing = set(expected_features) - set(input_df.columns)
        if missing:
            raise ValueError(f"Missing expected columns in CSV: {missing}")
        X_input = input_df[expected_features].fillna(0)

        # Prediction
        X_scaled = csv_scaler.transform(X_input)
        predictions = csv_model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)

        # Convert predicted index → float value → human-readable label
        readable_labels = []
        for cls_idx in predicted_classes:
            class_value = label_map.get(cls_idx)  # e.g., 0.0, 1.0, ...
            readable_label = BEHAVIOR_CLASSES.get(class_value, f"Behavior {class_value}")
            readable_labels.append(readable_label)

        # Summary
        from collections import Counter
        summary = dict(Counter(readable_labels))
        bar_plot = plot_to_image(summary)

        # Natural language summary
        top_behavior = max(summary, key=summary.get)
        top_count = summary[top_behavior]
        total = sum(summary.values())
        other_behaviors = [label for label in summary if label != top_behavior]

        nl_summary = f"🔍 Most common observed behavior is **{top_behavior}** with {top_count} instances."
        if other_behaviors:
            nl_summary += f" The CSV also includes behaviors like: {', '.join(other_behaviors)}."

        return {
            "summary": summary,
            "bar_plot": bar_plot,
            "natural_language_summary": nl_summary
        }

    except Exception as e:
        print(f"[ERROR] CSV Prediction failed: {e}")
        return {
            "summary": {"Error": 1},
            "bar_plot": None,
            "natural_language_summary": "⚠️ An error occurred during prediction."
        }

    
def run_video_behavior(video_path):
    if behavior_prediction_model is None:
        raise RuntimeError("Behavior prediction model is not loaded.")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 4 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (48, 48)) / 255.0
        frames.append(frame)
    cap.release()
    while len(frames) < 4:
        frames.append(np.zeros((48, 48, 3)))
    sequence = np.expand_dims(np.array(frames), axis=0)
    reconstructed = behavior_prediction_model.predict(sequence)
    error = np.mean((sequence - reconstructed) ** 2)

    label, confidence, comment = interpret_mse(error)
    return {
        "type": "video_behavior",
        "reconstruction_mse": float(error),
        "interpretation": label,
        "confidence": confidence,
        "explanation": comment
    }

# ------------ Visualization ------------------
def plot_to_image(data_dict):
    plt.figure(figsize=(6, 3))
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    plt.bar(labels, values, color='skyblue')
    plt.title("Behavior Predictions")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_b64}"

# ------------- Run Flask ----------------------
if __name__ == '__main__':
    app.run(debug=True)