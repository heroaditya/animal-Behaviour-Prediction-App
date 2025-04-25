# Enhanced Wildlife Monitoring System

This system is designed to help wildlife researchers monitor animal behavior near water bodies using advanced deep learning techniques and computer vision.

## Features

- Animal detection using a pre-trained Keras model
- Behavior prediction using transformer models and ensemble methods
- Water interaction analysis for tracking animal behavior near water bodies
- Video, image, and GPS data analysis
- Interactive web interface for uploading and analyzing data
- Comprehensive visualization of results

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
3. Ensure you have the pre-trained animal detection model:
   - Place `myModel.keras` in the root directory (trained on animals-detection-images-dataset)
4. Run the preprocessing script to prepare the data:
   \`\`\`
   python preprocessing.py
   \`\`\`
5. Train the behavior prediction models:
   \`\`\`
   python train_model.py
   \`\`\`
6. Start the web application:
   \`\`\`
   python main_app.py
   \`\`\`
7. Open your browser and navigate to `http://localhost:5000`

## Model Architecture

The system uses multiple models for different tasks:

1. **Animal Detection**: Pre-trained Keras model (`myModel.keras`) for detecting animals in images and videos
2. **Behavior Prediction**: 
   - Transformer model with multi-head attention for sequence analysis
   - Deep neural network for feature-based classification
   - Bidirectional LSTM/GRU for temporal pattern recognition
   - Stacked ensemble combining XGBoost, LightGBM, and Random Forest

3. **Water Interaction Analysis**: Specialized models for detecting and analyzing animal behavior near water bodies

## Data Processing

The system processes three types of data:

1. **Video Data**: Extracts frames, detects animals, and analyzes behavior patterns over time
2. **Image Data**: Detects animals and provides basic classification
3. **GPS Data**: Analyzes movement patterns, identifies behavior types, and detects water interaction points

## Behavior Categories

The system classifies animal behaviors into the following categories:

- Unknown (0)
- Resting (1)
- Foraging (2)
- Traveling (3)
- Diving/Swimming (4)

## Analysis Types

The system supports three types of analysis:

1. **Animal Detection**: Basic detection and classification of animals
2. **Behavior Analysis**: Comprehensive analysis of animal behavior patterns
3. **Water Interaction**: Specialized analysis focusing on animal behavior near water bodies

## Acknowledgments

- Animal detection dataset: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset 
- Animal detection model based on: https://www.kaggle.com/code/nimapourmoradi/animal-detection/notebook (One ca download the MyModel.keras file that can be utiized to predict the animal prediction behaviour.)
- Animal Behaviour Dataset Link: https://www.kaggle.com/datasets/sttaseen/animal-behaviour 
- Behavior prediction inspired by: https://github.com/robot-perception-group/animal-behaviour-inference
