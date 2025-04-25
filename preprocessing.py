import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Define file paths for datasets
data_paths = {
    'water_quality': 'Datasets/water_potability.csv',
    'animal_migration': 'Datasets/migration_original.csv',
    'bird_migration': 'Datasets/bird_migration.csv',
    'seabird_migration': 'Datasets/anon_gps_tracks_with_dive.csv',
    'ponds': 'Datasets/Ponds Data/Ponds.csv',
    'ponds1': 'Datasets/Ponds Data/Ponds1.csv',
    'soil_moisture_1': 'Datasets/Soil_Moisture_Dataset/plant_vase1.CSV',
    'soil_moisture_2': 'Datasets/Soil_Moisture_Dataset/plant_vase1(2).CSV',
    'soil_moisture_3': 'Datasets/Soil_Moisture_Dataset/plant_vase2.CSV'
}

# Define label columns for each dataset
label_columns = {
    'water_quality': ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability'],
    'animal_migration': ["event-id","visible","timestamp","location-long","location-lat","manually-marked-outlier","visibility","sensor-type","individual-taxon-canonical-name","tag-local-identifier","individual-local-identifier","study-name","ECMWF Interim Full Daily Invariant Low Vegetation Cover","NCEP NARR SFC Vegetation at Surface","ECMWF Interim Full Daily Invariant High Vegetation Cover"],
    'bird_migration': ['Sr','altitude','date_time','device_info_serial','direction','latitude','longitude','speed_2d','bird_name'],
    'ponds': ['Station', 'Date', 'NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'],
    'ponds1': ['station', 'Date', 'Time', 'NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'],
    'soil_moisture_1': ['year', 'month', 'day', 'hour', 'minute', 'second', 'moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4', 'irrgation'],
    'soil_moisture_2': ['year', 'month', 'day', 'hour', 'minute', 'second', 'moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4', 'irrgation'],
    'soil_moisture_3': ['year', 'month', 'day', 'hour', 'minute', 'second', 'moisture0', 'moisture1', 'moisture2', 'moisture3', 'moisture4', 'irrgation'],
    'seabird_migration': ["Sr.","lat","lon","alt","unix","bird","species","year","date_time","max_depth.m","colony2","coverage_ratio","is_dive","is_dive_1m","is_dive_2m","is_dive_4m","is_dive_5m","is_dive_0m"]
}

# Create necessary directories
os.makedirs('Datasets', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

print("Starting data preprocessing...")

# Function to check if file exists
def check_file_exists(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} does not exist.")
        return False
    return True

# Function to extract water-related features
def extract_water_features(df):
    """Extract features related to water proximity and interaction"""
    water_features = {}
    
    # Check for water quality indicators
    for col in ['ph', 'PH', 'Turbidity', 'TURBIDITY', 'DO']:
        if col in df.columns:
            water_features[col] = True
    
    # Check for proximity to water bodies (if GPS coordinates are available)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # This is a placeholder - in a real system, you would use a GIS database
        # to check proximity to known water bodies
        water_features['has_coordinates'] = True
    
    # Check for diving behavior in seabirds
    for col in ['is_dive', 'is_dive_1m', 'is_dive_2m', 'is_dive_4m', 'is_dive_5m']:
        if col in df.columns:
            water_features['has_dive_data'] = True
    
    return water_features

# Function to create behavior labels
def create_behavior_labels(df, water_features):
    """Create behavior labels based on available data"""
    
    # Initialize behavior column
    df['water_interaction'] = 0
    
    # If dive data is available, use it directly
    if 'has_dive_data' in water_features and 'is_dive' in df.columns:
        df['water_interaction'] = df['is_dive'].astype(int)
    
    # If we have water quality data and animal movement data, infer interaction
    elif 'has_coordinates' in water_features and any(col in df.columns for col in ['speed_2d', 'altitude']):
        # Simplified logic: if animal is moving slowly and near water, assume interaction
        if 'speed_2d' in df.columns:
            slow_movement = df['speed_2d'] < df['speed_2d'].quantile(0.3)
            df.loc[slow_movement, 'water_interaction'] = 1
    
    # Create behavior categories
    df['behavior_category'] = 'unknown'
    
    # Classify behaviors based on available features
    if 'speed_2d' in df.columns:
        # Fast movement
        df.loc[df['speed_2d'] > df['speed_2d'].quantile(0.7), 'behavior_category'] = 'traveling'
        # Medium movement
        df.loc[(df['speed_2d'] <= df['speed_2d'].quantile(0.7)) & 
               (df['speed_2d'] > df['speed_2d'].quantile(0.3)), 'behavior_category'] = 'foraging'
        # Slow movement
        df.loc[df['speed_2d'] <= df['speed_2d'].quantile(0.3), 'behavior_category'] = 'resting'
    
    # If we have dive data, override with more specific behaviors
    if 'is_dive' in df.columns:
        df.loc[df['is_dive'] == 1, 'behavior_category'] = 'diving'
    
    # Convert behavior categories to numeric codes
    behavior_mapping = {
        'unknown': 0,
        'resting': 1,
        'foraging': 2,
        'traveling': 3,
        'diving': 4
    }
    
    df['behavior_code'] = df['behavior_category'].map(behavior_mapping)
    
    return df

# Load and preprocess each dataset
dataframes = []

for name, path in data_paths.items():
    try:
        # Check if file exists
        if not check_file_exists(path):
            continue
            
        df = pd.read_csv(path)
        print(f"Loaded {name} with shape {df.shape}")
        
        # Determine which label columns to use
        if 'soil_moisture' in name:
            dataset_type = 'soil_moisture_1'
        else:
            dataset_type = name.split('_')[0] if '_' in name else name
            
        # Get expected labels for this dataset type
        if dataset_type in label_columns:
            expected_labels = label_columns[dataset_type]
        else:
            print(f"Warning: No label columns defined for {dataset_type}. Using original columns.")
            expected_labels = df.columns.tolist()
        
        # Adjust columns safely by checking for label-column mismatch
        if len(df.columns) != len(expected_labels):
            print(f"Warning: Column mismatch in {name}. Found {len(df.columns)}, expected {len(expected_labels)}.")
            # Use original columns if mismatch is too large
            if abs(len(df.columns) - len(expected_labels)) > 5:
                print(f"Using original columns for {name} due to large mismatch.")
            else:
                # Try to adjust columns to match expected length
                if len(df.columns) > len(expected_labels):
                    df = df.iloc[:, :len(expected_labels)]
                df.columns = expected_labels[:len(df.columns)]
        else:
            df.columns = expected_labels
        
        # Resolve duplicates by adding a suffix with dataset name
        df.columns = pd.Index([
            f"{col}_{name}" if col in df.columns[df.columns.duplicated()] else col
            for col in df.columns
        ])

        # Add source column
        df['source'] = name  # Track source dataset
        
        # Extract water-related features
        water_features = extract_water_features(df)
        print(f"Water features for {name}: {water_features}")
        
        # Create behavior labels
        if len(water_features) > 0:
            df = create_behavior_labels(df, water_features)
            print(f"Created behavior labels for {name}")
        
        dataframes.append(df.reset_index(drop=True))
        print(f"Successfully processed {name}")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Combine all datasets into one if any were successfully loaded
if dataframes:
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    # Handle missing values
    numeric_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_combined.select_dtypes(include=['object']).columns
    
    df_combined[numeric_cols] = df_combined[numeric_cols].fillna(df_combined[numeric_cols].median())
    for col in categorical_cols:
        df_combined[col] = df_combined[col].fillna(df_combined[col].mode().iloc[0] if not df_combined[col].mode().empty else "Unknown")
    
    # Save the combined data
    df_combined.to_csv('combined_data.csv', index=False)
    print(f"Data combined and saved to 'combined_data.csv' with shape {df_combined.shape}")
    
    # Create a standardized dataset for deep learning models
    if 'behavior_code' in df_combined.columns:
        # Select relevant features
        feature_cols = []
        
        # Include movement features
        for col in ['speed_2d', 'altitude', 'direction']:
            if col in df_combined.columns:
                feature_cols.append(col)
        
        # Include location features
        for col in ['latitude', 'longitude', 'location-lat', 'location-long', 'lat', 'lon']:
            if col in df_combined.columns:
                feature_cols.append(col)
        
        # Include water quality features
        for col in ['ph', 'PH', 'Turbidity', 'TURBIDITY', 'DO', 'TEMP']:
            if col in df_combined.columns:
                feature_cols.append(col)
        
        # Include time features if available
        for col in ['timestamp', 'date_time', 'unix']:
            if col in df_combined.columns:
                feature_cols.append(col)
        
        # Create a standardized dataset with selected features
        if feature_cols:
            print(f"Creating standardized dataset with features: {feature_cols}")

            # Select features and target
            X = df_combined[feature_cols].copy()
            y = df_combined['behavior_code'].copy() if 'behavior_code' in df_combined.columns else None

            # Convert all feature columns to numeric, coerce errors to NaN
            X = X.apply(pd.to_numeric, errors='coerce')

            # Drop rows with NaN values (caused by '#VALUE!' or other non-numeric strings)
            if y is not None:
                valid_rows = X.dropna().index
                X = X.loc[valid_rows]
                y = y.loc[valid_rows]
            else:
                X = X.dropna()

            # ✅ Check if X is empty
            if X.empty:
                print("❌ No valid data found after cleaning! Check your dataset for non-numeric or missing values.")
            else:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

                # Ensure directories exist
                os.makedirs('processed_data', exist_ok=True)
                os.makedirs('models', exist_ok=True)

                # Save standardized data
                X_scaled.to_csv('processed_data/X_standardized.csv', index=False)
                if y is not None:
                    y.to_csv('processed_data/y_labels.csv', index=False)

                # Save the scaler
                joblib.dump(scaler, 'models/feature_scaler.pkl')

                print(f"✅ Standardization complete. Processed {len(X)} valid rows.")
    
    # Print column information
    print("\nColumn information:")
    for col in df_combined.columns:
        print(f"- {col}: {df_combined[col].dtype}")
else:
    print("No datasets were successfully processed. Check file paths and try again.")

# Process video data for behavior analysis if available
video_dir = 'archive/animal-kingdom/video'
if os.path.exists(video_dir):
    print("\nProcessing video data for behavior analysis...")
    
    # Create output directory for video frames
    frames_dir = 'processed_data/video_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Process each video file
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing video: {video_file}")
        
        # Create directory for this video's frames
        video_frames_dir = os.path.join(frames_dir, os.path.splitext(video_file)[0])
        os.makedirs(video_frames_dir, exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract frames at regular intervals
        frame_interval = max(1, int(fps))  # Extract 1 frame per second
        
        for i in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i % frame_interval == 0:
                frame_path = os.path.join(video_frames_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        cap.release()
        print(f"Extracted frames from {video_file} at {frame_interval} frame intervals")

print("Preprocessing complete!")
