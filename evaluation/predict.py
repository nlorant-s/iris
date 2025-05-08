import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from neural_network import GazeToScreenModel # Assuming neural_network.py is in the same directory

# --- Configuration ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CALIBRATION_DATA_FILE = "../data/training/training-data-0251.json" # Corrected path
TEMP_TRAIN_DATA_FILE = "../data/training/temp_train_data_for_predict.json" # Corrected path
# Use a different model path for this script to avoid overwriting the main one
MODEL_FILE_PATH_PREDICT = "../data/models/gaze_model_predict.joblib" # Corrected path

def load_calibration_data(filepath):
    """Loads calibration data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded calibration data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: Calibration data file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def calculate_prediction_errors(test_data, model):
    """Calculates prediction errors on the test set."""
    errors = []
    if not model.is_trained:
        print("Model is not trained. Cannot calculate errors.")
        return errors

    print(f"\\n--- Evaluating on Test Set ({len(test_data)} points) ---")
    for i, entry in enumerate(test_data):
        target_px = entry.get("target_screen_px")
        raw_gaze_px = entry.get("raw_gaze_camera_px")
        pupil_norm = entry.get("avg_normalized_pupil_coord_xy")
        rvec = entry.get("avg_head_pose_rvec")
        tvec = entry.get("avg_head_pose_tvec")

        if not target_px or len(target_px) != 2 or not all(isinstance(c, (int, float)) for c in target_px):
            print(f"Skipping test entry {i}: Invalid target_screen_px: {target_px}")
            continue

        pred_x, pred_y = None, None
        # Try to predict with all features first
        if raw_gaze_px and pupil_norm and rvec and tvec and \
           all(v is not None for v in raw_gaze_px) and \
           all(v is not None for v in pupil_norm) and \
           all(v is not None for v in rvec) and \
           all(v is not None for v in tvec):
            pred_x, pred_y = model.predict(
                raw_gaze_px[0], raw_gaze_px[1],
                pupil_norm[0], pupil_norm[1],
                rvec, tvec
            )
        # Fallback to raw_gaze_px only if other features are missing
        elif raw_gaze_px and all(v is not None for v in raw_gaze_px):
            print(f"Test entry {i}: Using fallback prediction (raw_gaze_px only).")
            pred_x, pred_y = model.predict(raw_gaze_px[0], raw_gaze_px[1])
        else:
            print(f"Skipping test entry {i}: Insufficient data for prediction.")
            continue
        
        if pred_x is not None and pred_y is not None:
            prediction_px = np.array([pred_x, pred_y])
            target_px_np = np.array(target_px)
            error = np.linalg.norm(target_px_np - prediction_px)
            errors.append(error)
            # print(f"Entry {i}: Target: {target_px}, Predicted: ({pred_x:.2f}, {pred_y:.2f}), Error: {error:.2f}px")
        else:
            print(f"Skipping test entry {i}: Prediction failed.")
            
    return errors

def main():
    print("Starting prediction error calculation script...")
    calibration_data = load_calibration_data(CALIBRATION_DATA_FILE)

    if not calibration_data or len(calibration_data) < 2: # Need at least 2 samples for train/test split
        print("Exiting: Insufficient calibration data loaded or data is empty.")
        return

    # Split data
    try:
        train_data, test_data = train_test_split(calibration_data, test_size=0.2, random_state=42)
        print(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples.")
    except ValueError as e:
        print(f"Error during data splitting: {e}. Ensure you have enough samples.")
        return

    if not train_data:
        print("Exiting: No data in the training set after split.")
        return
    if not test_data:
        print("Exiting: No data in the test set after split.")
        return

    # Write training data to a temporary file
    try:
        with open(TEMP_TRAIN_DATA_FILE, 'w') as f:
            json.dump(train_data, f)
        print(f"Temporary training data saved to {TEMP_TRAIN_DATA_FILE}")
    except IOError as e:
        print(f"Error writing temporary training data: {e}")
        return
        
    # Initialize and train the model
    # The model will save to its own MODEL_FILENAME when train() is called.
    # We can specify a different model_path if GazeToScreenModel's constructor supports it,
    # or ensure GazeToScreenModel.MODEL_FILENAME is configured not to overwrite critical files.
    # For this script, we'll let it use its default, or a custom one if we modify GazeToScreenModel.
    
    # Check if a model specific to this script exists and remove it to ensure fresh training
    if os.path.exists(MODEL_FILE_PATH_PREDICT):
        print(f"Removing existing prediction-specific model: {MODEL_FILE_PATH_PREDICT}")
        try:
            os.remove(MODEL_FILE_PATH_PREDICT)
        except OSError as e:
            print(f"Error removing existing model file {MODEL_FILE_PATH_PREDICT}: {e}")
            # Decide if we should proceed or exit
            # For now, we'll proceed, but training might use an old model if not careful

    gaze_model = GazeToScreenModel(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        model_path=MODEL_FILE_PATH_PREDICT # Pass the custom model path
    )
    
    print("\\n--- Training Model for Prediction Script ---")
    gaze_model.train(TEMP_TRAIN_DATA_FILE) # Train on the 80%

    if gaze_model.is_trained:
        print("Model training complete.")
        errors = calculate_prediction_errors(test_data, gaze_model)

        if errors:
            print("\\n--- Prediction Error Statistics (on 20% test set) ---")
            print(f"Number of test points evaluated: {len(errors)} / {len(test_data)}")
            print(f"Average Error (pixels): {np.mean(errors):.2f}")
            print(f"Median Error (pixels): {np.median(errors):.2f}")
            print(f"Standard Deviation of Error (pixels): {np.std(errors):.2f}")
            print(f"Min Error (pixels): {np.min(errors):.2f}")
            print(f"Max Error (pixels): {np.max(errors):.2f}")
        else:
            print("\\nNo errors calculated. Check test data or model training.")
    else:
        print("Model training failed. Cannot proceed with error calculation.")

    # Clean up temporary training file
    try:
        if os.path.exists(TEMP_TRAIN_DATA_FILE):
            os.remove(TEMP_TRAIN_DATA_FILE)
            print(f"Temporary training data file {TEMP_TRAIN_DATA_FILE} removed.")
    except OSError as e:
        print(f"Error removing temporary training data file: {e}")
    
    # Optionally, remove the model created by this script if it's not needed afterwards
    # if os.path.exists(MODEL_FILE_PATH_PREDICT):
    #     os.remove(MODEL_FILE_PATH_PREDICT)
    #     print(f"Prediction-specific model file {MODEL_FILE_PATH_PREDICT} removed.")

    print("\\nPrediction script finished.")

if __name__ == "__main__":
    main()
