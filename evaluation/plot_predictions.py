import json
import os # Added import
import cv2
import numpy as np
import matplotlib.pyplot as plt
from neural_network import GazeToScreenModel # Assuming this is in the same directory or accessible

# Get the absolute path to the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Added for robust paths

# --- Configuration (should match your main.py and calibration.py) ---
SCREEN_WIDTH = 1920  # Your screen resolution
SCREEN_HEIGHT = 1080 # Your screen resolution
CAMERA_WIDTH = 640   # Width of the camera frame used by the model
CAMERA_HEIGHT = 480  # Height of the camera frame used by the model
CALIBRATION_DATA_FILE = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "data", "training", "training-data-0336.json")) # Modified for robustness
MODEL_FILE_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "gaze-model.joblib")) # Modified for robustness # Default path for the trained model

def load_calibration_data(filepath):
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

def simple_scale_gaze_to_screen(gaze_cam_x, gaze_cam_y, camera_width, camera_height, screen_width, screen_height):
    if gaze_cam_x is None or gaze_cam_y is None:
        return None, None
    if camera_width <= 0 or camera_height <= 0:
        return None, None
        
    normalized_gaze_x = gaze_cam_x / camera_width
    normalized_gaze_y = gaze_cam_y / camera_height
    
    screen_x = int(max(0, min(normalized_gaze_x * screen_width, screen_width - 1)))
    screen_y = int(max(0, min(normalized_gaze_y * screen_height, screen_height - 1)))
    return screen_x, screen_y

def analyze_data(calibration_data, model):
    target_points_for_stats = [] # Keep original target points for error calculation
    predicted_points_model_for_stats = []
    errors_model = []

    # Data for plotting
    plot_target_coords = []
    plot_target_colors = []
    
    plot_model_coords = []
    plot_model_colors = []
    
    # For drawing lines: list of (start_point, end_point, color)
    line_segments_model = []

    if not calibration_data:
        print("No calibration data to analyze.")
        return

    # Generate a unique color for each calibration entry/target
    num_entries = len(calibration_data)
    # Using a colormap like 'hsv' or 'tab20' can provide distinct colors.
    # 'hsv' provides a good spectrum. Using 0.9 for linspace end to avoid color wrap-around for some maps.
    entry_colors = plt.cm.hsv(np.linspace(0, 0.9, num_entries)) if num_entries > 0 else []


    for i, entry in enumerate(calibration_data):
        current_color = entry_colors[i] if num_entries > 0 else 'black' # Default color if no entries
        
        target_px = entry.get("target_screen_px")
        raw_gaze_px = entry.get("raw_gaze_camera_px")
        
        pupil_norm = entry.get("normalized_pupil_coord_xy")
        if pupil_norm is None:
            pupil_norm = entry.get("avg_normalized_pupil_coord_xy")
            
        rvec = entry.get("head_pose_rvec")
        if rvec is None:
            rvec = entry.get("avg_head_pose_rvec")
            
        tvec = entry.get("head_pose_tvec")
        if tvec is None:
            tvec = entry.get("avg_head_pose_tvec")
            
        # samples = entry.get("samples") # samples not used in current logic, but good to have if needed

        if not all(isinstance(c, (int, float)) for c in target_px if c is not None) or len(target_px) != 2:
            print(f"Skipping entry {i}: Invalid target_screen_px: {target_px}")
            continue
        
        # Store for plotting
        plot_target_coords.append(target_px)
        plot_target_colors.append(current_color)
        
        # Store for stats (original logic)
        target_points_for_stats.append(target_px)


        # Get predictions from the model if available and features are present
        model_pred_x, model_pred_y = None, None
        if model and model.is_trained:
            if raw_gaze_px and pupil_norm and rvec and tvec and \
               all(v is not None for v in raw_gaze_px) and \
               all(v is not None for v in pupil_norm) and \
               all(v is not None for v in rvec) and \
               all(v is not None for v in tvec):
                model_pred_x, model_pred_y = model.predict(
                    raw_gaze_px[0], raw_gaze_px[1],
                    pupil_norm[0], pupil_norm[1],
                    rvec, tvec
                )
            elif raw_gaze_px and all(v is not None for v in raw_gaze_px): # Fallback
                 model_pred_x, model_pred_y = model.predict(raw_gaze_px[0], raw_gaze_px[1])

        if model_pred_x is not None and model_pred_y is not None:
            model_coord = (model_pred_x, model_pred_y)
            plot_model_coords.append(model_coord)
            plot_model_colors.append(current_color)
            line_segments_model.append((target_px, model_coord, current_color))
            
            # For stats
            predicted_points_model_for_stats.append(model_coord) # Keep this for 1-to-1 with target_points_for_stats if needed
            errors_model.append(np.linalg.norm(np.array(target_px) - np.array(model_coord)))
        else:
            # For stats, if a target had no valid model prediction
            predicted_points_model_for_stats.append((None, None))


    # --- Visualization ---
    plt.figure(figsize=(12, 7))
    plt.gca().invert_yaxis() 
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(SCREEN_HEIGHT, 0) 
    plt.title("Calibration Data Analysis: Target vs. Gaze (Color-Coded by Target)")
    plt.xlabel("Screen X Coordinate")
    plt.ylabel("Screen Y Coordinate")
    plt.grid(True)

    # Plot target points
    if plot_target_coords:
        targets_np = np.array(plot_target_coords)
        plt.scatter(targets_np[:, 0], targets_np[:, 1], c=plot_target_colors, marker='x', s=100, label='Target Points')

    # Plot model-predicted points and lines
    if plot_model_coords:
        model_preds_np = np.array(plot_model_coords)
        plt.scatter(model_preds_np[:, 0], model_preds_np[:, 1], c=plot_model_colors, marker='o', s=70, alpha=0.7, label='Model Predicted Gaze')
        for start_pt, end_pt, color in line_segments_model:
            plt.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color=color, linestyle='-', alpha=0.3)

    plt.legend()
    plt.axis('equal') 
    plt.show()

    # --- Print Statistics ---
    # Statistics calculation remains the same, using errors_model
    if errors_model:
        print(f"\n--- Model Prediction Statistics ---")
        # Using len(calibration_data) for total, as errors_model only counts valid predictions
        print(f"Number of points with model predictions: {len(errors_model)} / {len(target_points_for_stats)}")
        print(f"Average Error (pixels): {np.mean(errors_model):.2f}")
        print(f"Median Error (pixels): {np.median(errors_model):.2f}")
        print(f"Standard Deviation of Error (pixels): {np.std(errors_model):.2f}")
        print(f"Min Error (pixels): {np.min(errors_model):.2f}")
        print(f"Max Error (pixels): {np.max(errors_model):.2f}")
        
    if not errors_model:
        print("\nNo valid predictions could be made to calculate error statistics.")


def main():
    print("Starting calibration data analysis...")
    calibration_data = load_calibration_data(CALIBRATION_DATA_FILE)

    if not calibration_data:
        print("Exiting due to issues loading calibration data.")
        return

    # Initialize the GazeToScreenModel
    # It will attempt to load a pre-trained model if MODEL_FILE_PATH exists
    gaze_model = GazeToScreenModel(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        model_dir="." # Assuming model is in the current directory
    )

    if not gaze_model.is_trained:
        print(f"Warning: Model at '{MODEL_FILE_PATH}' is not trained or could not be loaded.")
        print("Analysis will proceed, but model predictions might be based on fallback or not available.")
    
    analyze_data(calibration_data, gaze_model)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
