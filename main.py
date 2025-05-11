import cv2
import pyautogui
import time
import vision.eyes as eyes
import vision.gaze as gaze
from evaluation.neural_network import GazeToScreenModel
import os
import argparse # Added for command-line arguments

# --- Configuration ---
SCREEN_WIDTH = 1920  # Your screen resolution
SCREEN_HEIGHT = 1080 # Your screen resolution
CAMERA_WIDTH = 640   # Width of the camera frame used by the model
CAMERA_HEIGHT = 480  # Height of the camera frame used by the model

# --- File Paths ---
# Get the directory of the current script (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAINING_DATA_FILE = os.path.join(BASE_DIR, "data", "training", "training-data-0336.json")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "evaluation") # Model is saved/loaded from evaluation directory
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_MODEL_DIR, GazeToScreenModel.MODEL_FILENAME) # e.g., evaluation/gaze_model.joblib

# Mouse movement smoothing factor (0.0 to 1.0). Lower values mean smoother but slower.
SMOOTHING_FACTOR = 0.05 # Adjust as needed
previous_mouse_x, previous_mouse_y = pyautogui.position()

def main_realtime_gaze_mouse(retrain_model=False, training_file=DEFAULT_TRAINING_DATA_FILE, model_file_path=DEFAULT_MODEL_FILE):
    global previous_mouse_x, previous_mouse_y

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Initialize the GazeToScreenModel
    # The model_path for GazeToScreenModel should point to the full path of gaze_model.joblib.
    model = GazeToScreenModel(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        model_path=model_file_path # Pass the full model file path
    )

    if retrain_model:
        print(f"--- Retraining model using {training_file} ---")
        if not os.path.exists(training_file):
            print(f"Error: Training data file not found: {training_file}")
            print("Please run calibration first or ensure the file path is correct.")
            # Decide if to proceed with an untrained/previously loaded model or exit
            if not model.is_trained:
                print("Model is not trained and retraining failed. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                return
            else:
                print("Proceeding with the previously loaded model.")
        elif model.train(training_file): # train method saves the model to model.model_path
            print(f"Model retraining successful. Model saved to {model.model_path}")
        else:
            print("Model retraining failed.")
            if not model.is_trained:
                print("Model is not trained and retraining failed. Predictions will use simple scaling.")
            else:
                print("Proceeding with the previously loaded model.")
    elif not model.is_trained:
        print("Model is not trained and no retraining was requested.")
        print(f"Attempting to train with default data: {DEFAULT_TRAINING_DATA_FILE}")
        if os.path.exists(DEFAULT_TRAINING_DATA_FILE):
            if model.train(DEFAULT_TRAINING_DATA_FILE):
                print("Initial model training successful.")
            else:
                print("Initial model training failed. Predictions will use simple scaling.")
        else:
            print(f"Default training file {DEFAULT_TRAINING_DATA_FILE} not found. Predictions will use simple scaling.")

    # PyAutoGUI setup
    pyautogui.FAILSAFE = False # Be cautious: disables the failsafe (moving mouse to corner to stop)
                               # Consider keeping it True or implementing your own safety mechanism.
    pyautogui.PAUSE = 0    # No automatic pause after each PyAutoGUI call

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam. Exiting.")
                break

            # 1. Detect eye regions
            # We pass a copy of the frame to avoid drawing on the frame used for detection
            # Convert frame to RGB for MediaPipe
            rgb_frame_for_detection = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            detected_faces_data = eyes.get_eye_regions(rgb_frame_for_detection)

            # 2. Determine gaze features (raw camera coordinates, normalized pupil, head pose)
            gaze_features = gaze.get_gaze_features(detected_faces_data, frame.shape)

            predicted_screen_x, predicted_screen_y = None, None
            raw_gaze_camera_coords = None # For visualization

            if model.is_trained: # Use model.predict only if model claims to be trained
                if gaze_features:
                    raw_gaze_camera_coords = gaze_features.get('raw_iris_pixels')
                    avg_pupil_norm = gaze_features.get('avg_pupil_normalized')
                    head_rvec = gaze_features.get('head_pose_rvec')
                    head_tvec = gaze_features.get('head_pose_tvec')

                    if raw_gaze_camera_coords and avg_pupil_norm and head_rvec is not None and head_tvec is not None:
                        gaze_cam_x, gaze_cam_y = raw_gaze_camera_coords
                        pupil_norm_x, pupil_norm_y = avg_pupil_norm
                        
                        predicted_screen_x, predicted_screen_y = model.predict(
                            gaze_cam_x, gaze_cam_y, 
                            pupil_norm_x, pupil_norm_y, 
                            head_rvec, head_tvec
                        )
                    elif raw_gaze_camera_coords: 
                        gaze_cam_x, gaze_cam_y = raw_gaze_camera_coords
                        # This call might trigger a warning in predict if model expects more features
                        predicted_screen_x, predicted_screen_y = model.predict(gaze_cam_x, gaze_cam_y)
                        if predicted_screen_x is not None and predicted_screen_y is not None:
                            print("INFO: Using model prediction with raw_gaze_camera_coords only.") # Added notification
            
            elif gaze_features and gaze_features.get('raw_iris_pixels'): # Model is not trained, do simple scaling in main.py
                raw_gaze_camera_coords = gaze_features.get('raw_iris_pixels')
                print("INFO: Model not trained or required features missing. Reverting to simple gaze tracking (scaling in main.py).") # Added notification
                gaze_cam_x, gaze_cam_y = raw_gaze_camera_coords
                
                if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
                    normalized_gaze_x = 0
                    normalized_gaze_y = 0
                else:
                    normalized_gaze_x = gaze_cam_x / CAMERA_WIDTH
                    normalized_gaze_y = gaze_cam_y / CAMERA_HEIGHT
                
                predicted_screen_x = int(max(0, min(normalized_gaze_x * SCREEN_WIDTH, SCREEN_WIDTH - 1)))
                predicted_screen_y = int(max(0, min(normalized_gaze_y * SCREEN_HEIGHT, SCREEN_HEIGHT - 1)))
            # If model is not trained and no gaze_features or raw_iris_pixels, predicted_screen_x/y remain None

            if predicted_screen_x is not None and predicted_screen_y is not None:
                # Apply smoothing
                current_mouse_x = previous_mouse_x * (1 - SMOOTHING_FACTOR) + predicted_screen_x * SMOOTHING_FACTOR
                current_mouse_y = previous_mouse_y * (1 - SMOOTHING_FACTOR) + predicted_screen_y * SMOOTHING_FACTOR

                # 4. Move the mouse
                pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y), duration=0) # duration=0 for fastest movement

                previous_mouse_x, previous_mouse_y = current_mouse_x, current_mouse_y


            # --- Visualization (optional, but helpful for debugging) ---
            # Draw rectangles around detected eyes
            # The structure of eye_regions_found has changed to detected_faces_data
            if detected_faces_data: # Check if any face data was returned
                # Assuming we process the first detected face for main.py visualization
                face_data = detected_faces_data[0]
                eye_bboxes_to_draw = []
                if 'left_eye' in face_data and face_data['left_eye']:
                    eye_bboxes_to_draw.append(face_data['left_eye'])
                if 'right_eye' in face_data and face_data['right_eye']:
                    eye_bboxes_to_draw.append(face_data['right_eye'])
                
                for (ex, ey, ew, eh) in eye_bboxes_to_draw:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # Green for eyes

            if raw_gaze_camera_coords: # Changed from gaze_camera_coords
                cv2.circle(frame, raw_gaze_camera_coords, 5, (255, 0, 0), -1) # Blue for raw gaze point
                cv2.putText(frame, f"Cam Gaze: ({raw_gaze_camera_coords[0]},{raw_gaze_camera_coords[1]})",
                            (raw_gaze_camera_coords[0] + 10, raw_gaze_camera_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if predicted_screen_x is not None and predicted_screen_y is not None:
                cv2.putText(frame, f"Screen Target: ({predicted_screen_x},{predicted_screen_y})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Screen Target: (No gaze)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            cv2.imshow('Real-time Gaze Mouse Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Close MediaPipe face_mesh object if it exists and was initialized by eyes.py
        if hasattr(eyes, 'face_mesh') and eyes.face_mesh is not None:
            eyes.face_mesh.close()
        print("Webcam released and windows closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Gaze Mouse Control")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining of the gaze model before starting.")
    parser.add_argument("--training_file", type=str, default=DEFAULT_TRAINING_DATA_FILE,
                        help=f"Path to the training data JSON file. Defaults to {DEFAULT_TRAINING_DATA_FILE}")
    
    args = parser.parse_args()

    # Construct the model file path based on the directory GazeToScreenModel uses
    # GazeToScreenModel uses its model_path parameter to construct its internal model_path.
    # We ensure main.py is aware of this location for clarity if needed, but model handles its own path.

    main_realtime_gaze_mouse(retrain_model=args.retrain, training_file=args.training_file)
