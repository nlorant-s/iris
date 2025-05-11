import cv2
import numpy as np
import time
import json
import sys # Add sys import
import os # Add os import

# Add parent directory to sys.path to find the vision module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from vision import eyes, gaze
from datetime import datetime

# --- Configuration ---
SCREEN_WIDTH = 1920  # Replace with actual screen width if possible, or use a fixed calibration screen size
SCREEN_HEIGHT = 1080 # Replace with actual screen height if possible
CAMERA_WIDTH = 640   # Width of the camera frame
CAMERA_HEIGHT = 480  # Height of the camera frame

CALIBRATION_POINTS_PERCENTAGES = [
    # Original 9 points
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    # Extra corners (original)
    (0.05, 0.05), (0.95, 0.05),
    (0.05, 0.95), (0.95, 0.95),
    # Edges (original)
    (0.5, 0.05), (0.05, 0.5), (0.95, 0.5), (0.5, 0.95),
    # Additional intermediate points for more density
    (0.3, 0.3), (0.7, 0.3),
    (0.3, 0.7), (0.7, 0.7),
    (0.2, 0.7), (0.8, 0.7), # More horizontal coverage at different y
    (0.7, 0.2), (0.7, 0.8), # More vertical coverage at different x
]

# Convert percentage points to pixel coordinates
CALIBRATION_POINTS = [(int(p[0] * SCREEN_WIDTH), int(p[1] * SCREEN_HEIGHT)) for p in CALIBRATION_POINTS_PERCENTAGES]

# Define the target training data file
TARGET_TRAINING_FILE = os.path.normpath(os.path.join(current_dir, "..", "data", "training", "training-data-51125.json"))
# OUTPUT_FILE = f"training-data-{datetime.now().strftime('%H%M')}.json" # Commented out, using fixed target file

POINT_DISPLAY_TIME_SEC = 0.2  # Time to display each point before capture instruction
CAPTURE_DURATION_SEC = 5.0    # Increased duration to capture gaze data with head movement
SAMPLES_PER_POINT = 100     # Increased samples to try and collect per point

def display_calibration_point(screen_image, point_x, point_y):
    """Displays a single calibration point on the screen."""
    radius = 20
    thickness = -1 # Filled circle
    color = (0, 0, 255)  # Red
    cv2.circle(screen_image, (point_x, point_y), radius, color, thickness)
    cv2.circle(screen_image, (point_x, point_y), radius + 5, (255,255,255), 2) # White outline

def main_calibration_procedure():
    """Runs the main calibration loop."""
    calibration_data = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Create a named window for the calibration display
    calibration_window_name = "Calibration Screen - Press SPACE to capture, Q to quit"
    cv2.namedWindow(calibration_window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(calibration_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("--- Optimal Calibration Conditions ---")
    print("- Ensure good, consistent lighting on your face. Avoid strong backlighting or shadows.")
    print("- Keep your face clearly visible to the camera, without obstructions (e.g., hair over eyes, reflective glasses if possible).")
    print("- Ensure the camera is stable and does not move during the calibration process.")
    print("- Try to maintain a relatively consistent distance from the camera where your face is well-detected.")
    print("------------------------------------")
    print(f"Starting calibration process. Look at the red dots as they appear.")
    print(f"Screen resolution for calibration: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"For each point, focus on the dot. When 'CAPTURING...' appears,")
    print(f"continue looking at the dot while naturally moving your head (e.g., slight tilts, nods, turns).")
    print(f"Press 'SPACE' when ready to start capturing for a point.")
    print(f"Press 'Q' to quit the calibration early.")

    for i, (target_x, target_y) in enumerate(CALIBRATION_POINTS):
        print(f"\nDisplaying point {i+1}/{len(CALIBRATION_POINTS)}: ({target_x}, {target_y})")
        
        # Create a black screen for calibration
        calibration_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        display_calibration_point(calibration_screen, target_x, target_y)
        cv2.imshow(calibration_window_name, calibration_screen)
        
        # Give user time to look at the point
        start_display_time = time.time()
        while time.time() - start_display_time < POINT_DISPLAY_TIME_SEC:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
            if cv2.getWindowProperty(calibration_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration window closed unexpectedly.")
                cap.release()
                return

        # Instruction to capture (now combined with waiting for space)
        capture_instruction_screen = calibration_screen.copy()
        radius = 20 # Keep radius consistent
        cv2.imshow(calibration_window_name, capture_instruction_screen)

        # Wait for user to press space
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord(' '): # Spacebar
                print("Starting data capture for this point. Move your head naturally while looking at the dot.")
                break
            if cv2.getWindowProperty(calibration_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration window closed unexpectedly.")
                cap.release()
                return

        # Capture gaze data for this point
        capturing_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Fresh black screen
        display_calibration_point(capturing_screen, target_x, target_y) # Show the point
        # Change dot to green to indicate capture is active
        cv2.circle(capturing_screen, (target_x, target_y), 20, (0, 255, 0), -1) 
        cv2.imshow(calibration_window_name, capturing_screen)
        cv2.waitKey(1) # Ensure it displays

        start_capture_time = time.time()
        samples_collected_for_this_point = 0 # Renamed and used for SAMPLES_PER_POINT limit for current point
        no_face_consecutive_frames = 0  # Counter for consecutive frames with no face detected
        MAX_CONSECUTIVE_NO_FACE_FRAMES = 45 # Approx 1.5 seconds if FPS is around 30

        while time.time() - start_capture_time < CAPTURE_DURATION_SEC and samples_collected_for_this_point < SAMPLES_PER_POINT:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Create a fresh screen for this frame to draw warnings
            current_frame_display = capturing_screen.copy()
            warning_messages = []
            
            # Pass BGR frame directly to eyes.get_eye_regions
            detected_faces_data = eyes.get_eye_regions(frame.copy())
            
            all_eye_bboxes = []
            if not detected_faces_data:
                warning_messages.append("WARN: No face detected")
                no_face_consecutive_frames += 1
            else:
                no_face_consecutive_frames = 0 # Reset counter if face is detected
                # Assuming we're interested in the first detected face for calibration
                face_data = detected_faces_data[0] 
                if 'left_eye' in face_data and face_data['left_eye']:
                    all_eye_bboxes.append(face_data['left_eye'])
                if 'right_eye' in face_data and face_data['right_eye']:
                    all_eye_bboxes.append(face_data['right_eye'])
                if not all_eye_bboxes:
                    warning_messages.append("WARN: No eyes detected on face")


            # Get all gaze features
            gaze_features = gaze.get_gaze_features(detected_faces_data, frame.shape)

            # Visualization on the camera feed window
            for (ex, ey, ew, eh) in all_eye_bboxes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            raw_iris_pixels = None
            avg_pupil_norm = None
            head_rvec = None
            head_tvec = None

            if not gaze_features:
                warning_messages.append("WARN: Gaze features not found")
            else:
                raw_iris_pixels = gaze_features.get('raw_iris_pixels')
                avg_pupil_norm = gaze_features.get('avg_pupil_normalized') # This is instantaneous pupil
                head_rvec = gaze_features.get('head_pose_rvec')
                head_tvec = gaze_features.get('head_pose_tvec')

                all_data_valid_for_sample = True
                if raw_iris_pixels is None:
                    warning_messages.append("WARN: Iris pixels not detected for this sample")
                    all_data_valid_for_sample = False
                else:
                    cv2.circle(frame, raw_iris_pixels, 5, (0, 0, 255), -1) # Visualize

                if avg_pupil_norm is None:
                    warning_messages.append("WARN: Pupil norm not detected for this sample")
                    all_data_valid_for_sample = False
                
                if head_rvec is None:
                    warning_messages.append("WARN: Head rotation (rvec) not detected for this sample")
                    all_data_valid_for_sample = False
                
                if head_tvec is None:
                    warning_messages.append("WARN: Head translation (tvec) not detected for this sample")
                    all_data_valid_for_sample = False

                if all_data_valid_for_sample:
                    calibration_data.append({
                        "target_screen_px": (target_x, target_y),
                        "raw_gaze_camera_px": (float(raw_iris_pixels[0]), float(raw_iris_pixels[1])),
                        "normalized_pupil_coord_xy": (float(avg_pupil_norm[0]), float(avg_pupil_norm[1])),
                        "head_pose_rvec": head_rvec.flatten().tolist(),
                        "head_pose_tvec": head_tvec.flatten().tolist()
                    })
                    samples_collected_for_this_point += 1
            
            # Display warning messages on the calibration screen
            # Add prominent warning if no face is detected for too long
            if no_face_consecutive_frames > MAX_CONSECUTIVE_NO_FACE_FRAMES:
                pass # Keep the warning logic but remove the text

            for idx, msg in enumerate(warning_messages):
                pass # Keep the warning logic but remove the text

            cv2.imshow(calibration_window_name, current_frame_display) # Show warnings
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow quitting during capture phase too
                print("Calibration aborted during capture.")
                cap.release()
                cv2.destroyAllWindows()
                return
            if cv2.getWindowProperty(calibration_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration window closed unexpectedly.")
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"Collected {samples_collected_for_this_point} valid data entries for point ({target_x}, {target_y}).")
        
        # Brief pause or message before next point
        inter_point_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # Fresh black screen
        display_calibration_point(inter_point_screen, target_x, target_y) # Show the point again briefly (red)
        cv2.imshow(calibration_window_name, inter_point_screen)
        cv2.waitKey(1000) # Wait 1 second

    # End of calibration
    cap.release()
    cv2.destroyAllWindows()

    if calibration_data:
        existing_data = []
        if os.path.exists(TARGET_TRAINING_FILE):
            try:
                with open(TARGET_TRAINING_FILE, 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print(f"Warning: Existing data in {TARGET_TRAINING_FILE} is not a list. Overwriting with new data.")
                    existing_data = []
                print(f"Loaded {len(existing_data)} existing samples from {TARGET_TRAINING_FILE}.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {TARGET_TRAINING_FILE}. File will be overwritten.")
                existing_data = []
            except Exception as e:
                print(f"Error loading existing data from {TARGET_TRAINING_FILE}: {e}. File will be overwritten.")
                existing_data = []
        
        combined_data = existing_data + calibration_data
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(TARGET_TRAINING_FILE), exist_ok=True)
            with open(TARGET_TRAINING_FILE, 'w') as f:
                json.dump(combined_data, f, indent=4)
            print(f"\nCalibration complete. Data appended and saved to {TARGET_TRAINING_FILE}")
            print(f"Total individual data samples collected in this session: {len(calibration_data)}")
            print(f"Total samples in file now: {len(combined_data)}")
            print("This data can now be used to train or refine the GazeToScreenModel.")
        except Exception as e:
            print(f"Error saving combined data to {TARGET_TRAINING_FILE}: {e}")
    else:
        print("\nCalibration finished, but no data was collected.")

if __name__ == "__main__":
    main_calibration_procedure()
