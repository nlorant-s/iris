import cv2
import numpy as np
import time
import json
import eyes  # Assumes eyes.py is in the same directory or Python path
import gaze  # Assumes gaze.py is in the same directory or Python path

# --- Configuration ---
SCREEN_WIDTH = 1920  # Replace with actual screen width if possible, or use a fixed calibration screen size
SCREEN_HEIGHT = 1080 # Replace with actual screen height if possible
CAMERA_WIDTH = 640   # Width of the camera frame
CAMERA_HEIGHT = 480  # Height of the camera frame

CALIBRATION_POINTS_PERCENTAGES = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    (0.05, 0.05), (0.95, 0.05), # Extra corners
    (0.05, 0.95), (0.95, 0.95),
    (0.5, 0.05), (0.05, 0.5), (0.95, 0.5), (0.5, 0.95) # Edges
]

# Convert percentage points to pixel coordinates
CALIBRATION_POINTS = [(int(p[0] * SCREEN_WIDTH), int(p[1] * SCREEN_HEIGHT)) for p in CALIBRATION_POINTS_PERCENTAGES]

OUTPUT_FILE = "calibration_data.json"
POINT_DISPLAY_TIME_SEC = 0  # Time to display each point before capture instruction
CAPTURE_DURATION_SEC = 2.5    # Duration to capture gaze data for each point
SAMPLES_PER_POINT = 40      # Number of gaze samples to try and collect per point

def display_calibration_point(screen_image, point_x, point_y):
    """Displays a single calibration point on the screen."""
    radius = 20
    thickness = -1 # Filled circle
    color = (0, 0, 255)  # Red
    cv2.circle(screen_image, (point_x, point_y), radius, color, thickness)
    cv2.circle(screen_image, (point_x, point_y), radius + 5, (255,255,255), 2) # White outline
    cv2.putText(screen_image, f"Look Here: ({point_x}, {point_y})", (point_x + 30, point_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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

    print(f"Starting calibration process. Look at the red dots as they appear.")
    print(f"Screen resolution for calibration: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Press 'SPACE' when focused on a point to record data.")
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
            # Keep the window responsive
            if cv2.getWindowProperty(calibration_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration window closed unexpectedly.")
                cap.release()
                return


        # Instruction to capture
        capture_instruction_screen = calibration_screen.copy()
        cv2.putText(capture_instruction_screen, "Press SPACE to CAPTURE", (50, SCREEN_HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
                print("Capturing data...")
                break
            if cv2.getWindowProperty(calibration_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration window closed unexpectedly.")
                cap.release()
                return

        # Capture gaze data for this point
        point_gaze_data_x = []
        point_gaze_data_y = []
        point_pupil_norm_x = []
        point_pupil_norm_y = []
        point_head_rvecs = []
        point_head_tvecs = []
        
        capturing_screen = calibration_screen.copy()
        cv2.putText(capturing_screen, "CAPTURING...", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(calibration_window_name, capturing_screen)
        cv2.waitKey(1) # Ensure it displays

        start_capture_time = time.time()
        samples_collected = 0
        
        # Create a small window to show camera feed during capture
        camera_feed_window_name = "Camera Feed (Eye Detection)"
        cv2.namedWindow(camera_feed_window_name, cv2.WINDOW_AUTOSIZE)

        while time.time() - start_capture_time < CAPTURE_DURATION_SEC and samples_collected < SAMPLES_PER_POINT:
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
            else:
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
                avg_pupil_norm = gaze_features.get('avg_pupil_normalized')
                head_rvec = gaze_features.get('head_pose_rvec')
                head_tvec = gaze_features.get('head_pose_tvec')

                if raw_iris_pixels is None:
                    warning_messages.append("WARN: Iris pixels not detected")
                else:
                    cv2.circle(frame, raw_iris_pixels, 5, (0, 0, 255), -1)
                    point_gaze_data_x.append(raw_iris_pixels[0])
                    point_gaze_data_y.append(raw_iris_pixels[1])
                    samples_collected += 1 # Count sample only if raw gaze is good

                if avg_pupil_norm is None:
                    warning_messages.append("WARN: Pupil norm not detected")
                else:
                    point_pupil_norm_x.append(avg_pupil_norm[0])
                    point_pupil_norm_y.append(avg_pupil_norm[1])
                
                if head_rvec is None:
                    warning_messages.append("WARN: Head rotation (rvec) not detected")
                else:
                    point_head_rvecs.append(head_rvec.flatten().tolist())
                
                if head_tvec is None:
                    warning_messages.append("WARN: Head translation (tvec) not detected")
                else:
                    point_head_tvecs.append(head_tvec.flatten().tolist())

            # Display warning messages on the calibration screen
            for idx, msg in enumerate(warning_messages):
                cv2.putText(current_frame_display, msg, (10, 30 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text

            cv2.imshow(calibration_window_name, current_frame_display) # Show warnings
            cv2.imshow(camera_feed_window_name, frame)
            
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


        cv2.destroyWindow(camera_feed_window_name) # Close camera feed window after capture for this point

        avg_gaze_x, avg_gaze_y = None, None
        num_raw_gaze_samples = 0
        if point_gaze_data_x and point_gaze_data_y:
            if len(point_gaze_data_x) > 0: # Ensure not empty before mean
                avg_gaze_x = np.mean(point_gaze_data_x)
                avg_gaze_y = np.mean(point_gaze_data_y)
                num_raw_gaze_samples = len(point_gaze_data_x)
                print(f"Collected {num_raw_gaze_samples} raw gaze samples. Avg Raw Gaze: ({avg_gaze_x:.2f}, {avg_gaze_y:.2f})")
            else:
                print("Empty raw gaze data lists collected for this point.")
        else:
            print("No raw gaze data lists (point_gaze_data_x or point_gaze_data_y) found for this point.")

        avg_pupil_norm_coord_xy = None
        if point_pupil_norm_x and point_pupil_norm_y:
            if len(point_pupil_norm_x) > 0 and len(point_pupil_norm_x) == len(point_pupil_norm_y):
                avg_pupil_norm_coord_xy = (np.mean(point_pupil_norm_x), np.mean(point_pupil_norm_y))
                print(f"Avg Normalized Pupil: ({avg_pupil_norm_coord_xy[0]:.2f}, {avg_pupil_norm_coord_xy[1]:.2f})")
            elif len(point_pupil_norm_x) == 0:
                print("Empty normalized pupil data lists collected for this point.")
            else: # Mismatch length or one list empty
                print(f"Warning: Mismatch or empty lists for normalized pupil data. X_samples: {len(point_pupil_norm_x)}, Y_samples: {len(point_pupil_norm_y)}.")
        else:
            print("No normalized pupil data lists (point_pupil_norm_x or point_pupil_norm_y) found for this point.")

        avg_rvec_list = None
        if point_head_rvecs:
            if len(point_head_rvecs) > 0:
                avg_rvec_list = np.mean(np.array(point_head_rvecs), axis=0).tolist()
                print(f"Avg Head Rvec: {avg_rvec_list}")
            else:
                print("Empty head rotation vector data list collected for this point.")
        else:
            print("No head rotation vector data list (point_head_rvecs) found for this point.")

        avg_tvec_list = None
        if point_head_tvecs:
            if len(point_head_tvecs) > 0:
                avg_tvec_list = np.mean(np.array(point_head_tvecs), axis=0).tolist()
                print(f"Avg Head Tvec: {avg_tvec_list}")
            else:
                print("Empty head translation vector data list collected for this point.")
        else:
            print("No head translation vector data list (point_head_tvecs) found for this point.")

        calibration_data.append({
            "target_screen_px": (target_x, target_y),
            "raw_gaze_camera_px": (float(avg_gaze_x), float(avg_gaze_y)) if avg_gaze_x is not None else (None, None),
            "avg_normalized_pupil_coord_xy": avg_pupil_norm_coord_xy,
            "avg_head_pose_rvec": avg_rvec_list,
            "avg_head_pose_tvec": avg_tvec_list,
            "samples": num_raw_gaze_samples # Preserves the original meaning of 'samples'
        })
        
        # Brief pause or message before next point
        inter_point_screen = calibration_screen.copy() # Show the point again briefly
        cv2.putText(inter_point_screen, "Next point soon...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(calibration_window_name, inter_point_screen)
        cv2.waitKey(1000) # Wait 1 second

    # End of calibration
    cap.release()
    cv2.destroyAllWindows()

    if calibration_data:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print(f"\nCalibration complete. Data saved to {OUTPUT_FILE}")
        print(f"Total data points collected: {len(calibration_data)}")
        print("This data can now be used to train or refine the GazeToScreenModel.")
    else:
        print("\nCalibration finished, but no data was collected.")

if __name__ == "__main__":
    main_calibration_procedure()
