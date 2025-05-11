import cv2
from . import eyes # Import the refactored eyes.py module
import numpy as np # Add numpy for calculations if needed

# Define landmark indices for irises (based on MediaPipe Face Mesh)
# These are the presumed center points of the irises.
LEFT_IRIS_CENTER_LANDMARK_ID = 473 
RIGHT_IRIS_CENTER_LANDMARK_ID = 468

# Define landmark indices for eye corners (example, adjust as needed)
LEFT_EYE_OUTER_CORNER_LANDMARK_ID = 33
LEFT_EYE_INNER_CORNER_LANDMARK_ID = 133
RIGHT_EYE_OUTER_CORNER_LANDMARK_ID = 263
RIGHT_EYE_INNER_CORNER_LANDMARK_ID = 362

def get_normalized_pupil_coords(landmarks, landmark_id, frame_width, frame_height, eye_bbox=None):
    """
    Calculates normalized pupil coordinates.
    If eye_bbox is provided, coordinates are normalized relative to the eye bounding box.
    Otherwise, they are normalized relative to the full frame.
    """
    try:
        pupil_landmark = landmarks.landmark[landmark_id]
        # Allow using data even with low visibility
        # if pupil_landmark.visibility < 0.5:
        #     return None, None

        pupil_x_abs = pupil_landmark.x * frame_width
        pupil_y_abs = pupil_landmark.y * frame_height

        if eye_bbox:
            ex, ey, ew, eh = eye_bbox
            if ew == 0 or eh == 0: return None, None # Avoid division by zero
            norm_x = (pupil_x_abs - ex) / ew
            norm_y = (pupil_y_abs - ey) / eh
            return norm_x, norm_y
        else:
            norm_x = pupil_landmark.x # Already normalized to frame (0.0 - 1.0)
            norm_y = pupil_landmark.y # Already normalized to frame (0.0 - 1.0)
            return norm_x, norm_y
    except IndexError:
        return None, None

def get_gaze_features(detected_faces_data, frame_shape):
    """
    Determines gaze features including normalized pupil coordinates, raw iris pixel coordinates,
    and head pose from MediaPipe data.

    Args:
        detected_faces_data: A list of dictionaries from eyes.get_eye_regions.
                             Each dict contains 'landmarks', 'head_pose_rotation_vector',
                             'head_pose_translation_vector', and eye bounding boxes.
        frame_shape: The shape of the video frame (height, width, channels).

    Returns:
        A dictionary containing:
        - 'left_pupil_normalized': (x, y) or None
        - 'right_pupil_normalized': (x, y) or None
        - 'avg_pupil_normalized': (x, y) or None (average of available normalized pupils)
        - 'raw_iris_pixels': (x, y) pixel coordinates of average iris position or None
        - 'head_pose_rvec': rotation vector or None
        - 'head_pose_tvec': translation vector or None
        Returns None if no valid face data is processed.
    """
    if not detected_faces_data:
        return None

    face_data = detected_faces_data[0] # Use the first detected face
    landmarks = face_data.get('landmarks')
    head_pose_rvec = face_data.get('head_pose_rotation_vector')
    head_pose_tvec = face_data.get('head_pose_translation_vector')
    left_eye_bbox = face_data.get('left_eye')
    right_eye_bbox = face_data.get('right_eye')

    if not landmarks:
        return {
            'left_pupil_normalized': None,
            'right_pupil_normalized': None,
            'avg_pupil_normalized': None,
            'raw_iris_pixels': None,
            'head_pose_rvec': head_pose_rvec, # Still return pose if landmarks were somehow None but pose was estimated
            'head_pose_tvec': head_pose_tvec
        }

    frame_height, frame_width, _ = frame_shape
    
    # Get normalized pupil coordinates
    # Prefer normalization relative to eye bounding box if available
    ln_x, ln_y = get_normalized_pupil_coords(landmarks, LEFT_IRIS_CENTER_LANDMARK_ID, frame_width, frame_height, left_eye_bbox)
    rn_x, rn_y = get_normalized_pupil_coords(landmarks, RIGHT_IRIS_CENTER_LANDMARK_ID, frame_width, frame_height, right_eye_bbox)

    left_pupil_norm = (ln_x, ln_y) if ln_x is not None else None
    right_pupil_norm = (rn_x, rn_y) if rn_x is not None else None
    
    avg_pupil_norm = None
    valid_norm_pupils_x = []
    valid_norm_pupils_y = []
    if left_pupil_norm:
        valid_norm_pupils_x.append(left_pupil_norm[0])
        valid_norm_pupils_y.append(left_pupil_norm[1])
    if right_pupil_norm:
        valid_norm_pupils_x.append(right_pupil_norm[0])
        valid_norm_pupils_y.append(right_pupil_norm[1])

    if valid_norm_pupils_x:
        avg_pupil_norm = (np.mean(valid_norm_pupils_x), np.mean(valid_norm_pupils_y))

    # Calculate raw iris pixel coordinates (fallback/legacy)
    raw_iris_points_pixels = []
    try:
        left_iris_lm = landmarks.landmark[LEFT_IRIS_CENTER_LANDMARK_ID]
        lx_px = int(left_iris_lm.x * frame_width)
        ly_px = int(left_iris_lm.y * frame_height)
        raw_iris_points_pixels.append((lx_px, ly_px))
    except IndexError:
        pass

    try:
        right_iris_lm = landmarks.landmark[RIGHT_IRIS_CENTER_LANDMARK_ID]
        rx_px = int(right_iris_lm.x * frame_width)
        ry_px = int(right_iris_lm.y * frame_height)
        raw_iris_points_pixels.append((rx_px, ry_px))
    except IndexError:
        pass

    raw_iris_avg_pixels = None
    if raw_iris_points_pixels:
        raw_iris_avg_pixels = (int(np.mean([p[0] for p in raw_iris_points_pixels])),
                               int(np.mean([p[1] for p in raw_iris_points_pixels])))
    else:
        # Fallback to eye bbox centers if raw iris pixels are not good
        eye_bboxes_centers_px = []
        if left_eye_bbox:
            ex, ey, ew, eh = left_eye_bbox
            eye_bboxes_centers_px.append((ex + ew / 2, ey + eh / 2))
        if right_eye_bbox:
            ex, ey, ew, eh = right_eye_bbox
            eye_bboxes_centers_px.append((ex + ew / 2, ey + eh / 2))
        
        if eye_bboxes_centers_px:
            raw_iris_avg_pixels = (int(np.mean([p[0] for p in eye_bboxes_centers_px])),
                                   int(np.mean([p[1] for p in eye_bboxes_centers_px])))

    return {
        'left_pupil_normalized': left_pupil_norm,
        'right_pupil_normalized': right_pupil_norm,
        'avg_pupil_normalized': avg_pupil_norm,
        'raw_iris_pixels': raw_iris_avg_pixels,
        'head_pose_rvec': head_pose_rvec,
        'head_pose_tvec': head_pose_tvec
    }

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running gaze.py with MediaPipe. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Use the function from eyes.py to get eye regions and landmarks
        # eyes.py's get_eye_regions now expects an RGB frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces_data = eyes.get_eye_regions(rgb_frame)

        # Determine gaze features using the new function
        gaze_features = get_gaze_features(detected_faces_data, frame.shape)

        # Visualization
        # Draw rectangles around detected eyes (if bounding boxes are available)
        if detected_faces_data:
            face_data = detected_faces_data[0] # Assuming one face for simplicity in this main
            if 'left_eye' in face_data and face_data['left_eye']:
                (ex, ey, ew, eh) = face_data['left_eye']
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if 'right_eye' in face_data and face_data['right_eye']:
                (ex, ey, ew, eh) = face_data['right_eye']
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Optionally, draw all landmarks if needed for debugging (from eyes.py main_visualization_loop)
            # This might be too cluttered for gaze.py's main purpose.

        if gaze_features:
            raw_gaze_point_pixels = gaze_features.get('raw_iris_pixels')
            avg_pupil_norm = gaze_features.get('avg_pupil_normalized')
            head_rvec = gaze_features.get('head_pose_rvec')
            # head_tvec = gaze_features.get('head_pose_tvec') # tvec not directly visualized here yet

            if raw_gaze_point_pixels:
                cv2.circle(frame, raw_gaze_point_pixels, 7, (0, 0, 255), -1) # Red circle for raw gaze
                cv2.putText(frame, f"Raw Gaze: {raw_gaze_point_pixels}", (raw_gaze_point_pixels[0] + 15, raw_gaze_point_pixels[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            
            if avg_pupil_norm:
                 cv2.putText(frame, f"Norm Pupil XY: ({avg_pupil_norm[0]:.2f}, {avg_pupil_norm[1]:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if head_rvec is not None:
                # Simple display of head rotation vector (e.g., as text)
                # More complex visualization (like drawing axes) can be added if needed, similar to eyes.py
                rvec_text = f"Head Rvec: [{head_rvec[0][0]:.2f}, {head_rvec[1][0]:.2f}, {head_rvec[2][0]:.2f}]"
                cv2.putText(frame, rvec_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Example: Draw head pose line (if landmarks are available from detected_faces_data)
                # This reuses some logic from eyes.py for drawing the pose line for convenience
                if detected_faces_data and detected_faces_data[0].get('landmarks'):
                    face_data_for_pose_vis = detected_faces_data[0]
                    landmarks_for_pose_vis = face_data_for_pose_vis['landmarks']
                    tvec_for_pose_vis = face_data_for_pose_vis.get('head_pose_translation_vector')
                    if tvec_for_pose_vis is not None:
                        nose_end_point3D = np.array([[0, 0, 100.0]], dtype=np.float64)
                        h_frame, w_frame, _ = frame.shape
                        focal_length = w_frame
                        cam_matrix = np.array(
                            [[focal_length, 0, w_frame/2],
                                [0, focal_length, h_frame/2],
                                [0, 0, 1]], dtype="double")
                        try:
                            (nose_end_point2D, _) = cv2.projectPoints(nose_end_point3D, head_rvec, tvec_for_pose_vis, cam_matrix, np.zeros((4,1)))
                            nose_tip_landmark = landmarks_for_pose_vis.landmark[1] # Nose tip
                            p1 = (int(nose_tip_landmark.x * w_frame), int(nose_tip_landmark.y * h_frame))
                            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                            cv2.line(frame, p1, p2, (255, 0, 255), 2) # Magenta line for head direction
                        except Exception as e:
                            # print(f"Error projecting head pose line in gaze.py: {e}")
                            pass 

        cv2.imshow('Gaze Tracking (gaze.py - MediaPipe)', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Ensure MediaPipe resources are released if they were initialized in eyes.py's context
    # This is handled by eyes.py's face_mesh.close() if main_visualization_loop is run from there.
    # If gaze.py is run standalone, eyes.py's global face_mesh object might need explicit closing.
    # For now, assuming eyes.py manages its resources or is used as a library.
    if hasattr(eyes, 'face_mesh') and eyes.face_mesh is not None:
         eyes.face_mesh.close()


if __name__ == "__main__":
    main()
