import cv2
import mediapipe as mp # Import MediaPipe
import numpy as np

# Initialize MediaPipe Face Mesh globally
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define a simple 3D model of the face for solvePnP
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],             # Nose tip (landmark 1)
    [0.0, -330.0, -65.0],        # Chin (landmark 152) - Approximate
    [-225.0, 170.0, -135.0],     # Left eye left corner (landmark 33) - Approximate
    [225.0, 170.0, -135.0],      # Right eye right corner (landmark 263) - Approximate
    [-150.0, -150.0, -125.0],    # Left Mouth corner (landmark 61) - Approximate
    [150.0, -150.0, -125.0]      # Right mouth corner (landmark 291) - Approximate
], dtype=np.float64)

# Corresponding landmark indices for the 3D model points
FACE_MODEL_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
# Iris center landmark IDs from gaze.py (for visualization here)
LEFT_IRIS_CENTER_LANDMARK_ID = 473 
RIGHT_IRIS_CENTER_LANDMARK_ID = 468

def estimate_head_pose(face_landmarks, frame_shape):
    """
    Estimates head pose using cv2.solvePnP.

    Args:
        face_landmarks: MediaPipe face landmarks for a single face.
        frame_shape: The shape of the video frame (height, width, channels).

    Returns:
        A tuple (rotation_vector, translation_vector, num_points) or (None, None, num_points) if estimation fails.
    """
    h, w, _ = frame_shape
    image_points_2d = []
    model_points_3d_subset = []

    for i, landmark_id in enumerate(FACE_MODEL_LANDMARK_IDS):
        try:
            landmark = face_landmarks.landmark[landmark_id]
            # NOTE: Temporarily (or permanently, if visibility/presence scores are unreliable)
            # bypassing visibility and presence checks for head pose landmarks.
            # In some environments/setups, these scores might be consistently 0.0
            # even when landmarks are correctly detected. If head pose estimation
            # works without this check but fails with it, this is a likely cause.
            x, y = int(landmark.x * w), int(landmark.y * h)
            image_points_2d.append([x, y])
            model_points_3d_subset.append(FACE_MODEL_3D[i])
        except IndexError:
            continue
    
    if len(image_points_2d) < 4:
        return None, None, len(image_points_2d)

    image_points_2d = np.array(image_points_2d, dtype=np.float64)
    model_points_3d_subset = np.array(model_points_3d_subset, dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    try:
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points_3d_subset, image_points_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            return rotation_vector, translation_vector, len(image_points_2d)
        else:
            return None, None, len(image_points_2d)
    except Exception:
        return None, None, len(image_points_2d)

def get_eye_regions(frame_to_process):
    """
    Detects facial landmarks, focusing on eye regions and estimates head pose
    using MediaPipe Face Mesh.
    
    Args:
        frame_to_process: The input video frame (NumPy array).
        
    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'left_eye': bounding box (x, y, w, h) or None
        - 'right_eye': bounding box (x, y, w, h) or None
        - 'landmarks': MediaPipe NormalizedLandmarkList
        - 'head_pose_rotation_vector': rotation vector from solvePnP or None
        - 'head_pose_translation_vector': translation vector from solvePnP or None
        - 'head_pose_points_found': number of points used for head pose estimation
        Returns an empty list if no face is detected or in case of an error.
    """
    if frame_to_process is None:
        return []

    rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    eye_regions = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame_to_process.shape
            
            left_eye_landmarks = [33, 160, 158, 133, 153, 144] 
            right_eye_landmarks = [362, 385, 387, 263, 373, 380]

            def get_bounding_box(landmarks_indices):
                x_coords = [face_landmarks.landmark[i].x * w for i in landmarks_indices]
                y_coords = [face_landmarks.landmark[i].y * h for i in landmarks_indices]
                if not x_coords or not y_coords:
                    return None
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                return (x_min, y_min, x_max - x_min, y_max - y_min)

            left_eye_bbox = get_bounding_box(left_eye_landmarks)
            right_eye_bbox = get_bounding_box(right_eye_landmarks)
            
            rvec, tvec, num_pose_points = estimate_head_pose(face_landmarks, frame_to_process.shape)

            face_data = {'landmarks': face_landmarks, 
                         'head_pose_rotation_vector': rvec, 
                         'head_pose_translation_vector': tvec,
                         'head_pose_points_found': num_pose_points}

            if left_eye_bbox:
                face_data['left_eye'] = left_eye_bbox
            if right_eye_bbox:
                face_data['right_eye'] = right_eye_bbox
            
            eye_regions.append(face_data)

    return eye_regions

def main_visualization_loop():
    """
    Main loop for direct execution and visualization of facial landmark detection using MediaPipe.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Running eyes.py main_visualization_loop() with MediaPipe. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        head_pose_estimated_this_frame = False
        num_pose_points_found_this_frame = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                h_frame, w_frame, _ = frame.shape

                for landmark_id in FACE_MODEL_LANDMARK_IDS:
                    try:
                        landmark = face_landmarks.landmark[landmark_id]
                        if landmark.visibility > 0.5 and landmark.presence > 0.5:
                            x, y = int(landmark.x * w_frame), int(landmark.y * h_frame)
                            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                    except IndexError:
                        pass

                iris_landmarks_to_check = [LEFT_IRIS_CENTER_LANDMARK_ID, RIGHT_IRIS_CENTER_LANDMARK_ID]
                for landmark_id in iris_landmarks_to_check:
                    try:
                        landmark = face_landmarks.landmark[landmark_id]
                        if landmark.visibility > 0.5:
                            x, y = int(landmark.x * w_frame), int(landmark.y * h_frame)
                            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
                    except IndexError:
                        pass

                print("--- Landmark Visibilities & Presences ---")
                for idx, landmark_id in enumerate(FACE_MODEL_LANDMARK_IDS):
                    lm = face_landmarks.landmark[landmark_id]
                    print(f"Head Pose LM {landmark_id}: Vis={lm.visibility:.2f}, Pres={lm.presence:.2f}")

                left_iris_lm = face_landmarks.landmark[LEFT_IRIS_CENTER_LANDMARK_ID]
                right_iris_lm = face_landmarks.landmark[RIGHT_IRIS_CENTER_LANDMARK_ID]
                print(f"Left Iris LM {LEFT_IRIS_CENTER_LANDMARK_ID}: Vis={left_iris_lm.visibility:.2f}, Pres={left_iris_lm.presence:.2f}")
                print(f"Right Iris LM {RIGHT_IRIS_CENTER_LANDMARK_ID}: Vis={right_iris_lm.visibility:.2f}, Pres={right_iris_lm.presence:.2f}")
                print("------------------------------------")

                eye_data_list = get_eye_regions(frame)
                for eye_data in eye_data_list:
                    if 'left_eye' in eye_data and eye_data['left_eye']:
                        (ex, ey, ew, eh) = eye_data['left_eye']
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    if 'right_eye' in eye_data and eye_data['right_eye']:
                        (ex, ey, ew, eh) = eye_data['right_eye']
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                    num_pose_points_found_this_frame = eye_data.get('head_pose_points_found', 0)
                    if eye_data.get('head_pose_rotation_vector') is not None and \
                       eye_data.get('head_pose_translation_vector') is not None:
                        head_pose_estimated_this_frame = True
                        rvec = eye_data['head_pose_rotation_vector']
                        tvec = eye_data['head_pose_translation_vector']
                        
                        nose_end_point3D = np.array([[0, 0, 100.0]], dtype=np.float64)
                        cam_matrix = np.array(
                            [[w_frame, 0, w_frame/2],
                             [0, w_frame, h_frame/2],
                             [0, 0, 1]], dtype="double")
                        dist_coeffs_vis = np.zeros((4,1))
                        try:
                            (nose_end_point2D, _) = cv2.projectPoints(nose_end_point3D, rvec, tvec, cam_matrix, dist_coeffs_vis)
                            nose_tip_landmark = eye_data['landmarks'].landmark[1]
                            p1 = (int(nose_tip_landmark.x * w_frame), int(nose_tip_landmark.y * h_frame))
                            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                            cv2.line(frame, p1, p2, (255, 0, 255), 2)
                        except Exception:
                            pass
            
            print(f"Head pose estimated: {head_pose_estimated_this_frame}. Points found for pose: {num_pose_points_found_this_frame}")
            try:
                left_iris_lm = face_landmarks.landmark[LEFT_IRIS_CENTER_LANDMARK_ID]
                right_iris_lm = face_landmarks.landmark[RIGHT_IRIS_CENTER_LANDMARK_ID]
                print(f"Iris L vis: {left_iris_lm.visibility:.2f}, Iris R vis: {right_iris_lm.visibility:.2f}")
            except Exception:
                print("Iris landmarks not found for visibility check.")

        cv2.imshow('MediaPipe Face Mesh (eyes.py - Debug)', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main_visualization_loop()
