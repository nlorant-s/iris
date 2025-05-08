import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
import os
import json

class GazeToScreenModel:
    MODEL_FILENAME = "gaze_model.joblib"

    def __init__(self, screen_width, screen_height, camera_width, camera_height, model_dir="."):
        """
        Initializes the gaze to screen model.
        Tries to load a pre-trained model. If not found, initializes untrained models.

        Args:
            screen_width (int): The width of the screen in pixels.
            screen_height (int): The height of the screen in pixels.
            camera_width (int): The width of the camera frame in pixels.
            camera_height (int): The height of the camera frame in pixels.
            model_dir (str): Directory where the model file is stored.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.poly_features_x = PolynomialFeatures(degree=2, include_bias=False)
        self.model_x = LinearRegression()
        self.poly_features_y = PolynomialFeatures(degree=2, include_bias=False)
        self.model_y = LinearRegression()
        
        self.is_trained = False
        self.model_path = os.path.join(model_dir, self.MODEL_FILENAME)
        self._expected_feature_count = -1 

        print(f"GazeToScreenModel initialized for screen: {screen_width}x{screen_height}, camera: {camera_width}x{camera_height}")
        if self.load_model(self.model_path):
            print(f"Loaded pre-trained model from {self.model_path}")
        else:
            print(f"No pre-trained model found at {self.model_path}. Model needs training.")

    def train(self, calibration_data_file):
        """
        Trains the model using data from calibration_data.json.
        Removes outliers from each calibration location before training.

        Args:
            calibration_data_file (str): Path to the JSON file containing calibration data.
        """
        try:
            with open(calibration_data_file, 'r') as f:
                calibration_data_input = json.load(f)
        except FileNotFoundError:
            print(f"Error: Calibration data file not found: {calibration_data_file}")
            self.is_trained = False
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {calibration_data_file}")
            self.is_trained = False
            return False

        # Group data by target_screen_px for local outlier removal
        grouped_data = {}
        for i, entry in enumerate(calibration_data_input):
            # Add original index to keep track for detailed logging if needed
            entry['_original_index'] = i
            target_px_tuple = tuple(entry.get("target_screen_px", [None,None])) # Use tuple for dict key
            if None in target_px_tuple:
                print(f"Warning: Entry {i} missing target_screen_px, cannot group for local outlier removal. Skipping this entry for now.")
                continue
            if target_px_tuple not in grouped_data:
                grouped_data[target_px_tuple] = []
            grouped_data[target_px_tuple].append(entry)

        filtered_calibration_data = []
        for target_px, entries_at_target in grouped_data.items():
            if len(entries_at_target) < 3: # Not enough points to reliably detect outliers
                filtered_calibration_data.extend(entries_at_target)
                print(f"Target {target_px}: Kept {len(entries_at_target)} points (too few to filter).")
                continue

            gaze_points_at_target = []
            valid_entries_for_target = []
            for entry in entries_at_target:
                raw_gaze = entry.get("raw_gaze_camera_px")
                if isinstance(raw_gaze, list) and len(raw_gaze) == 2 and all(isinstance(v, (int, float)) for v in raw_gaze) and not any(np.isnan(v) or np.isinf(v) for v in raw_gaze):
                    gaze_points_at_target.append(np.array(raw_gaze))
                    valid_entries_for_target.append(entry)
                else:
                    print(f"Warning: Invalid raw_gaze_camera_px for entry at target {target_px}, original_index {entry.get('_original_index')}. Skipping for local outlier check.")

            if len(valid_entries_for_target) < 3: # Re-check after validating raw_gaze_camera_px
                filtered_calibration_data.extend(valid_entries_for_target)
                print(f"Target {target_px}: Kept {len(valid_entries_for_target)} valid points (too few to filter after raw_gaze validation).")
                continue
            
            gaze_points_at_target_np = np.array(gaze_points_at_target)
            centroid = np.mean(gaze_points_at_target_np, axis=0)
            distances = np.linalg.norm(gaze_points_at_target_np - centroid, axis=1)
            
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # If std_distance is very small (e.g., all points are very close), avoid aggressive filtering
            # This threshold can be adjusted. A std_dev of 0.1 pixels means points are extremely close.
            MIN_STD_DEV_FOR_FILTERING = 0.1 
            z_score_threshold = 2.5 # Adjusted from 2.0 to 2.5
            
            kept_entries_for_target = []
            removed_count = 0
            if std_distance < MIN_STD_DEV_FOR_FILTERING:
                kept_entries_for_target.extend(valid_entries_for_target)
                print(f"Target {target_px}: Kept all {len(valid_entries_for_target)} points (std dev of distances {std_distance:.4f} < {MIN_STD_DEV_FOR_FILTERING}).")
            else:
                for i, entry in enumerate(valid_entries_for_target):
                    z_score = (distances[i] - mean_distance) / std_distance if std_distance > 1e-9 else 0 # Avoid division by zero
                    if abs(z_score) <= z_score_threshold:
                        kept_entries_for_target.append(entry)
                    else:
                        removed_count +=1
                        print(f"Target {target_px}: Removed outlier entry (original_index {entry.get('_original_index')}) with raw_gaze {entry.get('raw_gaze_camera_px')}, z-score {z_score:.2f}.")
                print(f"Target {target_px}: Kept {len(kept_entries_for_target)} points, removed {removed_count} outliers.")

            filtered_calibration_data.extend(kept_entries_for_target)
        
        # The rest of the training proceeds with 'filtered_calibration_data'
        calibration_data = filtered_calibration_data 
        if not calibration_data:
            print("Error: No valid data remaining after local outlier filtering.")
            self.is_trained = False
            return False

        raw_gaze_coords = []
        target_screen_coords = []
        pupil_coords_list = []
        head_rvecs_list = []
        head_tvecs_list = []

        valid_entries = 0

        # Helper to check list of numbers for structure, type, and NaN/Inf
        def _validate_list_of_numbers(data, expected_len, name_for_error):
            if not (isinstance(data, list) and len(data) == expected_len and
                    all(isinstance(v, (int, float)) for v in data)):
                return False, None, f"Invalid structure/type for {name_for_error}: {data}"
            # Check for NaN or Inf after ensuring they are numbers
            if any(np.isnan(v) or np.isinf(v) for v in data):
                return False, None, f"Invalid numeric value (NaN/Inf) in {name_for_error}: {data}"
            return True, data, None

        for entry_idx, entry in enumerate(calibration_data):
            is_entry_valid = True
            error_messages = []
            
            # Validate all required fields
            valid_struct, raw_gaze_val, err_msg = _validate_list_of_numbers(entry.get("raw_gaze_camera_px"), 2, "raw_gaze_camera_px")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            pupil_coord_data = entry.get("normalized_pupil_coord_xy")
            if pupil_coord_data is None: # Try old key if new one isn't found
                pupil_coord_data = entry.get("avg_normalized_pupil_coord_xy")
            valid_struct, pupil_coord_val, err_msg = _validate_list_of_numbers(pupil_coord_data, 2, "normalized_pupil_coord_xy or avg_normalized_pupil_coord_xy")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            rvec_data = entry.get("head_pose_rvec")
            if rvec_data is None: # Try old key
                rvec_data = entry.get("avg_head_pose_rvec")
            valid_struct, rvec_val, err_msg = _validate_list_of_numbers(rvec_data, 3, "head_pose_rvec or avg_head_pose_rvec")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            tvec_data = entry.get("head_pose_tvec")
            if tvec_data is None: # Try old key
                tvec_data = entry.get("avg_head_pose_tvec")
            valid_struct, tvec_val, err_msg = _validate_list_of_numbers(tvec_data, 3, "head_pose_tvec or avg_head_pose_tvec")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)
            
            valid_struct, target_px_val, err_msg = _validate_list_of_numbers(entry.get("target_screen_px"), 2, "target_screen_px")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            if is_entry_valid:
                raw_gaze_coords.append(raw_gaze_val)
                pupil_coords_list.append(pupil_coord_val)
                head_rvecs_list.append(rvec_val)
                head_tvecs_list.append(tvec_val)
                target_screen_coords.append(target_px_val)
                valid_entries += 1
            else:
                print(f"Skipping entry #{entry_idx} due to: {'; '.join(error_messages)}. Entry data: {entry}")

        # For PolynomialFeatures(degree=2, include_bias=False) with 10 input features,
        # the number of output polynomial features is 65.
        n_poly_features = 65 
        min_samples_needed = n_poly_features + 1 # For LinearRegression

        if valid_entries < min_samples_needed:
            print(f"Error: Not enough valid data points ({valid_entries}) after filtering. "
                  f"Need at least {min_samples_needed} for the current model configuration (10 base features, degree 2 polynomial).")
            self.is_trained = False
            return False

        gaze_points = np.array(raw_gaze_coords, dtype=float)
        screen_points = np.array(target_screen_coords, dtype=float)
        pupil_points = np.array(pupil_coords_list, dtype=float)
        rvec_points = np.array(head_rvecs_list, dtype=float)
        tvec_points = np.array(head_tvecs_list, dtype=float)

        # Final check for NaNs/Infs in the numpy arrays
        feature_arrays_map = {
            "gaze_points": gaze_points, "pupil_points": pupil_points,
            "rvec_points": rvec_points, "tvec_points": tvec_points
        }
        for name, arr in feature_arrays_map.items():
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Error: NaN or Inf detected in '{name}' array after processing. Training aborted.")
                self.is_trained = False
                return False
        
        all_features = np.concatenate((gaze_points, pupil_points, rvec_points, tvec_points), axis=1)
        self._expected_feature_count = all_features.shape[1]

        gaze_X_features = all_features
        screen_X_target = screen_points[:, 0]
        
        gaze_Y_features = all_features
        screen_Y_target = screen_points[:, 1]

        try:
            gaze_X_poly = self.poly_features_x.fit_transform(gaze_X_features)
            self.model_x.fit(gaze_X_poly, screen_X_target)

            gaze_Y_poly = self.poly_features_y.fit_transform(gaze_Y_features)
            self.model_y.fit(gaze_Y_poly, screen_Y_target)
            
            self.is_trained = True
            print(f"Model training complete. Expected feature count for prediction: {self._expected_feature_count}")
            self.save_model(self.model_path)
            return True
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            self.is_trained = False
            return False

    def predict(self, gaze_x, gaze_y, pupil_norm_x=None, pupil_norm_y=None, rvec=None, tvec=None):
        """
        Predicts the screen coordinates based on the gaze position, normalized pupil coordinates, and head pose.

        Args:
            gaze_x (float): The x-coordinate of the raw gaze point (from camera space).
            gaze_y (float): The y-coordinate of the raw gaze point (from camera space).
            pupil_norm_x (float, optional): Normalized x-coordinate of the pupil.
            pupil_norm_y (float, optional): Normalized y-coordinate of the pupil.
            rvec (list/np.array, optional): Head pose rotation vector (3 elements).
            tvec (list/np.array, optional): Head pose translation vector (3 elements).

        Returns:
            tuple: (screen_x, screen_y) predicted screen coordinates.
                   Returns (None, None) if input is invalid or model is not trained.
        """
        use_full_model = (self.is_trained and 
                          pupil_norm_x is not None and pupil_norm_y is not None and
                          rvec is not None and tvec is not None and
                          self._expected_feature_count == 10)

        if not use_full_model:
            warning_msg = "Warning: Model not trained, or missing new features (pupil, head pose), "
            warning_msg += "or model was trained with a different feature set. "
            warning_msg += "Falling back to simple scaling of raw gaze."
            print(warning_msg)
            if gaze_x is None or gaze_y is None:
                return (None, None)
            if self.camera_width <= 0 or self.camera_height <= 0:
                normalized_gaze_x = 0
                normalized_gaze_y = 0
            else:
                normalized_gaze_x = gaze_x / self.camera_width
                normalized_gaze_y = gaze_y / self.camera_height
            
            screen_x = normalized_gaze_x * self.screen_width
            screen_y = normalized_gaze_y * self.screen_height
            screen_x = int(max(0, min(screen_x, self.screen_width - 1)))
            screen_y = int(max(0, min(screen_y, self.screen_height - 1)))
            return (screen_x, screen_y)

        try:
            rvec_flat = np.array(rvec).flatten()
            tvec_flat = np.array(tvec).flatten()
            if rvec_flat.shape[0] != 3 or tvec_flat.shape[0] != 3:
                raise ValueError("rvec and tvec must have 3 elements each.")

            current_features = np.array([[gaze_x, gaze_y, 
                                          pupil_norm_x, pupil_norm_y, 
                                          rvec_flat[0], rvec_flat[1], rvec_flat[2],
                                          tvec_flat[0], tvec_flat[1], tvec_flat[2]]])
            
            if current_features.shape[1] != self._expected_feature_count:
                print(f"Error: Prediction feature count mismatch. Expected {self._expected_feature_count}, got {current_features.shape[1]}. Fallback.")
                if gaze_x is None or gaze_y is None: return (None, None)
                normalized_gaze_x = gaze_x / self.camera_width if self.camera_width > 0 else 0
                normalized_gaze_y = gaze_y / self.camera_height if self.camera_height > 0 else 0
                screen_x = normalized_gaze_x * self.screen_width
                screen_y = normalized_gaze_y * self.screen_height
                screen_x = int(max(0, min(screen_x, self.screen_width - 1)))
                screen_y = int(max(0, min(screen_y, self.screen_height - 1)))
                return (screen_x, screen_y)

            features_poly_x = self.poly_features_x.transform(current_features)
            screen_x = self.model_x.predict(features_poly_x)[0]
            
            features_poly_y = self.poly_features_y.transform(current_features)
            screen_y = self.model_y.predict(features_poly_y)[0]

        except Exception as e:
            print(f"Error during prediction with full features: {e}. Falling back to simple scaling.")
            if gaze_x is None or gaze_y is None: return (None, None)
            normalized_gaze_x = gaze_x / self.camera_width if self.camera_width > 0 else 0
            normalized_gaze_y = gaze_y / self.camera_height if self.camera_height > 0 else 0
            screen_x = normalized_gaze_x * self.screen_width
            screen_y = normalized_gaze_y * self.screen_height

        screen_x = int(max(0, min(screen_x, self.screen_width - 1)))
        screen_y = int(max(0, min(screen_y, self.screen_height - 1)))
        
        return (screen_x, screen_y)

    def save_model(self, filepath):
        """Saves the trained model to a file."""
        if not self.is_trained:
            print("Model is not trained. Nothing to save.")
            return
        try:
            model_data = {
                'poly_features_x_params': self.poly_features_x.get_params(),
                'model_x_coef': self.model_x.coef_,
                'model_x_intercept': self.model_x.intercept_,
                'poly_features_y_params': self.poly_features_y.get_params(),
                'model_y_coef': self.model_y.coef_,
                'model_y_intercept': self.model_y.intercept_,
                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'camera_width': self.camera_width,
                'camera_height': self.camera_height,
                'is_trained': self.is_trained
            }
            if hasattr(self.poly_features_x, 'n_features_in_'):
                 model_data['poly_features_x_n_features_in'] = self.poly_features_x.n_features_in_
            if hasattr(self.poly_features_y, 'n_features_in_'):
                 model_data['poly_features_y_n_features_in'] = self.poly_features_y.n_features_in_
            if hasattr(self, '_expected_feature_count'):
                model_data['_expected_feature_count'] = self._expected_feature_count

            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        """Loads a trained model from a file."""
        try:
            if not os.path.exists(filepath):
                return False
                
            model_data = joblib.load(filepath)
            
            self.poly_features_x = PolynomialFeatures(**model_data['poly_features_x_params'])
            if 'poly_features_x_n_features_in' in model_data and model_data['poly_features_x_n_features_in'] is not None:
                self.poly_features_x.n_features_in_ = model_data['poly_features_x_n_features_in']
                dummy_input_x = np.zeros((1, self.poly_features_x.n_features_in_))
                self.poly_features_x.fit(dummy_input_x)
            elif '_expected_feature_count' in model_data and model_data['_expected_feature_count'] > 0:
                 print("Warning: 'poly_features_x_n_features_in' not in model, using '_expected_feature_count'.")
                 self.poly_features_x.n_features_in_ = model_data['_expected_feature_count']
                 dummy_input_x = np.zeros((1, self.poly_features_x.n_features_in_))
                 self.poly_features_x.fit(dummy_input_x)
            else:
                print("Warning: 'poly_features_x_n_features_in' not found in loaded model. Model may not transform features correctly.")

            self.model_x = LinearRegression()
            self.model_x.coef_ = model_data['model_x_coef']
            self.model_x.intercept_ = model_data['model_x_intercept']

            self.poly_features_y = PolynomialFeatures(**model_data['poly_features_y_params'])
            if 'poly_features_y_n_features_in' in model_data and model_data['poly_features_y_n_features_in'] is not None:
                self.poly_features_y.n_features_in_ = model_data['poly_features_y_n_features_in']
                dummy_input_y = np.zeros((1, self.poly_features_y.n_features_in_))
                self.poly_features_y.fit(dummy_input_y)
            elif '_expected_feature_count' in model_data and model_data['_expected_feature_count'] > 0:
                 print("Warning: 'poly_features_y_n_features_in' not in model, using '_expected_feature_count'.")
                 self.poly_features_y.n_features_in_ = model_data['_expected_feature_count']
                 dummy_input_y = np.zeros((1, self.poly_features_y.n_features_in_))
                 self.poly_features_y.fit(dummy_input_y)
            else:
                print("Warning: 'poly_features_y_n_features_in' not found in loaded model. Model may not transform features correctly.")

            self.model_y = LinearRegression()
            self.model_y.coef_ = model_data['model_y_coef']
            self.model_y.intercept_ = model_data['model_y_intercept']
            
            self.is_trained = model_data.get('is_trained', False)
            self._expected_feature_count = model_data.get('_expected_feature_count', -1)
            
            if self.is_trained and hasattr(self.model_x, 'coef_') and hasattr(self.model_y, 'coef_'):
                print(f"Model successfully loaded from {filepath}. Expected feature count: {self._expected_feature_count}")
                return True
            else:
                print(f"Model loaded from {filepath}, but seems incomplete or marked as not trained.")
                self.is_trained = False
                return False

        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            self.is_trained = False
            return False

if __name__ == "__main__":
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    dummy_calibration_file = "dummy_calibration_data.json"
    dummy_data = [
        {"target_screen_px": [100, 100], "raw_gaze_camera_px": [50, 50], 
         "normalized_pupil_coord_xy": [0.5, 0.5], "head_pose_rvec": [0.1, 0.1, 0.1], "head_pose_tvec": [1, 1, -10]},
        {"target_screen_px": [1800, 100], "raw_gaze_camera_px": [600, 50], 
         "normalized_pupil_coord_xy": [0.5, 0.5], "head_pose_rvec": [-0.1, 0.1, 0.1], "head_pose_tvec": [-1, 1, -10]},
        {"target_screen_px": [100, 1000], "raw_gaze_camera_px": [50, 450], 
         "normalized_pupil_coord_xy": [0.5, 0.5], "head_pose_rvec": [0.1, -0.1, 0.1], "head_pose_tvec": [1, -1, -10]},
        {"target_screen_px": [1800, 1000], "raw_gaze_camera_px": [600, 450], 
         "normalized_pupil_coord_xy": [0.5, 0.5], "head_pose_rvec": [0.0, 0.0, 0.0], "head_pose_tvec": [0, 0, -12]},
        {"target_screen_px": [960, 540], "raw_gaze_camera_px": [320, 240], 
         "normalized_pupil_coord_xy": [0.5, 0.5], "head_pose_rvec": [0.2, 0.1, -0.1], "head_pose_tvec": [0.5, -0.5, -11]}
    ]
    for i in range(10): # Generate enough samples to pass the min_samples_needed check
        dummy_data.append({
            "target_screen_px": [np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)],
            "raw_gaze_camera_px": [np.random.randint(0, CAMERA_WIDTH), np.random.randint(0, CAMERA_HEIGHT)],
            "normalized_pupil_coord_xy": [np.random.rand(), np.random.rand()],
            "head_pose_rvec": (np.random.rand(3) * 0.4 - 0.2).tolist(),
            "head_pose_tvec": (np.array([np.random.uniform(-5,5), np.random.uniform(-5,5), np.random.uniform(-20,-5)])).tolist()
        })

    with open(dummy_calibration_file, 'w') as f:
        json.dump(dummy_data, f)

    model = GazeToScreenModel(
        screen_width=SCREEN_WIDTH, 
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT
    )

    print("\n--- Training Model ---")
    model.train(dummy_calibration_file)

    print("\n--- Testing Prediction (Post-Training) ---")
    gaze_inputs = [
        (50, 50),
        (320, 240),
        (600, 450),
        (0, 0),
        (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1)
    ]
    if model.is_trained:
        for gx, gy in gaze_inputs:
            print(f"Input Gaze (Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}): ({gx}, {gy})")
            pred_x, pred_y = model.predict(gx, gy, 
                                           pupil_norm_x=0.5, pupil_norm_y=0.5, 
                                           rvec=[0.05, 0.05, 0.05], tvec=[0.1, 0.1, -10])
            print(f"Predicted Screen Coords (Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}): ({pred_x}, {pred_y})")
            
            print("Testing prediction with missing features (should fallback):")
            pred_x_fallback, pred_y_fallback = model.predict(gx, gy)
            print(f"Predicted Screen Coords (Fallback): ({pred_x_fallback}, {pred_y_fallback})")
            print("-" * 20)
    else:
        print("Model was not trained successfully, skipping prediction test.")

    print("\n--- Testing Model Loading ---")
    model_loaded = GazeToScreenModel(
        screen_width=SCREEN_WIDTH, 
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT
    )
    if model_loaded.is_trained:
        print("Loaded model is trained. Testing prediction:")
        for gx, gy in gaze_inputs:
            print(f"Input Gaze (Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}): ({gx}, {gy})")
            pred_x, pred_y = model_loaded.predict(gx, gy, 
                                                  pupil_norm_x=0.51, pupil_norm_y=0.49, 
                                                  rvec=[-0.05, -0.05, -0.05], tvec=[-0.1, -0.1, -10.5])
            print(f"Predicted Screen Coords (Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}): ({pred_x}, {pred_y})")

            print("Testing loaded model prediction with missing features (should fallback):")
            pred_x_fallback, pred_y_fallback = model_loaded.predict(gx, gy)
            print(f"Predicted Screen Coords (Fallback): ({pred_x_fallback}, {pred_y_fallback})")
            print("-" * 20)
    else:
        print("Loaded model is NOT trained or failed to load properly.")

    if os.path.exists(dummy_calibration_file):
        os.remove(dummy_calibration_file)
    if os.path.exists(model.MODEL_FILENAME):
        os.remove(model.MODEL_FILENAME)
