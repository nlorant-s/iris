import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import json

class GazeToScreenModel:
    MODEL_FILENAME = "gaze_model.joblib" # Used by __main__ to construct default path

    def __init__(self, screen_width, screen_height, camera_width, camera_height, model_path): # Changed model_dir to model_path
        """
        Initializes the gaze to screen model.
        Tries to load a pre-trained model using the provided full model_path.
        If not found, initializes untrained models.

        Args:
            screen_width (int): The width of the screen in pixels.
            screen_height (int): The height of the screen in pixels.
            camera_width (int): The width of the camera frame in pixels.
            camera_height (int): The height of the camera frame in pixels.
            model_path (str): Full path to the model file (.joblib).
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.scaler = StandardScaler()  # Initialize scaler
        self.poly_features_x = PolynomialFeatures(degree=2, include_bias=False)
        self.model_x = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
        self.poly_features_y = PolynomialFeatures(degree=2, include_bias=False)
        self.model_y = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
        
        self.is_trained = False
        self.model_path = model_path # Use the provided full path
        self._original_model_path_for_reinit = model_path # Store for re-initialization on load failure
        self._expected_feature_count = -1 

        print(f"GazeToScreenModel initialized for screen: {screen_width}x{screen_height}, camera: {camera_width}x{camera_height}")
        # Removed self._model_dir_for_init as model_path is now the full path
        if self.load_model(self.model_path):
            if self.is_trained:
                print(f"Loaded pre-trained model from {self.model_path}")
            else:
                print(f"Model structure loaded from {self.model_path}, but it's not trained or failed validation. Model needs training.")
        else:
            print(f"No pre-trained model found at {self.model_path} or failed to load. Model needs training.")

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

        grouped_data = {}
        for i, entry in enumerate(calibration_data_input):
            entry['_original_index'] = i
            target_px_tuple = tuple(entry.get("target_screen_px", [None,None]))
            if None in target_px_tuple:
                print(f"Warning: Entry {i} missing target_screen_px, cannot group for local outlier removal. Skipping this entry for now.")
                continue
            if target_px_tuple not in grouped_data:
                grouped_data[target_px_tuple] = []
            grouped_data[target_px_tuple].append(entry)

        filtered_calibration_data = []
        for target_px, entries_at_target in grouped_data.items():
            if len(entries_at_target) < 3:
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

            if len(valid_entries_for_target) < 3:
                filtered_calibration_data.extend(valid_entries_for_target)
                print(f"Target {target_px}: Kept {len(valid_entries_for_target)} valid points (too few to filter after raw_gaze validation).")
                continue
            
            gaze_points_at_target_np = np.array(gaze_points_at_target)
            centroid = np.mean(gaze_points_at_target_np, axis=0)
            distances = np.linalg.norm(gaze_points_at_target_np - centroid, axis=1)
            
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            MIN_STD_DEV_FOR_FILTERING = 0.1 
            z_score_threshold = 2.5 
            
            kept_entries_for_target = []
            removed_count = 0
            if std_distance < MIN_STD_DEV_FOR_FILTERING:
                kept_entries_for_target.extend(valid_entries_for_target)
                print(f"Target {target_px}: Kept all {len(valid_entries_for_target)} points (std dev of distances {std_distance:.4f} < {MIN_STD_DEV_FOR_FILTERING}).")
            else:
                for i, entry in enumerate(valid_entries_for_target):
                    z_score = (distances[i] - mean_distance) / std_distance if std_distance > 1e-9 else 0
                    if abs(z_score) <= z_score_threshold:
                        kept_entries_for_target.append(entry)
                    else:
                        removed_count +=1
                        print(f"Target {target_px}: Removed outlier entry (original_index {entry.get('_original_index')}) with raw_gaze {entry.get('raw_gaze_camera_px')}, z-score {z_score:.2f}.")
                print(f"Target {target_px}: Kept {len(kept_entries_for_target)} points, removed {removed_count} outliers.")

            filtered_calibration_data.extend(kept_entries_for_target)
        
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

        def _validate_list_of_numbers(data, expected_len, name_for_error):
            if not (isinstance(data, list) and len(data) == expected_len and
                    all(isinstance(v, (int, float)) for v in data)):
                return False, None, f"Invalid structure/type for {name_for_error}: {data}"
            if any(np.isnan(v) or np.isinf(v) for v in data):
                return False, None, f"Invalid numeric value (NaN/Inf) in {name_for_error}: {data}"
            return True, data, None

        for entry_idx, entry in enumerate(calibration_data):
            is_entry_valid = True
            error_messages = []
            
            valid_struct, raw_gaze_val, err_msg = _validate_list_of_numbers(entry.get("raw_gaze_camera_px"), 2, "raw_gaze_camera_px")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            pupil_coord_data = entry.get("normalized_pupil_coord_xy")
            if pupil_coord_data is None:
                pupil_coord_data = entry.get("avg_normalized_pupil_coord_xy")
            valid_struct, pupil_coord_val, err_msg = _validate_list_of_numbers(pupil_coord_data, 2, "normalized_pupil_coord_xy or avg_normalized_pupil_coord_xy")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            rvec_data = entry.get("head_pose_rvec")
            if rvec_data is None:
                rvec_data = entry.get("avg_head_pose_rvec")
            valid_struct, rvec_val, err_msg = _validate_list_of_numbers(rvec_data, 3, "head_pose_rvec or avg_head_pose_rvec")
            if not valid_struct: is_entry_valid = False; error_messages.append(err_msg)

            tvec_data = entry.get("head_pose_tvec")
            if tvec_data is None:
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

        n_poly_features = 44
        min_samples_needed = n_poly_features + 1

        if valid_entries < min_samples_needed:
            print(f"Error: Not enough valid data points ({valid_entries}) after filtering. "
                  f"Need at least {min_samples_needed} for the current model configuration (8 base features, degree 2 polynomial).")
            self.is_trained = False
            return False

        screen_points = np.array(target_screen_coords, dtype=float)
        pupil_points = np.array(pupil_coords_list, dtype=float)
        rvec_points = np.array(head_rvecs_list, dtype=float)
        tvec_points = np.array(head_tvecs_list, dtype=float)

        feature_arrays_map = {
            "pupil_points": pupil_points,
            "rvec_points": rvec_points, "tvec_points": tvec_points
        }
        for name, arr in feature_arrays_map.items():
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Error: NaN or Inf detected in '{name}' array after processing. Training aborted.")
                self.is_trained = False
                return False
        
        all_features = np.concatenate((pupil_points, rvec_points, tvec_points), axis=1)
        self._expected_feature_count = all_features.shape[1]

        # Re-initialize the scaler for training to ensure it's fresh and avoids issues with partially loaded states
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(all_features)

        gaze_X_features = scaled_features
        screen_X_target = screen_points[:, 0]
        
        gaze_Y_features = scaled_features
        screen_Y_target = screen_points[:, 1]

        try:
            gaze_X_poly = self.poly_features_x.fit_transform(gaze_X_features)
            self.model_x.fit(gaze_X_poly, screen_X_target)

            gaze_Y_poly = self.poly_features_y.fit_transform(gaze_Y_features)
            self.model_y.fit(gaze_Y_poly, screen_Y_target)
            
            screen_X_pred = self.model_x.predict(gaze_X_poly)
            screen_Y_pred = self.model_y.predict(gaze_Y_poly)
            
            mse_x = mean_squared_error(screen_X_target, screen_X_pred)
            mse_y = mean_squared_error(screen_Y_target, screen_Y_pred)
            print(f"Training complete. MSE_X: {mse_x:.4f}, MSE_Y: {mse_y:.4f}")
            
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
                          self._expected_feature_count == 8)

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

            current_features = np.array([[pupil_norm_x, pupil_norm_y, 
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

            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                scaled_current_features = self.scaler.transform(current_features)
            else:
                print("Warning: Scaler not fitted during prediction. Using unscaled features.")
                scaled_current_features = current_features

            features_poly_x = self.poly_features_x.transform(scaled_current_features)
            screen_x = self.model_x.predict(features_poly_x)[0]
            
            features_poly_y = self.poly_features_y.transform(scaled_current_features)
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
                'model_x_object': self.model_x,

                'poly_features_y_params': self.poly_features_y.get_params(),
                'model_y_object': self.model_y,

                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'camera_width': self.camera_width,
                'camera_height': self.camera_height,
                'is_trained': self.is_trained,
                'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'scaler_n_features_in': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
                'scaler_n_samples_seen': self.scaler.n_samples_seen_ if hasattr(self.scaler, 'n_samples_seen_') else None,
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
        print(f"Debug: Attempting to load model from {filepath}") # Added
        try:
            if not os.path.exists(filepath):
                print(f"Debug: File not found at {filepath}") # Added
                return False
                
            model_data = joblib.load(filepath)
            print(f"Debug: Successfully loaded data from {filepath}") # Added

            self.screen_width = model_data.get('screen_width', self.screen_width)
            self.screen_height = model_data.get('screen_height', self.screen_height)
            self.camera_width = model_data.get('camera_width', self.camera_width)
            self.camera_height = model_data.get('camera_height', self.camera_height)
            self._expected_feature_count = model_data.get('_expected_feature_count', -1)
            print(f"Debug: _expected_feature_count loaded as: {self._expected_feature_count}") # Added

            if model_data.get('scaler_mean') is not None and model_data.get('scaler_scale') is not None:
                self.scaler = StandardScaler()
                self.scaler.mean_ = model_data['scaler_mean']
                self.scaler.scale_ = model_data['scaler_scale']
                if 'scaler_n_features_in' in model_data: self.scaler.n_features_in_ = model_data['scaler_n_features_in']
                if 'scaler_n_samples_seen' in model_data: self.scaler.n_samples_seen_ = model_data['scaler_n_samples_seen']
                print(f"Debug: Scaler loaded with mean: {self.scaler.mean_ is not None}, scale: {self.scaler.scale_ is not None}") # Added
                print(f"Debug: Scaler n_features_in_: {getattr(self.scaler, 'n_features_in_', 'N/A')}, n_samples_seen_: {getattr(self.scaler, 'n_samples_seen_', 'N/A')}") # Added
            else:
                self.scaler = StandardScaler()
                print("Debug: Scaler re-initialized (no mean/scale in model_data)") # Added

            poly_x_params = model_data.get('poly_features_x_params', {'degree': 2, 'include_bias': False})
            self.poly_features_x = PolynomialFeatures(**poly_x_params)
            n_features_in_x = model_data.get('poly_features_x_n_features_in')
            if n_features_in_x is None and self._expected_feature_count > 0:
                n_features_in_x = self._expected_feature_count
            print(f"Debug: poly_features_x - n_features_in_ to be used: {n_features_in_x}") # Added
            
            if n_features_in_x is not None and n_features_in_x > 0:
                self.poly_features_x.n_features_in_ = n_features_in_x
                try:
                    # Fit with dummy data to set internal states if necessary
                    self.poly_features_x.fit(np.zeros((1, n_features_in_x)))
                    print(f"Debug: poly_features_x fitted with n_features_in_={n_features_in_x}") # Added
                except Exception as e_fit_poly_x:
                    print(f"Warning Detail: Could not fit poly_features_x with n_features_in_={n_features_in_x}: {e_fit_poly_x}")

            poly_y_params = model_data.get('poly_features_y_params', {'degree': 2, 'include_bias': False})
            self.poly_features_y = PolynomialFeatures(**poly_y_params)
            n_features_in_y = model_data.get('poly_features_y_n_features_in')
            if n_features_in_y is None and self._expected_feature_count > 0:
                n_features_in_y = self._expected_feature_count
            print(f"Debug: poly_features_y - n_features_in_ to be used: {n_features_in_y}") # Added

            if n_features_in_y is not None and n_features_in_y > 0:
                self.poly_features_y.n_features_in_ = n_features_in_y
                try:
                    # Fit with dummy data to set internal states if necessary
                    self.poly_features_y.fit(np.zeros((1, n_features_in_y)))
                    print(f"Debug: poly_features_y fitted with n_features_in_={n_features_in_y}") # Added
                except Exception as e_fit_poly_y:
                    print(f"Warning Detail: Could not fit poly_features_y with n_features_in_={n_features_in_y}: {e_fit_poly_y}")

            if 'model_x_object' in model_data and 'model_y_object' in model_data:
                self.model_x = model_data['model_x_object']
                self.model_y = model_data['model_y_object']
                print(f"Debug: model_x type: {type(self.model_x)}, model_y type: {type(self.model_y)}") # Added

                core_model_objects_structurally_ok = isinstance(self.model_x, BaggingRegressor) and isinstance(self.model_y, BaggingRegressor)
                if not core_model_objects_structurally_ok:
                    print("Warning Detail: model_x or model_y are not BaggingRegressors prior to is_trained check. This implies a load issue not caught by earlier print.")

                loaded_is_trained_from_file = model_data.get('is_trained', False)
                print(f"Debug: 'is_trained' from file: {loaded_is_trained_from_file}")

                if core_model_objects_structurally_ok and loaded_is_trained_from_file:
                    self.is_trained = True
                    print(f"Debug: Preliminary self.is_trained status (set to True for detailed checks): {self.is_trained}")
                else:
                    self.is_trained = False
                    print(f"Debug: Preliminary self.is_trained status (models not ok or file says not trained, skipping detailed checks): {self.is_trained}")

                if self.is_trained:
                    model_x_ok = hasattr(self.model_x, 'estimators_') and len(self.model_x.estimators_) > 0
                    model_y_ok = hasattr(self.model_y, 'estimators_') and len(self.model_y.estimators_) > 0
                    print(f"Debug: model_x_ok: {model_x_ok} (estimators: {len(getattr(self.model_x, 'estimators_', []))})") # Added
                    print(f"Debug: model_y_ok: {model_y_ok} (estimators: {len(getattr(self.model_y, 'estimators_', []))})") # Added

                    poly_x_fitted = hasattr(self.poly_features_x, 'n_features_in_') and self.poly_features_x.n_features_in_ > 0
                    poly_y_fitted = hasattr(self.poly_features_y, 'n_features_in_') and self.poly_features_y.n_features_in_ > 0
                    print(f"Debug: poly_x_fitted: {poly_x_fitted} (n_features_in_: {getattr(self.poly_features_x, 'n_features_in_', 'N/A')})") # Added
                    print(f"Debug: poly_y_fitted: {poly_y_fitted} (n_features_in_: {getattr(self.poly_features_y, 'n_features_in_', 'N/A')})") # Added
                    expected_feature_count_ok = self._expected_feature_count > 0
                    print(f"Debug: _expected_feature_count > 0: {expected_feature_count_ok}") # Added

                    if not (model_x_ok and model_y_ok and poly_x_fitted and poly_y_fitted and expected_feature_count_ok):
                        print(f"Warning Detail: Model file at {filepath} marked as trained, but components are inconsistent or not properly loaded/fitted. Treating as untrained.")
                        print(f"Warning Detail: Breakdown - model_x_ok: {model_x_ok}, model_y_ok: {model_y_ok}, poly_x_fitted: {poly_x_fitted}, poly_y_fitted: {poly_y_fitted}, expected_feature_count_ok: {expected_feature_count_ok}") # Added
                        self.is_trained = False
                    else:
                        try:
                            dummy_input = np.zeros((1, self._expected_feature_count))
                            print(f"Debug: Validating with dummy_input shape: {dummy_input.shape}") # Added

                            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == self._expected_feature_count:
                                    self.scaler.transform(dummy_input)
                                    print("Debug: scaler.transform(dummy_input) successful.") # Added
                                else:
                                    print(f"Warning Detail: Scaler n_features_in_ ({getattr(self.scaler, 'n_features_in_', 'N/A')}) mismatch with _expected_feature_count ({self._expected_feature_count}). Scaler transform check skipped.")
                            else:
                                print("Debug: Scaler not fitted (no mean_), transform check skipped.") # Added

                            self.poly_features_x.transform(dummy_input)
                            print("Debug: poly_features_x.transform(dummy_input) successful.") # Added
                            self.poly_features_y.transform(dummy_input)
                            print("Debug: poly_features_y.transform(dummy_input) successful.") # Added
                            print("Debug: All validation checks passed for trained model.") # Added

                        except Exception as e_val:
                            print(f"Warning Detail: Error validating loaded trained model components (transform failed): {e_val}. Treating as untrained.")
                            self.is_trained = False

                    print(f"Debug: Final self.is_trained status after all checks: {self.is_trained}") # Added
                    return True

            else:
                print("Warning Detail: Saved model objects ('model_x_object', 'model_y_object') not found. This model file is incompatible or incomplete. Model will be treated as untrained.")
                self.model_x = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
                self.model_y = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
                self.is_trained = False

            return True

        except FileNotFoundError:
            print(f"Debug: FileNotFoundError caught for {filepath}") # Added
            self.is_trained = False
            return False 
        except Exception as e:
            print(f"General error loading model from {filepath}: {e}. Model will be re-initialized as untrained.")
            import traceback
            traceback.print_exc()
            current_screen_width = self.screen_width
            current_screen_height = self.screen_height
            current_camera_width = self.camera_width
            current_camera_height = self.camera_height

            self.screen_width = current_screen_width
            self.screen_height = current_screen_height
            self.camera_width = current_camera_width
            self.camera_height = current_camera_height
            self.scaler = StandardScaler()
            self.poly_features_x = PolynomialFeatures(degree=2, include_bias=False)
            self.model_x = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
            self.poly_features_y = PolynomialFeatures(degree=2, include_bias=False)
            self.model_y = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10, random_state=42)
            self.is_trained = False
            self.model_path = self._original_model_path_for_reinit # Use stored original path for re-init
            self._expected_feature_count = -1
            return False

if __name__ == "__main__":
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_file_path = os.path.join(script_dir, "..", "data", "training", "training-data-0336.json")
    calibration_file_path = os.path.normpath(calibration_file_path)
    
    # Construct the default full model path for __main__
    default_model_path = os.path.normpath(os.path.join(script_dir, GazeToScreenModel.MODEL_FILENAME))

    if not os.path.exists(calibration_file_path):
        print(f"Error: Calibration data file not found at {calibration_file_path}")
        print("Please ensure the file exists and the path is correct.")
    
    model = GazeToScreenModel(
        screen_width=SCREEN_WIDTH, 
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        model_path=default_model_path # Pass the full model path
    )

    print(f"\n--- Training Model using {calibration_file_path} ---")
    training_success = model.train(calibration_file_path)

    print("\n--- Testing Prediction (Post-Training) ---")
    gaze_inputs = [
        (50, 50),
        (320, 240),
        (600, 450),
        (0, 0),
        (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1)
    ]
    if model.is_trained and training_success:
        print("Model was trained successfully.")
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
        print(f"Model training with {calibration_file_path} was not successful. Skipping prediction test.")

    print("\n--- Testing Model Loading ---")
    
    model_loaded = GazeToScreenModel(
        screen_width=SCREEN_WIDTH, 
        screen_height=SCREEN_HEIGHT,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        model_path=default_model_path # Pass the full model path
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
        print("Loaded model is NOT trained or failed to load properly (this is expected if training failed or no model was saved).")

    if os.path.exists(model.model_path): # model.model_path should be correct now
        print(f"Removing model file: {model.model_path}")
        os.remove(model.model_path)
