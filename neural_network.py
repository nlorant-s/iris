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

        Args:
            calibration_data_file (str): Path to the JSON file containing calibration data.
        """
        try:
            with open(calibration_data_file, 'r') as f:
                calibration_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Calibration data file not found: {calibration_data_file}")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {calibration_data_file}")
            return False

        raw_gaze_coords = []
        target_screen_coords = []
        pupil_coords_list = []
        head_rvecs_list = []
        head_tvecs_list = []

        valid_entries = 0
        for entry in calibration_data:
            raw_gaze = entry.get("raw_gaze_camera_px")
            pupil_coord = entry.get("avg_normalized_pupil_coord_xy")
            rvec = entry.get("avg_head_pose_rvec")
            tvec = entry.get("avg_head_pose_tvec")
            target_px = entry.get("target_screen_px")

            if (raw_gaze and raw_gaze[0] is not None and raw_gaze[1] is not None and
                pupil_coord and pupil_coord[0] is not None and pupil_coord[1] is not None and
                rvec and len(rvec) == 3 and all(v is not None for v in rvec) and
                tvec and len(tvec) == 3 and all(v is not None for v in tvec) and
                target_px and target_px[0] is not None and target_px[1] is not None and
                entry.get("samples", 0) > 0):
                
                raw_gaze_coords.append(raw_gaze)
                pupil_coords_list.append(pupil_coord)
                head_rvecs_list.append(rvec)
                head_tvecs_list.append(tvec)
                target_screen_coords.append(target_px)
                valid_entries += 1
            else:
                print(f"Skipping invalid or incomplete entry: {entry}")

        if valid_entries < 2:
            print(f"Error: Not enough valid data points ({valid_entries}) in calibration data to train the model. Need at least 2.")
            return False

        gaze_points = np.array(raw_gaze_coords)
        screen_points = np.array(target_screen_coords)
        pupil_points = np.array(pupil_coords_list)
        rvec_points = np.array(head_rvecs_list)
        tvec_points = np.array(head_tvecs_list)

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
        {"target_screen_px": [100, 100], "raw_gaze_camera_px": [50, 50], "samples": 1,
         "avg_normalized_pupil_coord_xy": [0.5, 0.5], "avg_head_pose_rvec": [0.1, 0.1, 0.1], "avg_head_pose_tvec": [1, 1, -10]},
        {"target_screen_px": [1800, 100], "raw_gaze_camera_px": [600, 50], "samples": 1,
         "avg_normalized_pupil_coord_xy": [0.5, 0.5], "avg_head_pose_rvec": [-0.1, 0.1, 0.1], "avg_head_pose_tvec": [-1, 1, -10]},
        {"target_screen_px": [100, 1000], "raw_gaze_camera_px": [50, 450], "samples": 1,
         "avg_normalized_pupil_coord_xy": [0.5, 0.5], "avg_head_pose_rvec": [0.1, -0.1, 0.1], "avg_head_pose_tvec": [1, -1, -10]},
        {"target_screen_px": [1800, 1000], "raw_gaze_camera_px": [600, 450], "samples": 1,
         "avg_normalized_pupil_coord_xy": [0.5, 0.5], "avg_head_pose_rvec": [0.0, 0.0, 0.0], "avg_head_pose_tvec": [0, 0, -12]},
        {"target_screen_px": [960, 540], "raw_gaze_camera_px": [320, 240], "samples": 1,
         "avg_normalized_pupil_coord_xy": [0.5, 0.5], "avg_head_pose_rvec": [0.2, 0.1, -0.1], "avg_head_pose_tvec": [0.5, -0.5, -11]}
    ]
    for i in range(10):
        dummy_data.append({
            "target_screen_px": [np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)],
            "raw_gaze_camera_px": [np.random.randint(0, CAMERA_WIDTH), np.random.randint(0, CAMERA_HEIGHT)],
            "samples": 1,
            "avg_normalized_pupil_coord_xy": [np.random.rand(), np.random.rand()],
            "avg_head_pose_rvec": (np.random.rand(3) * 0.4 - 0.2).tolist(),
            "avg_head_pose_tvec": (np.array([np.random.uniform(-5,5), np.random.uniform(-5,5), np.random.uniform(-20,-5)])).tolist()
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
