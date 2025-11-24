import numpy as np

# Feature Names ko yahan rakho
FEATURE_NAMES = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

def create_feature_array(form_data):
    """
    Form data ko process karke NumPy array mein badalta hai.
    """
    try:
        # Features ko list mein collect karna
        features = []
        for name in FEATURE_NAMES:
            # Data validation yahan bhi ho rahi hai
            value = float(form_data.get(name))
            features.append(value)
            
        # 2D array mein reshape karna model ke liye
        features_array = np.array([features])
        return features_array
    except (TypeError, ValueError) as e:
        # Agar koi input number nahi hai toh error throw karna
        raise ValueError(f"Invalid input provided for one or more features: {e}")

class ModelPredictor:
    """
    Model ko load aur manage karne ke liye ek class.
    """
    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Pickle file se model load karta hai."""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("ML Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def predict(self, features_array):
        """Loaded model se prediction karta hai."""
        if self.model is None:
            raise Exception("Model is not loaded. Cannot predict.")
            
        prediction = self.model.predict(features_array)
        # Prediction ko round karke return karna
        output = round(prediction[0], 2)
        return output
