from flask import Flask, request
import pickle
import numpy as np
# Nayi file import ki (Agar aap ek aur .py file banate hain)
# from data import feature_names 

app = Flask(__name__)

# Aapke 13 features ke naam
FEATURE_NAMES = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

# HTML Form ko Python Multi-line String mein define karein
# Isse aapka sara HTML code Python file mein count hoga.
HTML_FORM_CONTENT = f"""
<form method="POST" action="/predict">
    <h2>Boston Housing Price Prediction</h2>
    <p>Please enter the 13 required features:</p>
    {
        ''.join([f'<label for="{name}">{name}:</label><input type="text" name="{name}" required><br>' 
                 for name in FEATURE_NAMES])
    }
    <input type="submit" value="Predict Price">
</form>
"""

# Model loading (As it is)
try:
    with open('housppp_price_prediction.pkl', 'rb') as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False
    
# Function to create the full HTML response
def create_html_response(prediction_text=""):
    # Yahaan hum HTML ko code kar rahe hain
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Prediction App</title>
        <style>body{{font-family: Arial;}}</style>
    </head>
    <body>
        <h1>ML Prediction Service (Python-Heavy Code)</h1>
        {HTML_FORM_CONTENT}
        <hr>
        <p style="font-size: 20px; font-weight: bold;">{prediction_text}</p>
    </body>
    </html>
    """
    return full_html


@app.route('/')
def home():
    # render_template ki jagah seedhe HTML string return karein
    return create_html_response() 

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return create_html_response(prediction_text="Error: Model file could not be loaded.")
        
    try:
        # features list ko for loop se banao, jo Python code ko badhata hai
        features = []
        for name in FEATURE_NAMES:
            features.append(float(request.form[name]))

        # NumPy array aur prediction logic as it is
        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)

        # Result ko usi function se pass karein
        return create_html_response(prediction_text=f"Predicted Price: ${output}k")

    except Exception as e:
        # Error handling
        return create_html_response(prediction_text=f"Input Error: Please check all 13 fields. Details: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
