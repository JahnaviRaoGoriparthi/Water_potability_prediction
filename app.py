from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import subprocess
import sys

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Check if catboost is installed, if not, install it
try:
    import catboost
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])

# Load the model
model_path = 'static/catboost_gsa_model.pkl'
model = joblib.load(model_path)

# Define a function for preprocessing
def preprocess(data):
    columns_to_drop = ['Organic_carbon', 'Turbidity']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    data.fillna(data.mean(), inplace=True)
    return data

# Define a function for making predictions
def predict(input_data):
    input_data = preprocess(input_data)
    predictions = model.predict(input_data)
    return predictions

# Define route for prediction
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # Get JSON data from POST request
        json_data = request.get_json()
        input_data = pd.DataFrame([json_data])  # Wrap json_data in a list to create DataFrame
        
        # Print input data to the terminal
        print("Input data:", input_data)
        
        # Make predictions
        predictions = predict(input_data)
        
        # Print predictions to the terminal
        print("Predictions:", predictions)
        
        # Return predictions as JSON response
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': 'Prediction failed. Please check your input data.'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
