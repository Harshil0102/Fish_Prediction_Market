from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model/model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'model/scaler.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate form data
        features = [request.form.get(f) for f in ['Length1', 'Length2', 'Length3', 'Height', 'Width']]
        if None in features or '' in features:
            return render_template('index.html', prediction_text='Please fill out all fields.')

        # Convert features to floats
        features = [float(x) for x in features]
        final_features = [np.array(features)]
        
        # Scale the features
        final_features_scaled = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features_scaled)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted Weight of Fish: {output}g')
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter valid numbers for all fields.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
