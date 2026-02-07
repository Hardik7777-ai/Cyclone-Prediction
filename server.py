from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('cyclone_model_xgb (1).pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        features = [
            data['Sea_Surface_Temperature'],
            data['Atmospheric_Pressure'],
            data['Humidity'],
            data['Wind_Shear'],
            data['Vorticity']
        ]

        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0][int(prediction)]

        prediction_text = "Cyclone likely" if prediction == 1 else "No Cyclone"
        response = {
            "prediction_text": prediction_text,
            "prediction_probability": f"{round(prob * 100, 2)}%"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
