import numpy as np
from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            model = joblib.load("./model.pkl")
            query = []

            for key in data:
                query.append(data[key])

            input = np.array(query).reshape(1, -1)

        except ValueError:
            return jsonify("Invalid input data.")

        return jsonify(model.predict(input).tolist())
            

if __name__ == '__main__':
    app.run(debug=True)