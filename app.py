from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            model = joblib.load("./model.pkl")
            input = []

            for key in data:
                input.append(data[key])

            print(input)

        except ValueError:
            return jsonify("Invalid input data.")

        input.reshape(1, -1)
        return jsonify(model.predict(input).tolist())
            

if __name__ == '__main__':
    app.run(debug=True)