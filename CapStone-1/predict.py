import pickle
from flask import Flask, jsonify, request

with open('model_logistic.bin', 'rb') as f:
   model, dv = pickle.load(f)


app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.get_json()
        X = dv.transform([features])
        y_pred = model.predict(X)

        prediction = ' <=50K' if y_pred[0] == 1 else ' >50K'

        response = {
            "status": prediction
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=4041)