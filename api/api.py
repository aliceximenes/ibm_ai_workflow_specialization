from flask import Flask, jsonify, request, send_from_directory
from model import model_train, model_predict, DATA_DIR
import os

app = Flask(__name__)
LOGS_DIR = "./logs"

@app.route("/api/train", methods=["GET"])
def train():
    test = request.args.get("test")
    if test is None:
        test = ""
    model_train(DATA_DIR, test=test.lower()=="true" )
    return jsonify("Model trained")

@app.route("/api/predict", methods=["GET"])
def predict():
    country = request.args.get("country")
    year = request.args.get("year")
    month = request.args.get("month")
    day = request.args.get("day")
    return jsonify(model_predict(country, year, month, day))

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(os.listdir(LOGS_DIR))

@app.route("/api/logs/<filename>", methods=["GET"])
def get_log(filename):
    file_path = os.path.join(LOGS_DIR, filename)
    if os.path.exists(file_path) and filename.endswith(".log"):
        return send_from_directory(LOGS_DIR, filename)
    else:
        jsonify(None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050, debug=True)
