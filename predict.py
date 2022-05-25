from flask import Flask, request
import os
import tensorflow as tf
import json
import decimal


app = Flask(__name__)
images = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = "images"


@app.route("/")
def visualisations():
    image_string = "".join(
        f'<div style="margin: auto;"><img src="/static/images/{name} comments.png"></div><br>'
        for name in ["Negative", "Positive", "Neutral"]
    )
    return f'<div style="width: 100%">{image_string}</div>'


@app.route("/predict", methods=["GET"])
def predict():
    model = tf.keras.models.load_model("saved_model/my_model")
    result = model.predict([request.args.get("tweet")])[0]
    return {
        "result": float(result[0]),
        "message": "Negative, positive, neutral and unknown correspond to integers: 0 to 3 respectively",
    }
