from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from tensorflow.keras.models import load_model
import numpy as np
from util import base64_to_pil

app = Flask(__name__)

MODEL_PATH = "./ResNet152_model.h5"

model = load_model(MODEL_PATH)


def model_predict(img, model):
    img = np.array(img)
    preds = model.predict(img[None,:,:,:3])
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        preds = model_predict(img, model)
        print(preds)
        y_class = np.argmax(preds)
        print(y_class)
        lesion_type = [
            'Actinic keratoses',
            'Basal cell carcinoma',
            'Benign keratosis-like lesions ',
            'Dermatofibroma',
            'Melanocytic nevi',
            'Melanoma',
            'Vascular lesions'
        ]
        result = str(lesion_type[y_class])
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    app.run()