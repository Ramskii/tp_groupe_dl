import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import tensorflow as tf
import cv2
from collections import OrderedDict

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.secret_key = "secret key"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)

        pred = model.predict(np.array([img_resized]))

        cla = OrderedDict()
        for cl, indice in class_indices.items():
            cla[cl] = round(pred[0][indice], 4)
        cla = sorted(cla.items(), key=lambda x:x[1], reverse=True)

        return render_template('upload.html', filename=filename, pred=cla[0][0], proba=str(cla[0][1]))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    
    with open("./model/class_indices.json", "r") as f:
        class_indices = json.load(f)

    model = tf.keras.models.load_model('./model/vgg_intel_v1.h5')

    app.run(debug=True)