from flask import Flask, render_template, request, redirect, url_for
from test_ import predict_cancer_risk
from regression import predict_percentage
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Özellik sütunları
feature_columns = [
    'Age', 'Gender', 'AirPollution', 'Alcoholuse', 'DustAllergy', 'OccuPationalHazards',
    'GeneticRisk', 'chronicLungDisease', 'BalancedDiet', 'Obesity', 'Smoking',
    'PassiveSmoker', 'ChestPain', 'CoughingofBlood', 'Fatigue', 'WeightLoss',
    'ShortnessofBreath', 'Wheezing', 'SwallowingDifficulty', 'ClubbingofFingerNails',
    'FrequentCold', 'DryCough', 'Snoring'
]

class_labels = ['adenocarcinomia', 'large.cell.carcinomia','normal', 'squamous.cell.carcinomia']
model_path = "final_model_xception.keras"  # Eğitilmiş modelin dosya yolu
model = load_model(model_path)  # Kaydedilen modeli yüklendi

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Xception modeli için 299x299x3 istiyo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizasyon yaptık
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    return predicted_label

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[col]) for col in feature_columns]
            cancer_risk = predict_cancer_risk(features)
            risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            risk_level = risk_map[cancer_risk]
            percentage = predict_percentage(features)
            return render_template('index.html', feature_columns=feature_columns, cancer_risk=risk_level, percentage=percentage)
        except ValueError as e:
            return str(e), 400
    return render_template('index.html', feature_columns=feature_columns)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predicted_label = classify_image(filepath)
            return render_template('upload_image.html', label=predicted_label)
    return render_template('upload_image.html')

if __name__ == '__main__':
    app.run(debug=True)