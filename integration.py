import os
import cv2
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

MODEL_PATH = 'rf_model.pkl'
EXCEL_PATH = r"D:\college_M_project\project\Book1.xlsx"
DATASET_PATH = r"D:\college_M_project\project\rice_leaf_diseases"

def load_excel_data():
    """Loads disease data from Excel and extracts necessary columns."""
    excel_data = pd.read_excel(EXCEL_PATH, engine="openpyxl", header=0)
    excel_data.columns = excel_data.columns.str.strip()

    required_columns = [
        "Disease", "Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4",
        "Favorable Conditions", "Suggestion for Yield 1", "Suggestion for Yield 2", "Suggestion for Yield 3",
        "Pesticide 1 (Content)", "Pesticide 1 (Products)", "Pesticide 2 (Content)", "Pesticide 2 (Products)",
        "Pesticide 3 (Content)", "Pesticide 3 (Products)", "Fertilizer 1 (Content)", "Fertilizer 1 (Products)",
        "Fertilizer 2 (Content)", "Fertilizer 2 (Products)"
    ]

    missing_columns = [col for col in required_columns if col not in excel_data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in Excel file: {missing_columns}")

    return excel_data[required_columns]

disease_df = load_excel_data()

def get_disease_data(disease_name):
    """Fetches disease-related information with separated products."""
    disease_info = disease_df[disease_df['Disease'] == disease_name]
    if disease_info.empty:
        return {}

    row = disease_info.iloc[0]
    return {
        'favorable_conditions': row.get('Favorable Conditions', 'No data available'),
        'precautions': [row[f'Precaution {i}'] for i in range(1, 5) if pd.notna(row.get(f'Precaution {i}'))],
        'suggestions': [row[f'Suggestion for Yield {i}'] for i in range(1, 4) if pd.notna(row.get(f'Suggestion for Yield {i}'))],
        'pesticides': {
            'contents': [row[f'Pesticide {i} (Content)'] for i in range(1, 4) if pd.notna(row.get(f'Pesticide {i} (Content)'))],
            'products': [row[f'Pesticide {i} (Products)'] for i in range(1, 4) if pd.notna(row.get(f'Pesticide {i} (Products)'))]
        },
        'fertilizers': {
            'contents': [row[f'Fertilizer {i} (Content)'] for i in range(1, 3) if pd.notna(row.get(f'Fertilizer {i} (Content)'))],
            'products': [row[f'Fertilizer {i} (Products)'] for i in range(1, 3) if pd.notna(row.get(f'Fertilizer {i} (Products)'))]
        }
    }

def extract_features_from_image(image):
    """Extracts GLCM and color histogram features."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, hist])

def load_training_data():
    """Loads dataset, extracts features, and prepares training data."""
    X, y = [], []
    for label, disease in enumerate(disease_df['Disease'].unique()):
        folder_path = os.path.join(DATASET_PATH, disease)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                continue

            features = extract_features_from_image(image)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def train_model():
    """Trains a RandomForest model using the extracted features."""
    X, y = load_training_data()
    if len(X) == 0:
        print("No training data found!")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved!")

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, feature extraction, and disease prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'})
    
    feature_vector = extract_features_from_image(image).reshape(1, -1)
    
    try:
        model = joblib.load(MODEL_PATH)
    except (FileNotFoundError, ValueError):
        return jsonify({'error': 'Model not found or invalid'})

    prediction = model.predict(feature_vector)[0]
    disease_name = disease_df.iloc[prediction]['Disease']
    
    disease_data = get_disease_data(disease_name)
    response = {
        'predicted_disease': disease_name,
        'favorable_conditions': disease_data.get('favorable_conditions', 'No data available'),
        'precautions': disease_data.get('precautions', []),
        'suggestions': disease_data.get('suggestions', []),
        'pesticides': disease_data.get('pesticides', {'contents': [], 'products': []}),
        'fertilizers': disease_data.get('fertilizers', {'contents': [], 'products': []})
    }
    return jsonify(response)

if __name__ == '__main__':
    train_model()
    app.run(debug=True, threaded=True)
