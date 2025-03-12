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
EXCEL_PATH = r"C:\Users\Rahul\OneDrive\Desktop\Book1.xlsx"
DATASET_PATH = r"D:\college_M_project\project\rice_leaf_diseases"  # Path to images dataset

# Load Excel dataset
def load_excel_data():
    """Loads disease data from Excel and extracts necessary columns."""
    excel_data = pd.read_excel(EXCEL_PATH, engine="openpyxl", header=0)

    # Remove any extra spaces from column names
    excel_data.columns = excel_data.columns.str.strip()

    # Ensure all required columns exist
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

    # Merge multiple columns into single string
    excel_data["Precautions"] = excel_data[
        ["Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4"]
    ].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

    excel_data["Suggestions"] = excel_data[
        ["Suggestion for Yield 1", "Suggestion for Yield 2", "Suggestion for Yield 3"]
    ].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

    excel_data["Pesticides"] = excel_data[
        ["Pesticide 1 (Content)", "Pesticide 1 (Products)",
         "Pesticide 2 (Content)", "Pesticide 2 (Products)",
         "Pesticide 3 (Content)", "Pesticide 3 (Products)"]
    ].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

    excel_data["Fertilizers"] = excel_data[
        ["Fertilizer 1 (Content)", "Fertilizer 1 (Products)",
         "Fertilizer 2 (Content)", "Fertilizer 2 (Products)"]
    ].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

    return excel_data[["Disease", "Favorable Conditions", "Precautions", "Suggestions", "Pesticides", "Fertilizers"]]

disease_df = load_excel_data()

# Extract disease information
def get_disease_data(disease_name):
    """Fetches disease-related information."""
    disease_info = disease_df[disease_df['Disease'] == disease_name]
    if disease_info.empty:
        return {}

    return {
        'favorable_conditions': disease_info.iloc[0].get('Favorable Conditions', 'No data available'),
        'precautions': disease_info.iloc[0].get('Precautions', 'No data available'),
        'suggestions': disease_info.iloc[0].get('Suggestions', 'No data available'),
        'pesticides': disease_info.iloc[0].get('Pesticides', 'No data available'),
        'fertilizers': disease_info.iloc[0].get('Fertilizers', 'No data available')
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

# Load images and extract features
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

# Train and save model
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
    
    # Load model
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
        'precautions': disease_data.get('precautions', 'No data available'),
        'suggestions': disease_data.get('suggestions', 'No data available'),
        'pesticides': disease_data.get('pesticides', 'No data available'),
        'fertilizers': disease_data.get('fertilizers', 'No data available')
    }
    return jsonify(response)

if __name__ == '__main__':
    train_model()  # Train the model before starting the app
    app.run(debug=True, threaded=True)
