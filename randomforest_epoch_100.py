import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import graycomatrix, graycoprops

# Feature extraction function
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, hist])

# Load dataset
def load_dataset(dataset_dir):
    features, labels = [], []
    disease_names = []
    
    for label, disease in enumerate(os.listdir(dataset_dir)):
        disease_dir = os.path.join(dataset_dir, disease)
        disease_names.append(disease)
        
        for img_name in os.listdir(disease_dir):
            img_path = os.path.join(disease_dir, img_name)
            feature = extract_features(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels), disease_names

# Dataset directory
dataset_dir = r"D:\college_M_project\project\rice_leaf_diseases"
X, y, disease_names = load_dataset(dataset_dir)

# Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save trained model
joblib.dump(rf_model, 'rf_model.pkl')
print("Model saved as rf_model.pkl")

# Evaluate the model
y_pred = rf_model.predict(X_test)

# Calculate overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1-Score: {f1:.2f}")

# Load trained model for testing
rf_model = joblib.load('rf_model.pkl')

# Test with a single image
test_image_path = r"D:\college_M_project\project\rice_leaf_diseases\Leaf smut\DSC_0309.JPG"  # Change this to your image path
feature_vector = extract_features(test_image_path)

if feature_vector is not None:
    feature_vector = feature_vector.reshape(1, -1)  # Reshape for prediction
    prediction = rf_model.predict(feature_vector)
    print(f"Predicted Disease: {disease_names[prediction[0]]}")
else:
    print("Could not extract features from the image.")
