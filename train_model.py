import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Training Dataset
training_data = pd.read_csv("Datasets/Training.csv")

print("=" * 80)
print("MEDICAL RECOMMENDATION SYSTEM - MODEL TRAINING")
print("=" * 80)

# Display dataset info
print(f"\nDataset Shape: {training_data.shape}")
print(f"Columns: {list(training_data.columns)}")
print(f"\nFirst few rows:")
print(training_data.head())

# Define symptoms dictionary (must match main.py)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'stomach_pain': 7, 'muscle_wasting': 10, 'vomiting': 11, 'spotting_ urination': 13, 'patches_in_throat': 22, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'headache': 31, 'dark_urine': 33, 'back_pain': 37, 'constipation': 38, 'diarrhoea': 40, 'yellow_urine': 42, 'acute_liver_failure': 44, 'swelling_of_stomach': 46, 'chest_pain': 56, 'weakness_in_limbs': 57, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'extra_marital_contacts': 75, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'loss_of_balance': 85, 'weakness_of_one_body_side': 87, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'altered_sensorium': 98, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'polyuria': 105, 'lack_of_concentration': 109, 'distention_of_abdomen': 115, 'blood_in_sputum': 118, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

# Disease mapping (must match main.py)
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Inverse disease mapping for encoding
disease_to_index = {v: k for k, v in diseases_list.items()}

print(f"\nTotal Symptoms: {len(symptoms_dict)}")
print(f"Total Diseases: {len(diseases_list)}")

# Prepare Features (X) and Labels (y)
print("\n" + "=" * 80)
print("PREPROCESSING DATA")
print("=" * 80)

X = []
y = []

# Get disease column name (usually the last column)
disease_column = training_data.columns[-1]
print(f"Disease column identified as: '{disease_column}'")

# Process each row
for idx, row in training_data.iterrows():
    # Create feature vector
    feature_vector = np.zeros(len(symptoms_dict))
    
    # Process each symptom column (all except last column which is disease)
    for symptom_col in training_data.columns[:-1]:
        symptom_name = symptom_col.lower().strip()
        
        # Map symptom column name to symptoms_dict key
        if symptom_name in symptoms_dict:
            if row[symptom_col] == 1:  # If symptom is present
                feature_vector[symptoms_dict[symptom_name]] = 1
    
    X.append(feature_vector)
    
    # Get disease label
    disease = row[disease_column]
    if disease in disease_to_index:
        y.append(disease_to_index[disease])
    else:
        print(f"Warning: Disease '{disease}' not found in disease mapping")

X = np.array(X)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")
print(f"Unique diseases in training data: {len(np.unique(y))}")

# Split data into 70% training and 30% testing
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (70% - 30%)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples (70%)")
print(f"Testing set size: {X_test.shape[0]} samples (30%)")
print(f"Training labels distribution: {np.bincount(y_train)}")
print(f"Testing labels distribution: {np.bincount(y_test)}")

# Train SVC Model
print("\n" + "=" * 80)
print("TRAINING SVC MODEL")
print("=" * 80)

svc = SVC(kernel='linear', C=1.0, random_state=42)
print("Training SVC with linear kernel...")
svc.fit(X_train, y_train)
print("[OK] Model training completed!")

# Make predictions on test set
print("\nGenerating predictions on test set...")
y_pred = svc.predict(X_test)
print("[OK] Predictions completed!")

# Model Evaluation
print("\n" + "=" * 80)
print("MODEL EVALUATION METRICS")
print("=" * 80)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# F1 Score (weighted for multi-class)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (Weighted): {f1:.4f}")

# Classification Report
print("\n" + "-" * 80)
print("CLASSIFICATION REPORT")
print("-" * 80)
class_report = classification_report(y_test, y_pred, target_names=[diseases_list[i] for i in sorted(diseases_list.keys())], zero_division=0)
print(class_report)

# Save the trained model
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

# Create model directory if it doesn't exist
# if not os.path.exists('model'):
#     os.makedirs('model')
#     print("[OK] Created 'model' directory")

# model_path = 'model/svc.pkl'
# with open(model_path, 'wb') as f:
#     pickle.dump(svc, f)

# print(f"[OK] Model successfully saved to '{model_path}'")

# Summary Report
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"Total Training Samples: {X_train.shape[0]}")
print(f"Total Testing Samples: {X_test.shape[0]}")
print(f"Number of Features: {X_train.shape[1]}")
print(f"Number of Classes: {len(np.unique(y))}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")
print(f"Model saved at: model/svc.pkl")
print("=" * 80)