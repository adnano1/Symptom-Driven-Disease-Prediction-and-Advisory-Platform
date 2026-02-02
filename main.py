from flask import Flask, request, render_template, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

#Load Database dataset

sym_des = pd.read_csv("Datasets/symtoms_df.csv")
precautions = pd.read_csv("Datasets/precautions_df.csv")
workout = pd.read_csv("Datasets/workout_df.csv")
description = pd.read_csv("Datasets/description.csv")
medications = pd.read_csv('Datasets/medications.csv')
diets = pd.read_csv("Datasets/diets.csv")


# load model===========================================
svc = pickle.load(open('model/svc.pkl','rb'))

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'stomach_pain': 7, 'muscle_wasting': 10, 'vomiting': 11, 'spotting_ urination': 13, 'patches_in_throat': 22, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'headache': 31, 'dark_urine': 33, 'back_pain': 37, 'constipation': 38, 'diarrhoea': 40, 'yellow_urine': 42, 'acute_liver_failure': 44, 'swelling_of_stomach': 46, 'chest_pain': 56, 'weakness_in_limbs': 57, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'extra_marital_contacts': 75, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'loss_of_balance': 85, 'weakness_of_one_body_side': 87, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'altered_sensorium': 98, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'polyuria': 105, 'lack_of_concentration': 109, 'distention_of_abdomen': 115, 'blood_in_sputum': 118, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    # If the model exposes feature names, use them to create the DataFrame
    if hasattr(svc, 'feature_names_in_'):
        feature_cols = list(svc.feature_names_in_)
    else:
        # Fallback: use symptom dictionary keys
        feature_cols = list(symptoms_dict.keys())

    # Create a single-row DataFrame initialized to 0 with exact feature columns
    X = pd.DataFrame(0, index=[0], columns=feature_cols)

    # For each reported symptom, try to match to a column name (several common variants)
    for s in patient_symptoms:
        if not s:
            continue
        candidates = [
            s,
            s.strip(),
            s.lower(),
            s.lower().replace(' ', '_'),
            s.replace(' ', '_'),
            s.replace('_', ' '),
        ]
        matched = False
        for c in candidates:
            if c in X.columns:
                X.at[0, c] = 1
                matched = True
                break
        if not matched:
            # Last resort: use numeric index from symptoms_dict (if it maps into feature length)
            if s in symptoms_dict:
                idx = symptoms_dict[s]
                if isinstance(idx, int) and idx < X.shape[1]:
                    X.iat[0, idx] = 1

    # Predict using the DataFrame (preserves feature names and avoids sklearn warnings)
    pred = svc.predict(X)[0]
    return diseases_list.get(pred, str(pred))

@app.route("/")
def index():
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        print(symptoms)
        if not symptoms or symptoms == ["Symptoms"]:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Please select at least one symptom'}), 400
            message = "Please select at least one symptom"
            symptoms_list = sorted(list(symptoms_dict.keys()))
            return render_template('index.html', message=message, available_symptoms=symptoms_list)
        else:
            user_symptoms = [s.strip() for s in symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            # Convert pandas Series/objects to list properly
            try:
                medications_list = [str(m).strip() for m in medications.values.tolist()] if hasattr(medications, 'values') else [str(m).strip() for m in medications]
                workout_list = [str(w).strip() for w in workout.values.tolist()] if hasattr(workout, 'values') else [str(w).strip() for w in workout]
                rec_diet_list = [str(d).strip() for d in rec_diet.values.tolist()] if hasattr(rec_diet, 'values') else [str(d).strip() for d in rec_diet]
            except Exception as e:
                print(f"Error converting data: {e}")
                medications_list = []
                workout_list = []
                rec_diet_list = []

            # Return JSON for AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'predicted_disease': str(predicted_disease),
                    'dis_des': str(dis_des),
                    'my_precautions': my_precautions,
                    'medications': medications_list,
                    'workout': workout_list,
                    'my_diet': rec_diet_list
                })

            symptoms_list = sorted(list(symptoms_dict.keys()))
            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout, available_symptoms=symptoms_list)

    symptoms_list = sorted(list(symptoms_dict.keys()))
    return render_template('index.html', available_symptoms=symptoms_list)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)