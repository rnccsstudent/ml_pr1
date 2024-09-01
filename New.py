import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

# Define symptoms and diseases
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
           'Dengue', 'Typhoid', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemorrhoids (piles)',
           'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthritis',
           'Arthritis', 'Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Load and preprocess training data
df = pd.read_csv("C:/Users/Pranab Mahata/Desktop/ml_pr1/Training.csv")

# Identify columns missing from the DataFrame
missing_cols = [col for col in l1 if col not in df.columns]
if missing_cols:
    st.warning(f"The following columns are missing from the DataFrame: {missing_cols}")

# Filter out only the columns that are present in the DataFrame
l1 = [col for col in l1 if col in df.columns]

# Prepare training data
X = df[l1]
df.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
y = np.ravel(df[['prognosis']])

# Load and preprocess testing data
tr = pd.read_csv("C:/Users/Pranab Mahata/Desktop/ml_pr1/Testing.csv")
tr.replace({'prognosis': {d: i for i, d in enumerate(disease)}}, inplace=True)
X_test = tr[l1]
y_test = np.ravel(tr[['prognosis']])

# Define the DecisionTree function
def DecisionTree(symptoms):
    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(X, y)

    # Calculate accuracy (optional)
    y_pred = clf3.predict(X_test)
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Update symptom list
    l2 = [1 if symptom in symptoms else 0 for symptom in l1]

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    result = "Not Found"
    if predicted < len(disease):
        result = disease[predicted]

    return result

# Streamlit UI
st.title("Disease Prediction System")

# Symptom selection
symptoms = []
for i in range(5):
    symptom = st.selectbox(f"Select Symptom {i+1}", options=[""] + l1, key=f"symptom{i}")
    if symptom:
        symptoms.append(symptom)

if st.button("Analyse"):
    result = DecisionTree(symptoms)
    st.write(f"Predicted Disease: {result}")

