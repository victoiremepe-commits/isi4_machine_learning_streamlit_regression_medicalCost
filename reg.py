import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

# chargement du modele
with open('reg.pkl', 'rb') as file:
    model = pickle.load(file)

# titre et mise en page
st.set_page_config(page_title="Predicteur de charges médicales")
st.title("Prediction des charges médicales")
st.markdown('remplis les informations ci dessous pour prédire les charges médicales')
# ajout des animations
with st.spinner('Chargement du modèle...'):
    time.sleep(2)  # Simuler un chargement

# Entrees utilisateur
col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 18, 100, 30)
with col2:
    sex = st.selectbox('Sexe', ['male', 'female'])

col3, col4 = st.columns(2)
with col3:
    bmi = st.number_input('BMI(Indice de masse corporelle)', 10.0, 50.0, 25.0)
with col4:
    children = st.slider('Nombre d\'enfants', 0, 5, 1)

col5, col6 = st.columns(2)
with col5:
    smoker = st.selectbox('Fumeur', ['yes', 'no'])
with col6:
    region = st.selectbox('Région', ['southwest', 'southeast', 'northwest', 'northeast'])


# Encodage des variables catégorielles
sex_encoder = 1 if sex == 'male' else 0
smoker_encoder = 1 if smoker == 'yes' else 0
region_dict = {"southwest": 0.24308153, "southeast":0.27225131, "northwest":0.24233358, "northeast":0.27225131}
region_encoded = region_dict[region]

# encodage du sexe
#le_sex = LabelEncoder()
#le_sex.fit(['male', 'female'])
#sex_encoder = le_sex.transform([sex])[0]

# encodage du fumeur
#le_smoker = LabelEncoder()
#le_smoker.fit(['yes', 'no'])
#smoker_encoder = le_smoker.transform([smoker])[0]

# encodage de la région
#le_region = LabelEncoder()
#le_region.fit(['southwest', 'southeast', 'northwest', 'northeast'])
#region_encoder = le_region.transform([region])[0]


# preparation des données pour la prédiction
input_data = [[age, sex_encoder, bmi, children, smoker_encoder, region_encoded]]

# prédiction
if st.button('Prédiction des charges médicales'):
    with st.spinner('Calcul en cours...'):
        prediction = model.predict(input_data)
        time.sleep(2)  # Simuler un délai de calcul
    st.success('Prédiction terminée !🎉')
    st.markdown(f"### Charges médicales prédites : {prediction[0]:.2f} $")
    st.balloons()










