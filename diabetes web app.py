# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:04:31 2025

@author: sneha
"""

import os
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#new for deploying

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the correct file path
model_path = os.path.join(BASE_DIR, "trained_model.sav")

# Load the model
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

#loaded_model = pickle.load(open('trained_model.sav','rb'))
#creating a function for prediction
def diabetes_prediction(input_data):
    # Ensure input_data is a list or tuple
    if not isinstance(input_data, (list, tuple)):
        raise ValueError("Input data should be a list or tuple")

    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for model prediction (if necessary)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"
      
      
      
      
      
def main():
    #giving a title
    st.title('Diabetes Prediction Web App')
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    Pregnancies = float(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = float(Age)

    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        #user_input = [float(x) for x in user_input]

        #diab_prediction = diabetes_model.predict([user_input])

        #if diab_prediction[0] == 1:
           # diab_diagnosis = 'The person is diabetic'
       # else:
            #diab_diagnosis = 'The person is not diabetic'
        
    st.success(diagnosis)
    
#if __name__ == '__main__':
#    main()
