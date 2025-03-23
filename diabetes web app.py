# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:04:31 2025

@author: sneha
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#new for deploying
working_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the correct file path
model_path = os.path.join(BASE_DIR, "trained_model.sav")

# Load the model
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

#loaded_model = pickle.load(open('trained_model.sav','rb'))
#creating a function for prediction
def diabetes_prediction(input_data):
    #input_data = (4,110,92,0,0,37.6,0.191,30)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'
      
      
      
      
      
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
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
