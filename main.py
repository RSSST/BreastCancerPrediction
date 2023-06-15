# import subprocess
# import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'joblib'])
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import xgboost as xgb

model = joblib.load('XGB_model.joblib')
st.title('Breast Cancer Prediction')
radius_mean = st.number_input('Radius mean:')
texture_mean = st.number_input('Texture mean:')
perimeter_mean = st.number_input('Perimeter mean:')
area_mean = st.number_input('Area_mean:')
smoothness_mean = st.number_input('Smoothness mean:')
compactness_mean = st.number_input('Compactness mean:')
concavity_mean = st.number_input('Concavity mean:')
concavepoint_mean = st.number_input('Concavepoint mean:')
symmetry_mean = st.number_input('Symmetry mean:')
fractal_dimension_mean = st.number_input('Fractal dimension mean:')

cols = ['radius_mean','texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

def predicedt():
    # row = np.array([radius_mean,texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavepoint_mean, symmetry_mean, fractal_dimension_mean])
    row = np.array([20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667])
    x = pd.DataFrame([row], columns=cols)
    prediction = model.predict(x)[0]

    if prediction == 0:
        st.success('Low risk')
    else:
        st.error('High risk')


st.button('Predict', on_click=predicted)
