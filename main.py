import subprocess
import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'joblib'])
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'scikit-learn'])

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame, read_csv, Series

model = joblib.load('RF_model.joblib')
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

def predict():
    row = np.array([radius_mean,texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavepoint_mean, symmetry_mean, fractal_dimension_mean])
    x = pd.DataFrame([row], columns=cols)
    prediction = model.predict(x)[0]

    if prediction == 0:
        st.success('Low risk')
    else:
        st.error('High risk')


st.button('Predict', on_click=predict)

