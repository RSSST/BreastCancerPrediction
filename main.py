# import subprocess
# import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'joblib'])
# subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])
subprocess.check_call([sys.executable, "-m", "pip", "install", 'sklearn'])

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

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

def predicted():
    # row = np.array([radius_mean,texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concavepoint_mean, symmetry_mean, fractal_dimension_mean])
    row = np.array([20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667])
    g = pd.DataFrame([row], columns=cols)
    prediction = model.predict(g)[0]

    if prediction == 0:
        st.success('Low risk')
    else:
        st.error('High risk')


st.button('Predict', on_click=predicted)
