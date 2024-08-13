import os 
import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler,OneHotEncoder 
from sklearn.linear_model import LogisticRegression 
import pickle 


st.title('Logistic Regression Assignment Project') 

model_path = os.path.join('artifacts','classifier.pkl')
preproccesor_path = os.path.join('artifacts','preproccessor.pkl') 

# Load trained classifiers & preproccesors 
try :
    model = pickle.load(open(model_path, 'rb')) 
    preproccesor = pickle.load(open(preproccesor_path,'rb')) 

except Exception as e:
    st.error("Error loading") 

# Pclass	Sex	Age	SibSp	Parch	Fare	Cabin	Embarked
Pclass = st.selectbox(label='Select Class :',options=[0,1])
sex = st.selectbox(options=['male','female'],label='Select Gender :') 
age = st.number_input('Enter Age :')
SibSp = st.selectbox(label='Select SibSp:',options=[0,1]) 
Parch = st.selectbox(label='Select Parch :',options=[0,1])
Fare = st.number_input('Enter Fare in Dollar :') 
Cabin = st.selectbox(label='Select Cabin :',options=[ 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6']) 
Embarked = st.selectbox(label='Select Embarked',options=['S', 'C', 'Q'])   

predict = 1 
if st.button(label='Predict'):
    if predict==1:
        st.success("Passager is Survived")
    else :
        st.error('Passager is Died')






