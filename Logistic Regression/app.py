import os 
import sys 
import pickle 
import pandas as pd 
import numpy as np  
import streamlit as st 
from sklearn.tree  import DecisionTreeClassifier  


model_path = os.path.join('artifacts','model.pkl')
model = pickle.load(open(model_path,'rb'))   

st.title("----Heart Disease Detector----")
st.markdown('----------------------------------------------------------------') 

# 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
col1, col2, col3 ,col4 = st.columns(4) 

with col1:
   Pclass = st.selectbox("Selected Class :")
   Sex = st.selectbox('Select Gender',('Male','Female')) 

with col2:
   Age = st.number_input('Enter Age :')
   SibSp = st.number_input('Enter SibSp :') 
   

with col3:
   Parch = st.number_input('Enter Parch :')
   Fare = st.number_input('Enter Fare') 

with col4: 
   Cabin = st.text_input('Enter Cabin :')
   Embarked = st.number_input('Enter Embarked :')   

if Sex == 'Female':
    Sex= 0
elif Sex == 'Male':
   Sex = 1 

if st.button(label='Predict'):
   data = pd.DataFrame(np.asarray([Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked]).reshape(1,-1),
   columns=['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked'])  
   result = model.predict(data)
   if result[0] == 0 :
      st.success('Passeger is live')
   elif result[0] == 1 : 
      st.success("Passeger don't live")
