import os 
import sys 
import pickle 
import pandas as pd 
import numpy as np  
import streamlit as st 
from sklearn.tree  import DecisionTreeClassifier  


model_path = os.path.join('artifacts','logistic.pkl')
procesor_path = os.path.join('artifacts','preproccessor.pkl')
model = pickle.load(open(model_path,'rb'))   
procesor = pickle.load(open(procesor_path,'rb')) 

st.title("---- Logistic Regression ----")
st.markdown('----------------------------------------------------------------') 

# 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
col1, col2, col3 ,col4 = st.columns(4) 

with col1:
   Pclass = st.selectbox("Selected Class :",options=[1,2,3]) 
   Sex = st.selectbox('Select Gender',options=['male','female']) 

with col2:
   Age = st.number_input('Enter Age :')
   SibSp = st.selectbox('select SibSp :',options=[0,1,2,3,4,5,8]) 
   

with col3:
   Parch = st.selectbox('Enter Parch :',options=[1,2,3,4,5,6])
   Fare = st.number_input('Enter Fare') 

with col4: 
   Cabin = st.selectbox('Enter Cabin :',options=['B96 B98', 'E10', 'F2', 'F4', 'B57 B59 B63 B66', 'C78', 'C68',
       'D26', 'E40', 'B20', 'B94', 'C62 C64', 'E34', 'C85', 'D', 'D35',
       'E49', 'D48', 'C65', 'D20', 'B35', 'D10 D12', 'B78', 'B28', 'C83',
       'C125', 'G6', 'D36', 'C23 C25 C27', 'B50', 'F33', 'A34', 'C30',
       'E38', 'B19', 'A20', 'C103', 'D11', 'C106', 'B49', 'E58', 'C2',
       'E36', 'A6', 'D17', 'B51 B53 B55', 'C91', 'E33', 'C124', 'D46',
       'E50', 'D45', 'C148', 'B18', 'E67', 'E68', 'E25', 'C126', 'E17',
       'E101', 'D19', 'A23', 'A26', 'F38', 'D47', 'D15', 'B79', 'A36',
       'A24', 'C123', 'B58 B60', 'B77', 'D50', 'D33', 'C128', 'C47',
       'C52', 'C22 C26', 'B41', 'C82', 'B73', 'C49', 'F E69', 'D21',
       'C86', 'A14', 'B82 B84', 'C110', 'A10', 'B4', 'B5', 'E44', 'A32',
       'C87', 'B71', 'F G73', 'C95', 'E8', 'C32', 'F G63', 'D6', 'C50',
       'C111', 'B22', 'A19', 'E31', 'C93', 'C99', 'C46', 'B30', 'D37',
       'B69', 'A7', 'A16', 'B37', 'C101', 'B38', 'C92', 'A5', 'C90', 'T',
       'B86', 'E77', 'E63'])
   Embarked = st.selectbox('Select Embarked :',options=['S','C','Q'])   


if st.button(label='Predict'):
   data = pd.DataFrame({'Pclass':[Pclass],'Sex':[Sex], 'Age':[Age], 'SibSp':[SibSp],'Parch':[Parch], 'Fare':[Fare], 'Cabin':[Cabin], 'Embarked':[Embarked]})  
   data = procesor.transform(data)
   result = model.predict(data)
   if result[0] == 0 :
      st.success('Passeger is live')
   elif result[0] == 1 : 
      st.success("Passeger don't live")
