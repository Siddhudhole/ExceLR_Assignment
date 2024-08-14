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
Sex = st.selectbox(options=['male','female'],label='Select Gender :') 
Age = st.number_input('Enter Age :')
SibSp = st.selectbox(label='Select SibSp:',options=[0,1]) 
Parch = st.selectbox(label='Select Parch :',options=[0,1])
Fare = st.number_input('Enter Fare in Dollar :') 
Embarked = st.selectbox(label='Select Embarked',options=['S', 'C', 'Q'])  
Cabin = st.selectbox(label='Select Cabin :',options=['C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
       'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
       'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
       'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
       'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
       'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
       'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
       'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
       'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44',
       'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14',
       'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38',
       'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68',
       'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48',
       'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
       'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
       'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
       'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
       'C148']) 

try :
    data = pd.DataFrame({'Pclass':[Pclass],'Sex':[Sex],'Age':[Age],'SibSp':[SibSp],'Parch':[Parch],'Fare':[Fare],'Cabin':[Cabin],'Embarked':[Embarked]})
    data = preproccesor.transform(data) 
    predict = model.predict(data)  
    

except Exception as e :
    st.error(str(e)) 

if st.button(label='Predict'):
    if predict==1:
        st.success("Passager is Survived")
    else :
        st.error('Passager is Died')






