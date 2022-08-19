#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn
import random
import xgboost as xgb
import numpy as np

import io
import base64

#st.set_page_config(
    #layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    #initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    #page_title=None,  # String or None. Strings get appended with "• Streamlit". 
    #page_icon=None,  # String, anything supported by st.image, or None.
#)
from PIL import Image
image = Image.open('C:/Users/Tomc/Cover_820x312.jpg')
st.image(image)
st.write("""
# eBike Competitor Analysis
🔎This page calculates recent pricing recommendations by different ML algorithms on amazon popular ebikes, and provides feature trends and competitor screening to assist product development.

""")
st.write('---')

# Loads the Boston House Price Dataset
df1 = pd.read_csv("C:/Users/Tomc/Downloads/bike info 6.csv")


#X = df1[['battery WH', 'motor watts','Throttle','Charge time','Weight']]
X = df1[['battery WH', 'motor watts','Rating','Absorbing fork','WHEEL SIZE','Foldable','SPEED GEAR','Fat Tire','external','top speed','Throttle','Charge time','Weight']]
#X = df1[['battery WH', 'motor watts','WHEEL SIZE','top speed','Throttle','Charge time','Weight','Foldable']]     
Y = df1[["Price"]]

# Sidebar
# Header of Specify Input Parameters

#Select a prediction method

make_choice = st.sidebar.selectbox('Select an ML Model:',['fine-tuned XGBoost','XGboost','RandomForest'])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    battery_WH = st.sidebar.slider('battery WH', float(X['battery WH'].min()), float(X['battery WH'].max()), float(X['battery WH'].mean()),step=1.0)
    motor_watts = st.sidebar.slider('motor watts', float(X['motor watts'].min()), float(X['motor watts'].max()), float(X['motor watts'].mean()),step=1.0)
    WHEEL_SIZE = st.sidebar.slider('WHEEL SIZE', float(X['WHEEL SIZE'].min()), float(X['WHEEL SIZE'].max()), float(X['WHEEL SIZE'].mean()),step=0.5)
    Fat_Tire = st.sidebar.slider('Fat Tire', float(X['Fat Tire'].min()), float(X['Fat Tire'].max()), float(X['Fat Tire'].max()),step=1.0)
    SPEED_GEAR = st.sidebar.slider('SPEED GEAR', float(X['SPEED GEAR'].min()), float(X['SPEED GEAR'].max()), float(X['SPEED GEAR'].mean()))
    Foldable = st.sidebar.slider('Foldable', float(X['Foldable'].min(),), float(X['Foldable'].max()),float(X['Foldable'].max()),step=1.0)
    external = st.sidebar.slider('external', float(X['external'].min()),float(X['external'].max()),float(X['external'].max()),step=1.0)
    top_speed = st.sidebar.slider('top speed', float(X['top speed'].min()), float(X['top speed'].max()), float(X['top speed'].mean()),step=0.5)
    Throttle = st.sidebar.slider('Throttle mode range', float(X['Throttle'].min()), float(X['Throttle'].max()), float(X['Throttle'].mean()),step=0.5)
    #disc_breaks = st.sidebar.slider('disc breaks', float(X['disc breaks'].min()), float(X['disc breaks'].max()), float(X['disc breaks'].mean()),step=0.5)
    Absorbing_fork = st.sidebar.slider('Absorbing fork', float(X['Absorbing fork'].min()), float(X['Absorbing fork'].max()), float(X['Absorbing fork'].mean()),step=0.5)
    Charge_time= st.sidebar.slider('Charge time', float(X['Charge time'].min()), float(X['Charge time'].max()), float(X['Charge time'].mean()),step=0.5)
    Weight = st.sidebar.slider('Weight', float(X['Weight'].min()), float(X['Weight'].max()), float(X['Weight'].mean()))
    #Days = st.sidebar.slider('Days', float(X['Days'].min()), float(X['Days'].max()), float(X['Days'].mean()),step=1.0)
    Rating = st.sidebar.slider('Rating', float(X['Rating'].min()), float(X['Rating'].max()), float(X['Rating'].mean()),step=0.1)
    
    
    data = {'battery WH': battery_WH,
            'motor watts': motor_watts,
            'WHEEL SIZE': WHEEL_SIZE,
            
            'Fat Tire': Fat_Tire,
            'SPEED GEAR': SPEED_GEAR,
            'Foldable': Foldable,
            'external': external,
            'top speed': top_speed,
            'Throttle' : Throttle,
            #'disc_breaks':disc_breaks,
            'Absorbing fork': Absorbing_fork,
            'Charge time': Charge_time,
            'Weight':Weight,
            #'Days':Days,
            'Rating':Rating
            
            
           }
            
          
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

import pickle
filename = 'finalized_model1.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


#^v^v^v^v^ Main Panel
# Build Regression Model
if make_choice=='XGboost':
    model = xgb.XGBRegressor(objective ='reg:squarederror',max_depth = 5,eta=0.31,n_estimators = 200,alpha = 1)
elif make_choice=='RandomForest':
    model=RandomForestRegressor(random_state=123)
else: 
    model=loaded_model

model.fit(X, Y.values.ravel())
# Apply Model to Make Prediction
prediction = model.predict(df)


#print('Accuracy for Random Forest',100*max(0,rmse)) 


st.header('I.Prediction of Price💰')
#st.write("""

#This app predicts the price from amazon list
#""")
st.metric(label="Predicted Price", value=prediction)


# Print specified input parameters
st.caption('Specified Input parameters')

st.dataframe(df,width=1000, height=88 )


st.write('---')



# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)



st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('II.Feature Importance⚙️')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')


st.write('---')

st.header('III.Feature Trend📈 ')

option = st.selectbox('Select a feature',('battery WH', 'motor watts', 'WHEEL SIZE','Foldable','top speed','Throttle'))
st.write('You selected:', option)
import seaborn as sns
x = pd.Series(df1[option], name=option)
ax = sns.distplot(x)

st.pyplot()
st.write('---')

st.header('IV.Feature Relationships🧬')
st.caption('the correlation coefficients between all variables, the lighter the color, the stronger the relationship!')
sns.heatmap(df1.corr())
st.pyplot()
st.write('---')

st.header('V.Competitor overview⚔️')
st.write(df1.groupby(df1.Brand).mean().sort_values(by='Total Reviews',ascending=False))
#df_gb = df1.groupby(['Brand'])
#st.bar_chart(df_gb['Brand'])

brands=df1.Brand.unique().tolist()
Name = st.selectbox('select a Competitor',[i for i in brands])
st.write('You selected:', Name)
st.write(df1[df1.Brand==Name])

towrite = io.BytesIO()
downloaded_file = df1[df1.Brand==Name].to_excel(towrite, encoding='utf-8', index=False, header=True)
towrite.seek(0)  # reset pointer
b64 = base64.b64encode(towrite.read()).decode()  # some strings
linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="amazon_bikes.xlsx">Download excel file</a>'
st.markdown(linko, unsafe_allow_html=True)




