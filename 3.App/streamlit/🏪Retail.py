#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from PIL import Image
image = Image.open('C:/Users/Tomc/Cover_820x312.jpg')
st.image(image)

#st.set_page_config(
#    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
#    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
#    page_title=None,  # String or None. Strings get appended with "• Streamlit". 
#    page_icon=None,  # String, anything supported by st.image, or None.
#)

st.write("""
#  eBike Competitor Analysis
🔎This page updates products on the market and their prices to assist with product pricing and also provides brand feature comparisons.
""")



st.write('---')

###########################################################################################################
df1 = pd.read_csv("C:/Users/Tomc/retails.csv")

#data manipulation...

#Drop the row with no price updated(ex: product discontinued)
df1=df1.dropna(subset = ['Price'])
#st.write(len(df1))
#Fill missing value with median
for col in df1.columns[3:-1]:
    df1[col].fillna(df1[col].median(), inplace=True)
   
###########################################################################################################
#st.header('I.Brand Summary')
#st.write(df1.groupby(df1.Brand).mean().sort_values(by='Price',ascending=False))



st.header('I.Price Range💰')

brands=df1.Brand.unique().tolist()
    
st.sidebar.header('Choose Brands:')    
options = st.sidebar.multiselect(
     'Brands',
     brands,
     brands)

st.sidebar.header("Choose Specs' ranges:")    
      
battery_WH = st.sidebar.slider('battery WH', float(df1['battery WH'].min()), float(df1['battery WH'].max()),
                               (float(df1['battery WH'].min()), float(df1['battery WH'].max())),step=1.0)
                               #(float(df1['battery WH'].min())+1/4*(float(df1['battery WH'].max())-float(df1['battery WH'].min())),
                               #float(df1['battery WH'].max())-1/4*(float(df1['battery WH'].max())-float(df1['battery WH'].min()))
                               #),step=1.0)

motor_watts = st.sidebar.slider('motor watts', float(df1['motor watts'].min()), float(df1['motor watts'].max()), 
                                (float(df1['motor watts'].min()), float(df1['motor watts'].max())
                               #(float(df1['motor watts'].min())+1/4*(float(df1['motor watts'].max())-float(df1['motor watts'].min())),
                                #float(df1['motor watts'].max())-1/4*(float(df1['motor watts'].max())-float(df1['motor watts'].min()))
                               ),step=1.0)

WHEEL_SIZE = st.sidebar.slider('WHEEL SIZE', float(df1['WHEEL SIZE'].min()), float(df1['WHEEL SIZE'].max()), 
                               (float(df1['WHEEL SIZE'].min()), float(df1['WHEEL SIZE'].max())
                                #(float(df1['WHEEL SIZE'].min())+1/4*(float(df1['WHEEL SIZE'].max())-float(df1['WHEEL SIZE'].min())),
                                #float(df1['WHEEL SIZE'].max())-1/4*(float(df1['WHEEL SIZE'].max())-float(df1['WHEEL SIZE'].min()))
                               ),step=1.0)

Fat_Tire = st.sidebar.slider('Fat Tire', float(df1['Fat Tire'].min()), float(df1['Fat Tire'].max()), 
                                (float(df1['Fat Tire'].min()),float(df1['Fat Tire'].max())),step=1.0)
                                
Foldable = st.sidebar.slider('Foldable', float(df1['Foldable'].min()), float(df1['Foldable'].max()), 
                                (float(df1['Foldable'].min()),float(df1['Foldable'].max())),step=1.0)                            

SPEED_GEAR = st.sidebar.slider('#SPEED GEAR', float(df1['SPEED GEAR'].min()), float(df1['SPEED GEAR'].max()), 
                               (float(df1['SPEED GEAR'].min()), float(df1['SPEED GEAR'].max())
                                #(float(df1['SPEED GEAR'].min())+1/4*(float(df1['SPEED GEAR'].max())-float(df1['SPEED GEAR'].min())),
                                #float(df1['SPEED GEAR'].max())-1/4*(float(df1['SPEED GEAR'].max())-float(df1['SPEED GEAR'].min()))
                               ),step=1.0)

external = st.sidebar.slider('External Battery', float(df1['external'].min()), float(df1['external'].max()), 
                                (float(df1['external'].min()),float(df1['external'].max())),step=1.0)       

disc_breaks = st.sidebar.slider('disc brakes', float(df1['disc breaks'].min()), float(df1['disc breaks'].max()), 
                                (float(df1['disc breaks'].min()),float(df1['disc breaks'].max())),step=1.0)     

Absorbing_fork = st.sidebar.slider('Absorbing fork', float(df1['Absorbing fork'].min()), float(df1['Absorbing fork'].max()), 
                                (float(df1['Absorbing fork'].min()),float(df1['Absorbing fork'].max())),step=1.0)                            

Weight = st.sidebar.slider('Weight', float(df1['Weight'].min()), float(df1['Weight'].max()), 
                           (float(df1['Weight'].min()), float(df1['Weight'].max())
                                #(float(df1['Weight'].min())+1/4*(float(df1['Weight'].max())-float(df1['Weight'].min())),
                                #float(df1['Weight'].max())-1/4*(float(df1['Weight'].max())-float(df1['Weight'].min()))
                               ),step=1.0)

#df1[(df1['Brand'].isin(options))&(df1['WHEEL SIZE'].isin(WHEEL_SIZE))]

#df1[(df1['Brand'].isin(options))]



df2=df1[(df1['Brand'].isin(options))][df1['WHEEL SIZE'].between(WHEEL_SIZE[0],WHEEL_SIZE[1])][df1['motor watts'].between(motor_watts[0],motor_watts[1])][df1['battery WH'].between(battery_WH[0],battery_WH[1])][df1['Fat Tire'].between(Fat_Tire[0],Fat_Tire[1])][df1['Foldable'].between(Foldable[0],Foldable[1])][df1['disc breaks'].between(disc_breaks[0],disc_breaks[1])][df1['Absorbing fork'].between(Absorbing_fork[0],Absorbing_fork[1])][df1['SPEED GEAR'].between(SPEED_GEAR[0],SPEED_GEAR[1])][df1['external'].between(external[0],external[1])][df1.Weight.between(Weight[0],Weight[1])]

st.write('please use sidebar to select brands and spec range')
import seaborn as sns
sns.set()
x = pd.Series(df2.Price, name='Price')

fig1, ax = plt.subplots()
ax=sns.distplot(x)
opt = st.selectbox('Select a bike',('20EB-FOLD', '20EB-FOLDST', '20EB-FD300','320-STFD','26EB-MTN','26EB-CITY','26EB-CITY-500W'))
st.write('Voltour:', opt)
vollist={'20EB-FOLD':1599,'20EB-FOLDST':1999,'20EB-FD300':899,'320-STFD':1399,'26EB-MTN':2499,'26EB-CITY':1799,'26EB-CITY-500W':1999}
p=vollist[opt]
plt.axvline(p,0,1, color='r')
plt.text(p+70,0,'%s:$%s'%(opt,p),color='r',rotation=90)

st.pyplot(fig1)

#sns.boxplot(x)
#st.pyplot()

st.table(x.describe())

st.write('Data:')
df2
towrite = io.BytesIO()
downloaded_file = df2.to_excel(towrite, encoding='utf-8', index=False, header=True)
towrite.seek(0)  # reset pointer
b64 = base64.b64encode(towrite.read()).decode()  # some strings
linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="retail_bikes.xlsx">Download excel file</a>'
st.markdown(linko, unsafe_allow_html=True)
st.write('---')
###########################################################################

from matplotlib import pyplot
st.header('II.Competitor Comparison⚔️')
st.subheader('1.Univariate')

#options3 = st.multiselect(
     #'Brands',
     #brands,
     #brands)
fig, ax = plt.subplots()
opt = st.selectbox('Select a feature',('battery WH', 'motor watts', 'WHEEL SIZE','Foldable','Range','Price'))
st.write('You selected:', opt)

st.write('Please also use sidebar to filter the brands for comparison')
#bins = numpy.linspace(float(df1['Price'].min()), float(df1['Price'].max()))
for i in range(len(options)):
    df1[df1.Brand==options[i]].Price.tolist()
    ax=pyplot.hist(df1[df1.Brand==options[i]][opt].tolist(), alpha=0.5, label=options[i])
pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
st.pyplot(fig)    
###############################################################################################
st.subheader('2.Multivariate')
import plotly.express as px
import pandas as pd

opt1 = st.selectbox('Select the 1st feature',('battery WH','motor watts', 'WHEEL SIZE','Foldable','Range','Price','Weight','SPEED GEAR','Charge time'),key="0")

opt2 = st.selectbox('Select the 2nd feature',('battery WH', 'motor watts', 'WHEEL SIZE','Foldable','Range','Price','Weight','SPEED GEAR','Charge time'),key = "1")

opt3 = st.selectbox('Select the 3rd feature',('battery WH', 'motor watts', 'WHEEL SIZE','Foldable','Range','Price','Weight','SPEED GEAR','Charge time'),key = "2")
st.write('Please also use sidebar to filter the brands for comparison, you may rotate and zoom the plot and hover the mouse to a specific point of product for further detail!')   
    

fig = px.scatter_3d(df2, x=opt1, y=opt2, z=opt3,
              color='Brand')
st.plotly_chart(fig)


