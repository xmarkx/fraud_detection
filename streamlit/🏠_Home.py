# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:21:51 2023

@author: katar
"""

import streamlit as st
from PIL import Image


# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

# Title
st.title('Credit Card Fraud Detection')

# Content
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

c1.image(Image.open('images/visa.png'))
c2.image(Image.open('images/master.png'))
c3.image(Image.open('images/american.png'))
c4.image(Image.open('images/paypal.png'))
c5.image(Image.open('images/cirrus.png'))
c6.image(Image.open('images/maestro.png'))
c7.image(Image.open('images/western.png'))


st.write(
    """
        
    Credit card fraud is a serious issue that affects both financial institutions and the individuals - credit card users. 
    According to a [2022 study by the **Association of Certified Fraud Examiners**](https://legacy.acfe.com/report-to-the-nations/2022/), organizations lose an estimated
    5% of their annual revenues to fraud, with the financial services industry being the hardest hit. 
   

    
    """
)

st.subheader('Dataset')
st.write(
    """
    Our aim is to help financial institution to combat credit card frauds, by building an effective fraud detection 
    model which based on some sneaky patterns and anomalies can mark suspicious transaction so people from AML can have a look on them. 
    To build the model we will use  model using the Credit Card Fraud Detection dataset from [**Kaggle**]((https://www.kaggle.com/mlg-ulb/creditcardfraud)).
        
    """
    #To combat credit card fraud, financial institutions need to have effective fraud detection systems in place.
    ##One approach is to use machine learning algorithms to automatically detect fraudulent transactions based on
    #patterns and anomalies in the data. In this project, we built a credit card fraud detection model using the 
    #Credit Card Fraud Detection dataset from [**Kaggle**]((https://www.kaggle.com/mlg-ulb/creditcardfraud)).
    )
    

st.subheader('Challenge')
st.write(
    """
    Our model needs to focus in a first line in identifying frauds however we shouldn't underestimate the precision.
    No one likes the idea of losing money to fraudsters, but mistakenly flagging to many transaction as fraudalent can make some people unhappy too. 
    That could really upset our customers whose transactions were bloced, not to mention the AML Department employees who will need to work some extra hours.
    
    In this presentation, we'll show the steps we took to build our fraud detection model, using 
    the Credit Card Fraud Detection dataset. We will cover data preprocessing, exploratory data analysis, model 
    selection and training, and its evaluation.  
    """
    #Detecting fraudulent transactions is critical for financial institutions to protect their customers and minimize
    #financial losses. While it's essential to catch as many fraudulent transactions as possible, flagging too many legitimate 
    #transactions as fraudulent can lead to customer dissatisfaction and lost revenue. Therefore, building an accurate
    #fraud detection model that minimizes false positives is a significant challenge for financial institutions.
    
    #In this presentation, we will walk through the process of building a credit card fraud detection model using 
    #the Credit Card Fraud Detection dataset. We will cover data preprocessing, exploratory data analysis, model 
    #selection and training, and evaluation. Finally, we will discuss how this model can be used by the Anti-Money 
    #Laundering (AML) department.
    )
    
st.info('**Data Scientists: [@Kasia](https://www.linkedin.com/in/katarzyna-zbroinska-76301b21/), [@Mark](https://www.linkedin.com/in/mark-meszaros-ds/)**', icon="💡")
st.info('**GitHub: [@xmarkx](https://github.com/xmarkx/fraud_detection)**', icon="💻")
st.info('**Data: [Kaggle]((https://www.kaggle.com/mlg-ulb/creditcardfraud))**', icon="🗃")
