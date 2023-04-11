# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:40:00 2023

@author: katar
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

#import 01_📝_Dataset
#01_📝_Dataset.load_data()


# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions

# ----------------------
# New function
#----------------------

@st.cache_data
def detect_outliers(data):
    outlier_info = {}
    for column in data.columns:
        if data[column].dtype != object:
            q1 = np.quantile(data[column], 0.25)
            q3 = np.quantile(data[column], 0.75)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            lower_bound = q1 - (1.5 * iqr)
            
            outliers = data[(data[column] > upper_bound) | (data[column] < lower_bound)][column]
            outlier_percentage = len(outliers) / len(data[column]) * 100
            outlier_info[column] = {"Outlier_percentage": outlier_percentage, "Total_count_of_outliers": len(outliers)}
                         
            outlier_dataframe = pd.DataFrame(outlier_info).T
                                                
    
    return outlier_dataframe.sort_values(by = 'Outlier_percentage', ascending = False)   


@st.cache_data
def plot_graphs():
    f, axes = plt.subplots(ncols=4, figsize=(20,8))
    colors = ["#0101DF", "#DF0101"]
        
    sns.boxplot(x="class", y="v27", data=raw_train, palette=colors, ax=axes[0])
    axes[0].set_title('v27 vs class')
    
    sns.boxplot(x="class", y="amount", data=raw_train, palette=colors, ax=axes[1])
    axes[1].set_title('amount vs class')
    
    sns.boxplot(x="class", y="v28", data=raw_train, palette=colors, ax=axes[2])
    axes[2].set_title('v28 vs class')
    
    sns.boxplot(x="class", y="v20", data=raw_train, palette=colors, ax=axes[3])
    axes[3].set_title('v20 vs class')
    
    return f

@st.cache_data
def plot_corr_matrix():
    features = [
    'time', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
    'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19',
    'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 'amount',
    'class'
    ]

    correlation_matrix = raw_train[features].corr(method="spearman") # pearson / spearman

    f, ax = plt.subplots(figsize=(20, 20))
    _ = sns.heatmap(
        correlation_matrix, 
        mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)), 
        cmap=sns.diverging_palette(230, 20, as_cmap=True), 
        center=0,
        square=True, 
        linewidths=.1, 
        cbar=False,
        ax=ax,
        annot=True,
    )
    _ = ax.set_title("Correlation Matrix", fontsize=15)
        
    return f


# this mean end of this part###################################################################################
# ---------------------------------------------------------------------------------

# ###############################
# Titel and subtitle of the page

st.markdown("# Credit Card Fraud Detection - EDA ❄️")
st.markdown(" Anonymized credit card transactions labeled as fraudulent or genuine")

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")

# ###################################################################################
# ---------------------------------------------------------------------------------


#####################################
# PYTHON CODE 




#here should be all the code which calculates sth

# end this part with all the variables/ models which you initalized above 
# and you can need in the next page 

#--------------------------------------------------
# Initializing all the datasets into SESSION_STATE
#--------------------------------------------------

#if "raw_data_un" not in st.session_state:
#    st.session_state["raw_data_un"] = raw_data_un


raw_train = st.session_state["raw_train"]


# ###################################################################################
# ---------------------------------------------------------------------------------


######################
# PRESENTATION


# --------------------
# The tab options
# --------------------


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Numeric Data", "Outliers", "Samplers", "Correlation matrix", "Amount", "Time"])

with tab1:
   st.header("Numeric Data")
   
   st.subheader("Missing values")
   # missing values 
      
#   st.write('Column Information:')
   df_info = pd.DataFrame({'Not_Null': raw_train.count(), 'Data_Type': raw_train.dtypes})
   df_info.index.name = 'Column_Name'
   st.write(df_info)
   
   st.subheader("Alternative missing values")
   st.write(f'Is there any transaction with amount 0? --> {any(raw_train.amount== 0)}  \n',
            f'How many transaction with 0 amounts is there? --> {len(raw_train[raw_train.amount==0])}  \n',
            f'Are all amounts bigger than or equal to 0? --> {all(raw_train.amount>= 0)}')
  #st.write(raw_train[raw_train['amount']==0].head())
   st.write(f'Is there any row with time 0? --> {any(raw_train.time== 0)}  \n',
            f'How many? --> {len(raw_train[raw_train.time==0])}')
   
   st.subheader("Analysis of numeric data")
   st.write(raw_train.describe().apply(lambda s: s.apply('{0:.5f}'.format)))
   
   
  # Define the checkbox to show/hide the dataset
   show_distributions = st.checkbox("Show features distributions")
   
   if show_distributions:
       st.image(Image.open('images/features_distributions.png'))
# Other way to get .info()  
#   buffer = io.StringIO()
#   raw_train.info(buf=buffer)
#   s = buffer.getvalue()
#  st.text(s)
  

   
    
with tab2:
   st.header("Outliers")
   st.write(detect_outliers(raw_train))
    # Further examples of outliers
   st.pyplot(plot_graphs())
   
   
with tab3:
   st.header("Samplers")
   #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
  
with tab4:
   st.header("Correlation matrix")
   #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   st.pyplot(plot_corr_matrix())
   
      
with tab5:
   st.header("Amount")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   
with tab6:
   st.header("Time")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   
   
#raw_data = st.session_state["raw_train"]
#st.write(raw_train.head())
#"st.session_state object: " , st.session_state