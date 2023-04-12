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
from sklearn.base import BaseEstimator, TransformerMixin

#import 01_📝_Dataset
#01_📝_Dataset.load_data()


# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions

# ------------------------------------
# Function showing info about outliers
#------------------------------------

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

# ------------------------------------
# Function to plot outliers graphs
#------------------------------------

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


# ------------------------------------
# Function ploting correlation matrix
#------------------------------------
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

# ------------------------------------
# Histgram and violin plot showing amount distribution
#------------------------------------
@st.cache_data
def plot_amount_distribution(raw_train):
    fig, axes = plt.subplots(1, 2, figsize=(12,6), layout='constrained')
    cols = [0, -2]

    p1 = sns.histplot(data=raw_train, x='amount', bins=100, ax=axes[0])
    p1.axes.set_title('Amount distribution',fontsize=15)

    p2 = sns.violinplot(data=raw_train, y=raw_train['amount'], x='class', ax=axes[1], color='red', cut=0, inner=None)
    boxprops = dict(color='r', alpha=0.5)
    flierprops = dict(marker='o', markeredgecolor='r', markersize=0.5, alpha=0.2)
    sns.boxplot(data = raw_train, y=raw_train['amount'], x='class', ax=axes[1], color='red', width=0.05, fliersize=0.5,\
                linewidth=1, whis=1.5, boxprops=boxprops, flierprops=flierprops)
    p2.axes.set_title('Amount distribution per Class',fontsize=15)
    p2.tick_params(labelsize=10)
    plt.setp(axes[1].collections, alpha=.5)
    
    st.pyplot(fig)
    
 # ------------------------------------
 # Log amount distribution 
 #------------------------------------   
@st.cache_data
def plot_amount_log():
    # Plot the distribution of the transformed amount
    fig, axes = plt.subplots(figsize=(6,6), layout='constrained')
    p1 = sns.histplot(data=raw_train_log, x='amount_log', bins=100, ax=axes)
    p1.axes.set_title('Amount_log distribution',fontsize=15)
    st.pyplot(fig)
    
    #return fig

# ------------------------------------
# Time distribution 
#------------------------------------  
@st.cache_data
def plot_time(data):
    fig, ax = plt.subplots(figsize=(25, 7), layout='constrained')
    p1 = sns.histplot(data['time'], bins = 48)
    p1.axes.set_title('Distribution of "Time" feature', fontsize=25)
    st.pyplot(fig)

####################################################################################
# ---------------------------------------------------------------------------------

# ###############################
# Titel and subtitle of the page

st.markdown("# Explanatory Data Analysis")
st.markdown("#### Our own personal treasure hunt for insights hidden in the data. ")


# ###################################################################################
# ---------------------------------------------------------------------------------


#####################################
# PYTHON CODE - DATA RELATED

#--------------------------------------------------
# Initializing all the datasets into/ from SESSION_STATE
#--------------------------------------------------


raw_train = st.session_state["raw_train"]


# ###################################################################################
# ---------------------------------------------------------------------------------


######################
# PRESENTATION


# --------------------
# The tab options
# --------------------



tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Numeric Data", "Outliers", "Samplers", "Correlation matrix", "Amount", "Time", "Features selection", "Summary of the EDA"])


with tab1:
   st.header("Numeric Data")
   
   with st.expander("Data type & missing values"):
       st.write("""On the first sight all the variables looks like numerical as they are of 𝑓𝑙𝑜𝑎𝑡 type.
            The target variable is a nominal categorical variable that takes the value 1 for frauds and 0 for non-frauds.
            """)
       st.write("""The dataset has no missing values.""")   
        
       df_info = pd.DataFrame({'Not_Null': raw_train.count(), 'Data_Type': raw_train.dtypes})
       df_info.index.name = 'Column_Name'
       st.write(df_info)
       
   with st.expander("Alternative missing values"):
       st.write('It is important to check for 0 values in the dataset as they can indicate missing data or errors in data collection.  \n',
                '* There is no point in analysing 0 values for v1, v2 ...v28 features as they are the result of principal component analisis (PCA) and there is no clear interpretation of them.  \n',
                f'* There are {len(raw_train[raw_train.amount==0])} transactions with 0 amount. "They can indicate transactions where the credit card was registered for future payments, but there was no actual money flow.  \n',
                '* The two transactions with Time of 0, represents the transactions registered in the same seconds as the registration started.'
                )
            
   with st.expander("Analysis of data distributions"):
       st.write(raw_train.describe().apply(lambda s: s.apply('{0:.5f}'.format)))
   
   
      # Define the checkbox 
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
   
   col1, col2 = st.columns([1, 2])
   with col1:
       st.write(detect_outliers(raw_train))
    
   with col2:
        st.pyplot(plot_graphs())
   
   
with tab3:
   st.header("Samplers")
   
   st.write("""
        For resampling strategy we used RandomOverSampler and SMOTENC. Both are resampling strategies found in the imbalanced-learn library. Both of these strategies\
        use oversampling, a strategy where we generate datapoints which will represent the minority class.\n
        We can see the difference between the two strategies below:""")
   st.image('images/ros_smotenc_zoomed.png')
   
   with st.expander("RandomOverSampler"):
       st.write("""
               It is the most naive strategy which creates new, synthetic samples 
               by randomly sampling with replacement and duplicating the current 
               available samples of the minority class.               
                """)
   with st.expander("SMOTE - Synthetic Minority Over-sampling Technique"):
        st.write("""
                It generates new smaples by interpolation by generating new samples
                next to the original samples using a k-Nearest Neighbors classifier.
                
                """)
  
with tab4:
   st.header("Correlation matrix")
      
   col1, col2 = st.columns([2, 1])
   with col1:
      fig = plot_corr_matrix()
      st.pyplot(fig)
   with col2:
      pass
   
   with st.expander("The Spearman correlation"):
        st.write("""
                The Spearman correlation between two variables is equal to the Pearson
                correlation between the rank values of those two variables; while Pearson's 
                correlation assesses linear relationships, Spearman's correlation assesses monotonic
                relationships (including linear relationships but also exponential or logarithmic). If there are no repeated data values, a perfect 
                Spearman correlation of +1 or −1 occurs when each of the variables is a perfect 
                monotone function of the other.
                
                """)
                
   with st.expander("Balanced data impact on correlations"):
         st.image('images/correlations.png')
      
with tab5:
   st.header("Amount")
   plot_amount_distribution(raw_train)
   st.write("""The 𝐴𝑚𝑜𝑢𝑛𝑡 variable is strongly skewed to the right with numerous extreme outliers. 
            One way to reduce the impact of the extreme values and make the distribution more simetrical is to
            use logarithmic transformation.It makes it easier for the model to learn patterns and relationships among the features.
            """)
            
   col1, col2, = st.columns([1, 1])
   with col1:
       raw_train_log = raw_train.copy()
       raw_train_log['amount_log'] = np.log1p(raw_train_log['amount'])
       
       plot_amount_log()
       
   with col2:
       pass
   
   
with tab6:
   st.header("Time")
   plot_time(raw_train)
   st.write("𝑇𝑖𝑚𝑒 is in seconds from 0 to 172792 (so it covers 48 hours). We can assume that the modes are daytimes and the dataset describes two days.  \n",
            "The variable looks like a continous one however we decided to treat it as a categorical variable and transfor it so it represents hours in 24 hour periods.  \n",
            "The main reasons for this approch are:  \n",
            "* The values of the Time variable do not have any ordering or directionality. For example, it doesn't make sense to say that a Time value of 100 is \"greater\" or \"less\" than a value of 50.  \n",
            "* There may be certain patterns or relationships in the data that are related to the time of day, rather than the number of seconds since the first transaction")
   

with tab7:
    st.header("Feature selection")
    st.write("We tried different methods of feature selection:")
    with st.expander("Mutual information"):
        st.write("""
                Mutual information (MI) measures the extent to which knowing the value of one variable 
                reduces uncertainty about another variable. It's similar to correlation in that it 
                measures a relationship between two variables, but it can detect any kind of 
                relationship. """)
                
    with st.expander("Forward regression"):
        st.write("""
                In forward regression, we start with no features and iteratively add one feature at a 
                time, evaluating the performance of the model after each addition. 
                The process continues until a predetermined number of features is selected or until 
                adding more features does not improve the performance of the model. """)
    
    with st.expander("Backward regression"):
        st.write("""
                 Backward regression starts with all available features and iteratively removes one 
                 feature at a time, evaluating the performance of the model after each removal.
                 The process continues until a predetermined number of features is selected or until
                 removing more features does not improve the performance of the model. """)


with tab8:
    st.header("Summary of the EDA.")
    
    st.write('After completing the exploratory data analysis we decided that the dataset should be preprocessed as follows:  \n',
             '* 𝑇𝑖𝑚𝑒 variable will be transformed to categorical variable ℎ𝑜𝑢𝑟, which represents an hour of a day.  \n',
             '* 𝐴𝑚𝑜𝑢𝑛𝑡 variable will be logaritmic transformed and scaled with 𝑅𝑜𝑏𝑢𝑠𝑡𝑆𝑐𝑎𝑙𝑒𝑟().  \n',
             '* The v1, v2 ... v28 features as they are the result of PCA (Principal Component Analysis) are already standardized.  \n')
    st.write('* Impact of 𝑅𝑎𝑛𝑑𝑜𝑚𝑂𝑣𝑒𝑟𝑆𝑎𝑚𝑝𝑙𝑒𝑟() and 𝑆𝑀𝑂𝑇𝐸𝐶𝑁𝐶() on the models performance needs further analysis.')
   
#raw_data = st.session_state["raw_train"]
#st.write(raw_train.head())
# "st.session_state object: " , st.session_state

 