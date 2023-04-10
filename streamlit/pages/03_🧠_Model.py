# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd




st.markdown("# Credit Card Fraud Detection - Dataset ❄️")
st.markdown(" Anonymized credit card transactions labeled as fraudulent or genuine")
st.sidebar.markdown("# Page 2 ❄️")


raw_data = st.session_state["raw_data"]

#Define the sidebar options
options = {
    "Raw Data": True,
    "Duplicates": False,
    "Class Imbalance": False,
    "Data Splitting": False
}

# Render the checkboxes in the sidebar
for key, value in options.items():
    options[key] = st.sidebar.checkbox(key, value)

# Render the selected sections
if options["Raw Data"]:
    st.subheader("Raw Data")
    
    st.write(
        """
        The dataset contains transaction data from a two-day period in September 2013, where a total of 492 fraud 
        cases out of 284,807 transactions occurred. The dataset includes a mix of numerical and categorical
        features, such as transaction amount, time, and various V1-V28 features, which are the result of a PCA
        transformation to protect user privacy.
        """
        )
    
    
    # Define the checkbox to show/hide the dataset
    show_dataset = st.checkbox("Show Dataset")

    if show_dataset:
        # Define columns to display
        cols_default = ['class', 'time', 'amount', 'v1', 'v2', 'v3', 'v4', 'v26', 'v27', 'v28']
        cols_to_display = st.multiselect("Select Columns to Display", raw_data.columns, default = cols_default)
                
        # Selected the columns
        data_subset = raw_data[cols_to_display]

        # Define the slider to choose how many rows to display
        num_rows = st.slider("Number of Rows to Display", min_value=1, max_value=len(raw_data), value=10)

        # Show dataset if checkbox is selected
        st.write(data_subset.head(num_rows))
        
    st.write('* The dataset has 31 features  \n',
             '* The dataset has 284807 observations in total  \n',
             '* The dataset has no missing values  \n',
             '* The target feature is Class  \n'
             '* Except for Time, Amount and Class features, all other features are unnamed, which is a result of PCA dimension reduction and also serves as a way to anonymize the data  \n',
             '* All the features have numeric data, except Time and Class  \n',
             '* Class is Nominal Categoric data  \n',
             '* Time could be Ordinal Categoric data  \n')

if options["Duplicates"]:
    num_duplicates = len(raw_data.duplicated(keep=False))
    duplicates = raw_data[raw_data.duplicated(keep=False)]
    st.subheader("Duplicates")
    #st.write("## Duplicates")
    st.write("The dataset contained  ", num_duplicates," transactions, which were identified and removed to ensure the integrity of the data. Duplicate transactions can skew the model's performance metrics and should be removed from the dataset.")


if options["Class Imbalance"]:
    st.subheader("Class Imbalance")
    st.write("The dataset is highly imbalanced, with fraudulent transactions making up only a small fraction of the total transactions. This can be challenging for machine learning models to handle, as they tend to prioritize accuracy over detecting the minority class. To address this, we will be using evaluation metrics that are better suited for imbalanced datasets, such as precision, recall, and F1 score.")
    
    st.write("We will be using  (...) to measure the performance of our model. These metrics take into account the imbalance in the dataset and provide a better measure of how well the model is detecting fraudulent transactions.")

if options["Data Splitting"]:
    st.subheader("Data Splitting")
    st.write("## Data Splitting")
    st.write("The dataset was split into a training set (70% of the data), a validation set (15% of the data), and a test set (15% of the data). This was done to ensure that the model is trained on a sufficient amount of data and can generalize well to unseen data.")



#########################
# Display chosen columns
#########################








