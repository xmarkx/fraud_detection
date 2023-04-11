# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os


# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions


# ----------------------
# Expandable container
#----------------------

def collab_text(title, text):
    with st.container():
        with st.expander(title, expanded=False):
            #st.markdown(f"<h2 style='color: blue; font-weight: bold'>{title}</h2>", unsafe_allow_html=True)
            st.write(text)

# -----------------------------
# Loading and caching the data
# ----------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(data_path)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

####################################################################################
# ---------------------------------------------------------------------------------


# -----------------------------
# Titel and subtitle of the page
# ------------------------------
st.markdown("# Credit Card Fraud Detection - Dataset ❄️")
st.markdown(" Anonymized credit card transactions labeled as fraudulent or genuine")
st.sidebar.markdown("# Page 2 ❄️")



#####################################
# DATA PROCESSING


#-------------------
# FETCH SOME DATA
#-------------------

st.write("Loading the data...")

# getting the path to the file with data
current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)

data_path = os.path.join(directory_path, 'data\\creditcard.csv')

# st.write(current_file_path)
# st.write(directory_path)
st.write(data_path)


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
raw_data_un = load_data()

# Notify the reader that the data was successfully loaded.
#data_load_state.text('Loading data...done!')
data_load_state.text("Done! (using st.cache_data)")


#-------------------------------------------
# Removing duplicates and saving the dataset
#-------------------------------------------
raw_data = raw_data_un.drop_duplicates(keep='first')


#-----------------------------------------------------------------
# Spliting the dataset into training, validation and test dataset
#-----------------------------------------------------------------

# spliting the data into train and test set
raw_train, raw_valid_test = train_test_split(raw_data, test_size=0.4, random_state = 42, shuffle=True, stratify=raw_data["class"])
# spliting the test set into validation and test set
raw_val, raw_test = train_test_split(raw_valid_test, test_size=0.5, random_state = 42, shuffle=True, stratify=raw_valid_test["class"])


#--------------------------------------------------
# Initializing all the datasets into SESSION_STATE
#--------------------------------------------------

if "raw_data_un" not in st.session_state:
    st.session_state["raw_data_un"] = raw_data_un

if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = raw_data
    
if "raw_train" not in st.session_state:
    st.session_state["raw_train"] = raw_train
    
if "raw_val" not in st.session_state:
    st.session_state["raw_val"] = raw_val

if "raw_test" not in st.session_state:
    st.session_state["raw_test"] = raw_test   

############################################################################
# ------------------------------------------------------------------------


######################
# PRESENTATION


# ----------------
# Config
# ----------------
#st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')




# --------------------
# The tab options
# --------------------

tab1, tab2, tab3, tab4, = st.tabs(["Raw data", "Duplicates", "Class Imbalance", "Data Splitting"])



# ----------------
# Raw data
# ----------------

with tab1:
   st.header("Raw data")

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
        cols_to_display = st.multiselect("Select Columns to Display", raw_data_un.columns, default = cols_default)
                
        # Selected the columns
        data_subset = raw_data_un[cols_to_display]
    
        # Define the slider to choose how many rows to display
        num_rows = st.slider("Number of Rows to Display", min_value=1, max_value=len(raw_data_un), value=10)
    
        # Show dataset if checkbox is selected
        st.write(data_subset.head(num_rows))
        
        st.write('* The dataset has 31 features  \n',
             '* The dataset has 284807 observations in total  \n',
             '* The dataset has no missing values  \n',
             '* The target feature is Class  \n'
             '* Except for Time, Amount and Class features, all other features are unnamed, which is a result of PCA dimension reduction and also serves as a way to anonymize the data  \n',
             '* All the features have numeric data, except Time and Class  \n',
             '* Class is Nominal Categoric data  \n',
             '* Time (copy the description from Kaggel could be Ordinal Categoric data  \n')



# ----------------
# Duplicates
# ----------------

with tab2:
   st.header("Duplicates")
   num_duplicates = raw_data_un.duplicated(keep='first').sum()
   st.subheader("Duplicates")
   st.write("The dataset contained  ", num_duplicates," transactions which are duplicates. The duplicates has been removed to ensure the integrity of the data as duplicated transactions can bias the model's performance metrics.")
   # Define the checkbox to show/hide the duplicates
   show_duplicates = st.checkbox("Show duplicates")
    
    
   if show_duplicates:
       
        # Selected the columns
        duplicates = raw_data_un[raw_data_un.duplicated(keep=False)]

        # Define the slider to choose how many rows to display
        num_rows_dup = st.slider("Number of Rows to Display", min_value=1, max_value=len(duplicates), value=5)

        # Show dataset if checkbox is selected
        st.write(duplicates.head(num_rows_dup))
        
        # The describtion as expandable 
        collab_text("More about duplicates", 
                    "A transaction is treated as duplicate olny if there is another transaction with exactly the same values in all columns.")



# ----------------
# Class Imbalance
# ----------------

with tab3:
   st.header("Class Imbalance")
   st.write("The dataset is highly imbalanced, with fraudulent transactions making up only a small fraction of the total transactions. This can be challenging for machine learning models to handle, as they tend to prioritize accuracy over detecting the minority class. To address this, we will be using evaluation metrics that are better suited for imbalanced datasets, such as precision, recall, and F1 score.")
   data = st.session_state['raw_data']

   # Get the class distribution
   class_counts = data['class'].value_counts()
   minority_class = class_counts.index[-1]
   majority_class = class_counts.index[0]
    
   proportion = raw_data['class'].value_counts(normalize=True)
   amount = raw_data['class'].value_counts(normalize=False)
    

    # Display the class distribution
   st.write("Class distribution:")
   fig, ax = plt.subplots()
   bar = sns.countplot(x='class', data=data, ax=ax)
   for i in range (len(proportion)):
       bar.text(i, amount[0]/2, str(round(proportion[i]*100,2)), 
                fontdict=dict(fontsize=12), horizontalalignment='center')
   ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
   st.pyplot(fig)

   
   # Discuss the impact of class imbalance
   st.write("The imbalance of this dataset may lead to models that are biased towards the majority class, and may perform poorly on the minority class. It's important to use techniques like oversampling or undersampling to address this issue.")

   # Discuss potential solutions
   st.write("Potential solutions to address class imbalance include oversampling the minority class, undersampling the majority class, or using techniques like SMOTE to generate synthetic samples of the minority class.")

    
    
    
   st.write("We will be using  (...) to measure the performance of our model. These metrics take into account the imbalance in the dataset and provide a better measure of how well the model is detecting fraudulent transactions.")


# ----------------
# Data splitting
# ----------------

with tab4:
   st.header("Data Splitting")
   st.write("The dataset was split into a training set (60% of the data), a validation set (20% of the data), and a test set (20% of the data). This was done to ensure that the model is trained on a sufficient amount of data and can generalize well to unseen data.")

   st.write("Training set dimensions: ", raw_train.shape)#, 'amd training lagels:', y_train.shape,)
   st.write("Test set dimensions: ", raw_test.shape)#, 'and test labels:', y_test.shape)
   st.write("Validation dataset dimensions: ", raw_val.shape)#, 'and validation labels:', y_val.shape)





