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


@st.cache_data
def plot_class_distribution(data):
    # Get the class distribution
    class_counts = data['class'].value_counts()
    minority_class = class_counts.index[-1]
    majority_class = class_counts.index[0]

    proportion = data['class'].value_counts(normalize=True)
    amount = data['class'].value_counts(normalize=False)

    # Display the class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bar = sns.countplot(x='class', data=data, ax=ax)
    for i in range(len(proportion)):
        bar.text(i, amount[0]/2, str(round(proportion[i]*100,2)) + "%", fontdict=dict(fontsize=12), horizontalalignment='center')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title("Class Distribution", fontsize=16)
    st.pyplot(fig)




####################################################################################
# ---------------------------------------------------------------------------------


# -----------------------------
# Titel and subtitle of the page
# ------------------------------
st.markdown("# Credit Card Fraud Detection - Dataset")
st.markdown("The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.")
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

   st.markdown(
        """
        The dataset contains transaction data from a two-day period in September 2013, where a total of 492 fraud 
        cases out of 284,807 transactions occurred. """)
       
     
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
        
   st.write('The dataset includes:  \n',
        "* Features $V1, V2, … V28$ which are the principal components obtained with PCA. (Due to confidentiality issues more background details couldn't be provided.')  \n",
        "* $Time$ contains the seconds elapsed between a transaction and the first transaction in the dataset.  \n",
        "* $Amount$ is the transaction amount.  \n",
        "* $Class$ is the response/target variable and it takes value $1$ in case of fraud and $0$ otherwise.  \n",
        '* All the features have numeric data, however $Time$ and $Amount$ needs further analysis. \n',
        '* $Class$ is nominal categoric data.')
          
  


# ----------------
# Duplicates
# ----------------

with tab2:
   st.header("Duplicates")
   num_duplicates = raw_data_un.duplicated(keep='first').sum()
   #st.subheader("Duplicates")
   st.write("The dataset contained  ", num_duplicates," transactions which are duplicates.")
   st.write("A transaction is treated as a duplicate only if there is another transaction with exactly the same values in all columns.")
   st.write("The duplicates has been removed to ensure the integrity of the data as duplicated transactions can bias the model's performance metrics.")
   # Define the checkbox to show/hide the duplicates
   show_duplicates = st.checkbox("Show duplicates")
    
    
   if show_duplicates:
       
        # Selected the columns
        duplicates = raw_data_un[raw_data_un.duplicated(keep=False)]

        # Define the slider to choose how many rows to display
        num_rows_dup = st.slider("Number of Rows to Display", min_value=1, max_value=len(duplicates), value=10)

        # Show dataset if checkbox is selected
        st.write(duplicates.head(num_rows_dup))
      



# ----------------
# Class Imbalance
# ----------------

with tab3:
   st.header("Class Imbalance")
   st.write("""
            The dataset is highly imbalanced, with fraudulent transactions making up only a small fraction of the total 
            transactions. This can be challenging for machine learning models to handle, as they tend to prioritize 
            accuracy over detecting the minority class.""")
   plot_class_distribution(raw_data)

   
   # Discuss the impact of class imbalance
   st.write("""A model that simply predicts all transactions as non-fraudulent would achieve an accuracy of 99.8%. 
            This means that we will need to find other metrics which provide a better measure of how well the model
            is detecting fraudulent transactions (precision, recall, F1 score, ROC AUC etc).
            """)

   # Discuss potential solutions
   st.write("Another solution which is worth considering is to include oversampling of the minority class or undersampling the majority class.")


#data = st.session_state['raw_data']
# ----------------
# Data splitting
# ----------------

with tab4:
   st.header("Data Splitting")
   st.write("The dataset was split into a training set (60% of the data), a validation set (20% of the data), and a test set (20% of the data).")
   st.write("* Training set dimensions: ", raw_train.shape, "  \n",
            "* Test set dimensions: ", raw_test.shape, "  \n",
            "* Validation dataset dimensions: ", raw_val.shape)
   
   datasets = {"Dataset": ["Raw Data", "Train Data", "Validation set", "Test Data"],
               "Total Rows": [len(raw_data), len(raw_train), len(raw_val), len(raw_test)],
               "Class 0": [len(raw_data[raw_data['class'] == 0]), len(raw_train[raw_train['class'] == 0]), len(raw_val[raw_val['class'] ==0]), len(raw_test[raw_test['class'] == 0])],
               "Class 1": [len(raw_data[raw_data['class'] == 1]), len(raw_train[raw_train['class'] == 1]), len(raw_val[raw_val['class'] ==1]),  len(raw_test[raw_test['class'] == 1])]
           }
   df_datasets = pd.DataFrame(datasets)
   st.write(df_datasets)


