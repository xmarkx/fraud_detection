# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd
import os
import seaborn as sns
from PIL import Image

# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions

# ----------------------
# New function
#----------------------

# this mean end of this part###################################################################################
# ---------------------------------------------------------------------------------

# ###############################
# Titel and subtitle of the page

st.markdown("# Model selection")
st.markdown("### Our process of finding the best models for a highly imbalanced dataset")




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


# ###################################################################################
# ---------------------------------------------------------------------------------


######################
# PRESENTATION

# Long list of models (maybe a graph showing that in our case it was a cartesian product of (smotecnc, ros, class weight, basic) X ( models list) <- a kind of graph maybe
# Results ->  with which models we decided to go further (upload from excel, I would show only parts like smotec results in one table and then additionally the three models with high recall.
# Results after parametres tuning. + Voters ad their results
# Final model (mabe its structure -> pipeline with wteps)\

# --------------------
# The tab options
# --------------------


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Metric", "Different strategies on imbalanced datasets", "Models list", "Chosen models for hyperparameter tuning", "Model results after hyperparameter tuning", "Final model"])

with tab1:
    st.write("""The first challenge with model selection was to choose a metric to be able to compare different models on.""")
    st.write("As our dataset is highly imbalanced, the usual accuracy metric would not be the best metric to evaluate our models on.")
    st.write("__Why?__ As the ratio of the majority class (class 0) in our dataset is 99.8%, any model would easily achieve 99.8% accuracy by simply classifying ever observation as class 0.")
    st.write("""We assume that the primary goal for the credit card company is to catch all the fraudulent transactions - high recall - , while minimising the number\
             of False Positive transactions as a secondary focus, so we don't overwhelm the controlling department with extra cases. Recall alone would not be a suitable metric\
             as we can quite easily achieve a high recall, by classifying most of the transactions as fraudulent.\
             We would need a metric which prioritizes recall, but takes precision into consideration as well. We would use this metric to compare the different models on.""")
    
    st.write('# INTERACTIVE?!?!?!')
    st.image('images/prec_rec_combined.png')         
    
    st.write(r"""F1 score would be an interesting candidate as a metric for our case: 
             $$
             F1 = \frac{2*Precision*Recall}{Precision+Recall}
             $$
             
The only problem with F1 score is, that it handles precision and recall as equally important factors.

Our solution was to use the FBeta score, which is a similar metric to F1, but it can weight precision and recall as needed:
             $$
             Fbeta = \frac{(1+\beta^2)*Precision*Recall}{\beta^2*Precision+Recall}
             $$
             
If $$\beta < 1$$, the Fbeta score will favour precision, and if $$\beta > 1$$, the Fbeta score will favour recall. We chose to have $$\beta = 5$$, so the Fbeta score\
will be relatively close to the recall score, but the precision score will have a small effect on the score.""")

with tab2:
    st.write("We tried different approaches of model selection for our imbalanced dataset:\n")
    
    with st.expander("Classification models on the transformed (baseline) dataset"):
        st.write("""
        This is the most straightforward strategy: We simply tried the standard classification models models on the transformed dataset. This approach was our baseline.
    """)
    
    with st.expander("Using the BalancerRandomForestClassifier model"):
        st.write("""
        A balanced random forest classifier. A balanced random forest randomly under-samples each boostrap sample to balance it. So every time the model takes\
        bootstrap samples (samples with replacement) from the dataset, it will reduce the number of samples from the majority class, to the same number as the\
        minority class is represented, thus we get a 'balanced' bootstrap sample set each time.
    """)
    
    with st.expander("Classification models with the class_weight hyperparameter"):
        st.write("""
        The scikit-learn library has several models which have a class_weight hyperparameter. By default these models give uniform weight to every class.\
        By defining the class weights, we can penalize a models mistake made on a certain class or classes.
    """)
    
    with st.expander("Classification models on balanced dataset"):
        st.write("""
        For resampling strategy we used RandomOverSampler and SMOTENC. Both are resampling strategies found in the imbalanced-learn library. Both of these strategies\
use oversampling, a strategy where we generate datapoints which will represent the minority class.\n
        We can see the difference between the two strategies below:""")
        st.image('images/ros_smotenc_zoomed.png')

    
with tab3:
    st.markdown("The full list of all the dataset and model combinations we tested during the project.")
    col1, col2 = st.columns([2, 2])

    col1.subheader("""List of Classifiers tested:
- RandomForestClassifier
- BalancedRandomForestClassifier
- LogisticRegression
- KNeighborsClassifier
- GaussianNB
- AdaBoostClassifier
- QuadraticDiscriminantAnalysis
- MLPClassifier
- XGBClassifier
- XGBRFClassifier
    """)
    
    col2.subheader("Fbeta score for the models")
    col2.image(Image.open('images/baseline.png'))
    col2.subheader("FN/FP errors of the models")
    col2.image(Image.open('images/baseline_fn_fp.png'))
        
with tab4:
    st.markdown("After testing the models with the different strategies, we needed to decide on which models we choose for hyperparameter tuning.\
                ")
    col1, col2 = st.columns([2, 2])

    col1.subheader("""List of Classifiers tested:
- RandomForestClassifier
- BalancedRandomForestClassifier
- LogisticRegression
    """)
    
    col2.subheader("Fbeta score for the models")
    col2.image(Image.open('images/baseline.png'))
    col2.subheader("FN/FP errors of the models")
    col2.image(Image.open('images/baseline_fn_fp.png'))

with tab5:
    st.markdown("""With this approach, we are trying out default classifier models on resampled datasets\n
For resampling strategy we used RandomOverSampler and SMOTENC\n
We can see the difference between the two strategies below:""")
    st.image('images/ros_smotenc.png')
    col1, col2 = st.columns([2, 2])

# ###################################################################################
# ---------------------------------------------------------------------------------






