# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import precision_recall_curve

# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions

def precrecplot(y, y_prob, recall_target=0.8):
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob[:,1])
    idx = np.argwhere(np.diff(np.sign(recalls - recall_target))).flatten().max()
    pre = precisions[idx]

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    ax.margins(x=0, y=0)
    sns.lineplot(x=recalls , y=precisions, errorbar=None)
    plt.scatter(recall_target, pre, color='red')
    
    plt.vlines(x= recall_target, ymin=0, ymax=pre, color='black', linestyles='dashed')
    plt.hlines(y= pre, xmin=0, xmax=recall_target, color='black', linestyles='dashed')

    plt.legend(loc='upper right', labels=['ROS_ADA'])
    plt.title('Precision-Recall curves for ROS_ADA model on validation data')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    st.write(f"Recall target: {recall_target}\n")
    st.write(f"Precision at recall target: {pre}")
    
    st.pyplot(fig)
           

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

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)

y_train_prob_data_path = os.path.join(directory_path, 'data\\y_train_prob_data.npz')
y_train_prob_data = np.load(y_train_prob_data_path)

y_train_final = y_train_prob_data["y_train_final"]
cv_prob_final = y_train_prob_data["cv_prob_final"]

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


# --------------------
# The tab options
# --------------------


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Metric", "Different strategies on imbalanced datasets", "Models list", "Chosen models for hyperparameter tuning", "Model results after hyperparameter tuning", "Final model"])


with tab1:
    st.write("""The first challenge with model selection was to choose a metric to be able to compare different models on.""")
    with st.expander("__Accuracy__"):
        st.write("""
        As our dataset is highly imbalanced, the usual accuracy metric would not be the best metric to evaluate our models on.\n
        __Why?__ \n
        As the ratio of the majority class (class 0) in our dataset is 99.8%, any model would easily achieve 99.8% accuracy by simply classifying ever observation as class 0.
    """)

    with st.expander("__Recall__"):
        st.write("""
        We assume that the primary goal for the credit card company is to catch all the fraudulent transactions -high recall- , while minimising the number\
                 of False Positive transactions as a secondary focus, so we don't overwhelm the controlling department with extra cases. Recall alone would not be a suitable metric\
                 as we can quite easily achieve a high recall, by classifying most of the transactions as fraudulent.\
                 We would need a metric which prioritizes recall, but takes precision into consideration as well. We would use this metric to compare the different models on.
    """)
    
        rec_target = st.number_input('Please enter the desired recall score:')
        precrecplot(y_train_final, cv_prob_final, rec_target)


    with st.expander("__F1 score__"):
        st.write(r"""
        F1 score would be an interesting candidate as a metric for our case:
            
$$
                 F1 = \frac{2*Precision*Recall}{Precision+Recall}
$$
                 
The only problem with F1 score is, that it handles precision and recall as equally important factors.
    """)

    with st.expander("__Precision-Recall AUC__"):
        st.write("""
                 Another viable metric we considered is the Area Under Curve of the Precision-Recall Curve.\n
While this metric effectively measures a models overall performance based on its precision and recall results, we are still facing the same problem as with the F1 score\
namely that precision and recall is equally weighted.\n
This means that we can have an AUC score of ex. 0.9, but this number can come from a low recall - high precision OR from a high recall - low precision.\n
In this project we want to put more emphasis on the recall.
""")         
    
    with st.expander("__FBeta score__"):
        st.write(r"""
        Our solution was to use the FBeta score, which is a similar metric to F1, but it can weight precision and recall as needed:
            
$$
                     Fbeta = \frac{(1+\beta^2)*Precision*Recall}{\beta^2*Precision+Recall}
$$
                     
If $$\beta < 1$$, the Fbeta score will favour precision, and if $$\beta > 1$$, the Fbeta score will favour recall. We chose to have $$\beta = 5$$, so the Fbeta score will be relatively close to the recall score, but the precision score will have a small effect on the score.
    """)


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
    
    st.image(Image.open('images/models.png'))

        
with tab4:
    st.markdown("After testing the models with the different strategies, we needed to decide on which models we choose for hyperparameter tuning.\
                We made the following decisions:\n")
    with st.expander("Top 8 models, with the highest Fbeta score on the SMOTENC balanced dataset"):
        st.write("""
        The models on the SMOTENC balanced dataset gave generally higher Fbeta score
- BalancedRandomForestClassifier
- RandomForestClassifier
- AdaBoostClassifier
- KNeighborsClassifier
- LogisticRegression
- MLPClassifier
- XGBClassifier
- XGBRFClassifier""")

    with st.expander("Top 2 models with the lowest False Negative results"):
        st.write("""
        We wanted to try out a VotingClassifier with models which gave different kinds of errors.
- BalancedRandomForestClassifier(class_weight={0 : 0.1, 1 : 0.9})
- RandomOverSampler → AdaBoostClassifier""")

    with st.expander("2 VotingClassifier models"):
        st.markdown("""
- 3 models with best FBeta score:
    - VotingClassifier → 
        - RandomOverSampler → AdaBoostClassifier
        - SMOTENC → BalancedRandomForestClassifier
        - SMOTENC → RandomForestClassifier)
- 5 models with best recall score:
    - VotingClassifier →
        - RandomOverSampler → AdaBoostClassifier
        - BalancedRandomForestClassifier(class_weight={0 : 0.1, 1 : 0.9}
        - SMOTENC → XGBRFClassifier
        - SMOTENC → LogisticRegression
        - SMOTENC → KNeighborsClassifier""")

with tab5:
    st.markdown("__Selected models performances after hyperparameter tuning__")    
    st.image('images/selected_models.png')
    
with tab6:
    st.markdown("""Based on the result of the models after the hyperparameter tuning, we decided to choose the model with the best Fbeta score:\n
__AdaBoostClassifier on the dataset balanced with RandomOverSampler__""")
    st.image('images/final_model.png')

# ###################################################################################
# ---------------------------------------------------------------------------------






