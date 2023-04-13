# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

##############################
# Useful functions

def precrecplotthr(y, y_prob, threshold_target=0.5):
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob[:,1])
    idx = np.argwhere(np.diff(np.sign(thresholds - threshold_target))).flatten()
    if not idx and threshold_target>0.5:
        idx = len(thresholds)
    elif not idx and threshold_target<0.5:
        idx = 0
    pre = precisions[idx]
    rec = recalls[idx]

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    ax.margins(x=0, y=0)
    sns.lineplot(x=recalls , y=precisions, errorbar=None)
    plt.scatter(rec, pre, color='red')
    
    plt.vlines(x= rec, ymin=0, ymax=pre, color='black', linestyles='dashed')
    plt.hlines(y= pre, xmin=0, xmax=rec, color='black', linestyles='dashed')

    plt.legend(loc='upper right', labels=['ROS_ADA'])
    plt.title('Best Precision recall curves based on default models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    st.write(f"Threshold target: {threshold_target}\n")
    st.write(f"Precision at target threshold: {pre}")
    st.write(f"Recall at target threshold: {rec}")
    st.write("(Precision, Recall and Threshold values are calculated as a result of a 5-fold Cross Validation probability prediction on the 'full' training set)")
    
    st.pyplot(fig)


def model_threshold_finder(y_val, y_prob, matrix=False):
    # keep probabilities for the positive outcome only
    yhat = y_prob[:, 1]    
    # calculate roc curves
    precision, recall, thresholds = precision_recall_curve(y_val, yhat)
    # convert to fbeta scorea
    beta=5
    fbeta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    # locate the index of the largest f score
    ix_fbeta = np.argmax(fbeta)
    
    st.write(f'Best Fbeta Threshold={thresholds[ix_fbeta]}, F-Score={fbeta[ix_fbeta]}')

    # make predictions from probabilities based on the optimal threshold
    ypred_from_prob = np.where(yhat>=thresholds[ix_fbeta], 1, 0)
    
    if matrix==True:
        # evaluations
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_val, ypred_from_prob), ax=ax, annot=True, fmt="d", annot_kws={"size": 10}, cmap='coolwarm', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
        st.pyplot(fig)
        
        
def final_model_thresholder(y, y_prob, threshold=0.5):
    # keep probabilities for the positive outcome only
    yhat = y_prob[:, 1]    
    # calculate roc curves
    precisions, recalls, thresholds = precision_recall_curve(y, yhat)
    idx = np.argwhere(np.diff(np.sign(thresholds - threshold))).flatten()
    pre = precisions[idx]
    rec = recalls[idx]
    # convert to fbeta scorea
    beta=5
    fbeta = ((1 + beta**2) * pre * rec) / (beta**2 * pre + rec)
    
    st.write(f'''Threshold={threshold},\n
Precision at target threshold: {pre}\n
Recall at target threshold: {rec}\n
F-Score={fbeta}''')

    ypred_from_prob = np.where(yhat>=threshold, 1, 0)
    
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, ypred_from_prob), ax=ax, annot=True, fmt="d", annot_kws={"size": 10}, cmap='coolwarm', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
    st.pyplot(fig)

    # return thresholds[ix_fbeta], fbeta[ix_fbeta], yhat

# this mean end of this part###################################################################################
# ---------------------------------------------------------------------------------

# ###############################
# Titel and subtitle of the page



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

y_test_prob_data_path = os.path.join(directory_path, 'data\\y_test_prob_data.npz')
y_test_prob_data = np.load(y_test_prob_data_path)

y_test_final = y_test_prob_data["y_test_final"]
y_prob_final = y_test_prob_data["y_prob_final"]



#--------------------------------------------------
# Initializing all the datasets into SESSION_STATE
#--------------------------------------------------

#if "raw_data_un" not in st.session_state:
#    st.session_state["raw_data_un"] = raw_data_un


# ###################################################################################
# ---------------------------------------------------------------------------------


######################
# PRESENTATION



st.markdown("# Final results")
st.markdown("""As a last step in the project, we had to test the 'best model' on the test dataset.""")

tab1, tab2 = st.tabs(["Final results", "Extra: Model threshold tuning"])

with tab1:            
    with st.expander("Merging the validation dataset to the training dataset"):
        st.write("""
        At this stage we don't have any specific use for the validation dataset, so we decided to merge it with the training dataset. This way we have more data to \
            train the best model on, (hopefully) resulting in a model with better prediction capabilities.""")
            
    with st.expander("Training an AdaBoostClassifier on the expanded training data"):
        st.write("""
        As we already got the 'best' hyperparameters for the AdaBoostClassifier during the hyperparameter tuning step, we simply trained another AdaBoostClassifier\
            model with these hyperparameters on the newly expanded trainind data. Of course we resampled the training data with RandomOverSampler before the training.""")
            
    with st.expander("Final test results"):
        c1, c2 = st.columns(2)
        c1.image('images/final_fbeta.png')
        c1.image('images/final_precrec_curve.png')
        c2.write("\n\n")
        c2.write("\n\n")

        c2.image('images/final_cnf_mtrx.png')
    
        st.write("""
        The final model trained on the expanded training dataset produced the best FBeta score we have seen so far in the project.\n
        Based on this result we can assume that the model does not overfit our data and that the model can generalize well on unseen data.\n
        We can note that while this model missed 13 fraudulent transactions from the total 95, which is quite a low number compared to the previous results\
        In the meantime the model 'only' made 97 False Positive predictions, which is relatively low compared to the high recall score, and thus we will not\
        flood the transaction controlling department with too many cases to look through.""")

with tab2:
    with st.expander("Default decision threshold"):
        
        st.markdown("""All Classification estimators in sklearn implement the .predict() method. When we call the .predict() method, the results we get are the\
predicted classes for the observation. Many models base these classification predictions on a probability threshold. Example for a binary classification:\n
y_proba = [0.76, 0.24]\n
The default decision threshold for these models is 0.5, that is, if P(class 0) > 0.5 the observation will be classified as 0:\n
y_proba = [0.76, 0.24] → y_pred = 0 """)

    with st.expander("Threshold tuning"):
        
        st.markdown("""
It is possible to use a different decision threshold value for the Classification models.\n
All we need is instead of calling the .predict() method, we call the .predict_proba() method. As a result we get out an array with probability values for each corresponding class\n
As a next step we need to 'translate' the probabilities to a prediction:\n
By getting the index of the highest probability, we can return the predicted class.""")

    with st.expander("Why is threshold tuning useful?"):
        
        st.markdown("""
When we use the .predict() method, we are effectively 'locked in' to a decision threshold of 0.5.\n
By adjusting the trheshold, we can effectively 'fine tune' our models sensitivity for the predictions: we can decide that even if the model predicts 10% probability\
for class 0 for an observation, we want to classify it as class 1.\n
This in turn, highly impacts our models predictions, and by that the models performance and precision/recall/F1 scores.""")
        
        threshold_target = st.number_input('Please enter the desired decision threshold:')
        precrecplotthr(y_train_final, cv_prob_final, threshold_target)

    with st.expander("Threshold tuning in our project"):
        
        st.markdown("""
We also tried to adjust the decision threshold in our project.\n
By predicting the probabilities for every observation with cross_val_predict(method='proba') on the final model, using the training dataset\
(which at this point included the validation dataset as well) we could find the decision threshold which gave the highest FBeta score.\n
Using this threshold, we could 'translate' the final models probability predictions into predicted classes.""")

        st.markdown("Threshold for the best FBeta score on the 'full' training data:")

        model_threshold_finder(y_train_final, cv_prob_final)

    with st.expander("Results for threshold tuning in our project"):
         
         st.markdown("""
 As a result of adjusting the decison threshold, we got a better recall score, but our False Positive rate also increased significantly.\n
""")
 
         col1, col2 = st.columns(2)
         
         with col1:
             st.markdown('__Final model performance WITH threshold tuning__')
             final_model_thresholder(y_test_final, y_prob_final, threshold=0.5016222702515795)
             
         with col2:
            st.markdown('__Final model with standard threshold__')
            final_model_thresholder(y_test_final, y_prob_final, threshold=0.5)
# ###################################################################################
# ---------------------------------------------------------------------------------











