# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st

# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

# -----------------------------
# Titel and subtitle of the page
# ------------------------------
st.markdown("# Project summary")

######################
# PRESENTATION

with st.expander("Results summay"):
    st.write("""At the end of the project we feel that we have found a pipeline and a model which is able to find the majority of fraudulent transactions in the sea of 
             credit card transactions.
""")

with st.expander("Mistakes / Possible improvements"):
    st.markdown("""
- __Lack of Cross-Validation__: most of the model testing was made on the validation data. While this made possible to train on the whole training data, in exchange we \
only got one value after every prediction. By this, we risk the situation, that random chance has an effect on the metric results. With Cross-Validation we could \
get several results for the metrics, and thus lower the effect of random chance on the metric results.
- __Lack of patience / computing power while performing GridSearch__: we can almost guarantee that the hyperparameters we found during Grid-; RandomizedSearch are not \
the optimal hyperparameters. A more thorough hyperparameter search would most likely further improove our results.
- __Missed to use StratifiedKFold__: when we cross-validated the models with class_weight on the unbalanced datasets, we forgot to use StratifiedKFold instead of the \
simple KFold.
- __Robust scaling a log-transformed variable__: RobustScaler shines when it comes to data which is not normally distributed, like the 'Amount' variable. We log-transform \
a variable to make its distribution resemble to normal distribution. We did log-transform first and then RobustScaler in our pipeline. Either log-transform first and \
then use StandardScaler, or RobustScale first and then log-transform.
- __Beta value at FBeta score__: we used a beta = 5 when calculating the FBeta score. While this does the job of prioritizing the recall, this beta value was chosen \
intuitively. It is a challenge to find / guess the suitable importance ratio for Precision and Recall.
- __Custom transformer not tested in pipeline__: by the end of the project when we wanted to use the custom transformer in the pipeline, it didn't work as intended. This was fixed.
- __XGBoost class weighting__: while XGBoost does have the capability to weight classes, we didn't try it due to the lack of time
""")

with st.expander("Challenges"):
    st.markdown("""
- __Version control with Github__: we had many problems and conflicts (with the git commits :sweat_smile: ) when it came to version control and github. \
Branching, merging branches and general workflow with git/github is still a mistery...
- __Keeping track of the project__: we found ourselves in a jungle of possibilities and it was a huge challenge to keep track of out path, to navigate \
and choose between the different possibilities, as almost all of the choices were interconnected.
- __Dealing with an imbalanced datasets__: imbalanced datasets are a special kind of animal...
- __Naming__: as the number of variables and functions grew, it was more and more challenging to name them properly.
- __Letting go of the idea of perfect solution__: it is very easy listen to one's curiosity and fall into the rabbithole of pursuing the perfect solution. 
""")

with st.expander("Insights / Lessons learned"):
    st.markdown("""
- __Dealing with an imbalanced datasets__: we learned about the different strategies to deal with imbalanced datasets.
- __Double-check calculations and results__: a good practice is to ALWAYS double check results, functions, calculations or the work in general.
- __Iterative process__: a project should be an iterative process, instead of striving for a perfect solution for the first try.
- __Documenting the project__: it would have been wise to document the process of the project more frequently, as it would help to keep a bird's eye view of the project \
to reflect on it, and make the writing of the final report drastically more easy.
- __The power of team work__: 'More eyes see more'. It is easy to get on a train-track like path when working alone, but working in a group is more like a walk/drive \
on an open plain. If all works well, we can reach further as well.
- __Non-standard metrics__: there are a vast amount of unique and niche metrics to use besides the usual suspects.
- __Cutsom scorer__: using custom scorers for cross-validation.
- __Creating a Custom Transformer for a pipeline__
- __Threshold tuning__
- __Weighting classes__
- __BalancedRandomForest models__
- __Streamlit__
- __ChatGPT as a helping tool__
""")

if st.button('The End'):
    st.balloons()