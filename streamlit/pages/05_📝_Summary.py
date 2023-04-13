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
- __Lack of Cross-Validation__: most of the model testing was made on the validation data. While this made possible to train on the whole training data, in exchange we\
only got one value after every prediction. By this, we risk the situation, that random chance has an effect on the metric results. With Cross-Validation we could\
get several results for the metrics, and thus lower the effect of random chance on the metric results.
- __Lack of patience / computing power while performing GridSearch__: we can almost guarantee that the hyperparameters we found during Grid-; RandomizedSearch are not\
the optimal hyperparameters. A more thorough hyperparameter search would most likely further improove our results.
- __Beta value at FBeta score__: we used a beta = 5 when calculating the FBeta score. While this does the job of prioritizing the recall, this beta value was chosen\
intuitively. It is a challenge to find / guess the suitable importance ratio for Precision and Recall.
""")

with st.expander("Challenges"):
    st.markdown("""
- __Version control with Github__: we had many problems and conflicts (with the git commits :sweat_smile: ) when it came to version control and github. \
Branching, merging branches and general workflow with git/github is still a mistery...
- __Keeping track of the project__: after the EDA, we found ourselves in a jungle of possibilities and it was a huge challenge to keep track of out path, to navigate\
and choose between the different possibilities.
- __Dealing with an imbalanced datasets__: imbalanced datasets are a special kind of animal...
""")

with st.expander("Insights / Lessons learned"):
    st.markdown("""
- __Dealing with an imbalanced datasets__: we learned about the different strategies to deal with imbalanced datasets.
- __ETC.__
""")