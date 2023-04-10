# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:39:04 2023

@author: katar
"""
import streamlit as st
import pandas as pd

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

st.markdown("# Credit Card Fraud Detection - Dataset ❄️")
st.markdown(" Anonymized credit card transactions labeled as fraudulent or genuine")
st.sidebar.markdown("# Page 2 ❄️")

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


# ###################################################################################
# ---------------------------------------------------------------------------------


######################
# PRESENTATION


# --------------------
# The tab options
# --------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Numeric Data", "Features distribution", "Outliers", "Samplers", "Correlation matrix", "Amount", "Time"])

with tab1:
   st.header("Numeric Data")
   st.image('images/models.png')

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab5:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   
with tab6:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
   
with tab7:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

# ###################################################################################
# ---------------------------------------------------------------------------------






