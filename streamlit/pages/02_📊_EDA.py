# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:40:00 2023

@author: katar
"""

import streamlit as st

#import 01_Dataset
#01_Dataset.load_data()

# Config
st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', initial_sidebar_state="expanded", layout='wide')

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")



tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Numeric Data", "Features distribution", "Outliers", "Samplers", "Correlation matrix", "Amount", "Time"])

with tab1:
   st.header("Numeric Data")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

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
   
   
raw_data = st.session_state["raw_data"]
st.write(raw_data)


#"st.session_state object: " , st.session_state