import streamlit as st

st.title("Supervised NLP Project")
st.text("Tiberio Zolzettich - Ocean Spiess")

st.header("Prediction of the review")
st.text_area("Your review", height=100, placeholder="Type here...")
col1, col2 = st.columns(2)
subcol1, subcol2 = col1.columns(2)
subcol1.text("Stars prediction : 4 stars")
col2.text("Type of review prediction : positive")
