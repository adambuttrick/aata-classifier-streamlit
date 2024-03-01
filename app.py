import streamlit as st
from predict import predict_labels, predict_categories

st.title('AATA Classification App')

title = st.text_input('Title', 'Enter the title here')
abstract = st.text_area('Abstract', 'Enter the abstract here')
if st.button('Predict'):
    with st.spinner('Predicting...'):
        text = title + " " + abstract
        index_terms = predict_labels(text)
        st.subheader('Predicted Index Terms')
        st.write(', '.join(index_terms))
        categories = predict_categories(text)
        st.subheader('Predicted Categories')
        st.write(', '.join(categories))