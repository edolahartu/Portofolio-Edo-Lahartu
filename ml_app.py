import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# Attribute information for the ML section
attribute_info = """
                 - Text: The news text to be tested
                 """

# Helper function to load pickle files
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as file:
        loaded_pickle = joblib.load(file)
    return loaded_pickle

# Function to remove punctuation from text
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to run the ML app
def run_ml_app():
    st.title('Beware of Hoax News!')
    st.subheader('Welcome to Membara-Anti-Hoax: Your Reliable Partner in Fake News Detection')
    st.write('''
            In today's digital age, the rampant spread of misinformation and fake news has become a significant challenge, 
            often misleading the public and distorting the truth. At DigitalSkola, we are deeply committed to addressing this issue. 
            With the launch of Membara-Anti-Hoax, our advanced machine learning platform, we aim to detect and filter out 
            false information from the vast sea of digital content. Our goal is to empower individuals with accurate and reliable news,
            fostering a well-informed and discerning society.

            Membara-Anti-Hoax is more than just a tool; it is a testament to our dedication to truth and transparency. 
            By leveraging cutting-edge technology and extensive data analysis, we strive to provide a robust solution to 
            the problem of fake news. We invite you to join us in this crucial endeavor. Together, we can combat misinformation, 
            promote truth, and build a more informed community.
            ''')
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    text_input = st.text_area("News Text", "Enter the news text here")

    if st.button("Predict"):
        with st.expander("Your Input"):
            st.write(text_input)

        # Clean the input text
        cleaned_text = remove_punctuation(text_input)
        
        # Load the models
        model = load_pickle('MEMBARA_ANTI_HOAX.pkl')
        vectorizer = load_pickle('PENG_VECTOR_TEXT.pkl')

        # Transform the text input
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)

        # Encode the result
        result = "Real News" if prediction[0] == 1 else "Fake News"

        st.subheader('Prediction Result')
        if result == "Fake News":
            st.warning('''
                        **Caution: This news has been identified as potentially hoax.** Before believing or spreading it, make sure to verify the information from reliable sources.
        
        Here are some reasons why people create and spread fake news:
        1. Sensation and Attention: Fake news is often created to attract attention and create sensation on social media or online platforms.
        2. Financial Gain: Some individuals or groups may create and spread fake news to gain financial benefits through web traffic or advertisements.
        3. Political or Ideological Agenda: Fake news is frequently used as a tool to influence public opinion or advance specific political or ideological agendas.
        4. Dividing Communities: Fake news can be used to divide communities by spreading information that creates tension or conflict between groups.
        5. Personal Pleasure: Some individuals may create or spread fake news solely for personal amusement or as a joke.
                   
        It's important to note that these are just some common reasons, and there are many factors that may drive someone to create and spread fake news. It's crucial for all of us to always verify information before believing or sharing it further.
        ''')
        else:
            st.success('''
                         **The news has been verified as authentic** by our advanced machine learning algorithms, 
                   providing you with confidence in its accuracy and reliability. 
                   Nonetheless, we urge you to remain vigilant and critical while consuming news, 
                   as even authentic information can sometimes be misleading or contain elements of disinformation. 
                   Always cross-check facts and consider the broader context to ensure a well-rounded understanding of the news you read.
                   ''')
                 

    # Display image at the bottom of the ML section
    try:
        st.image("News_Image.png", width=200)
    except FileNotFoundError:
        st.error("Image not found. Please make sure 'News_image.png' is available.")

if __name__ == '__main__':
    run_ml_app()
