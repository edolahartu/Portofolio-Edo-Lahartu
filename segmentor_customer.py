'''
Tahapan pembuatan code deployment adalah:
1. import libraries yg dibutuhkan
2. import pickle yang dibutuhkan
3. bikin user interface
    a. bikin code untuk input data
    b. bikin code untuk preprocessing dan encoder
5. bikin code untuk jalankan model
6. bikin code untuk output 
'''

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle


#load pickle
# fungsi bantuan untuk meng-import model machine learning buat prediksi
def load_pickle(pickle_file):
    loaded_pickle = joblib.load(open(os.path.join(pickle_file), 'rb'))
    return loaded_pickle

scaler              = load_pickle('3_SCALER_CUSTOMER.pkl')
dimensi_reductor    = load_pickle('3_PCA_REDUCTOR_DIMENSI.pkl')
segmentor_model     = load_pickle('3_KMEANS_SEGMENTOR.pkl')

def run_segmentor_customer():

    st.title('Welcome to Our Customer Segmentation Survey!')
    st.subheader('Help Us Serve You Better')
    st.write('''        
        At EnC Retail, we are committed to providing the best services, offers, and promotions tailored specifically to you. 
        To achieve this goal, we need your valuable input. By participating in our Customer Segmentation Survey, 
        you will help us understand your preferences, needs, and behaviors better. 
        This understanding allows us to personalize our offerings, ensuring you receive the best possible experience.''')
    st.subheader('Why Participate?')
    st.write(''' 
        Personalized Services: Your feedback helps us identify what services are most important to you and how we can improve them.
        Best Offers: By understanding your preferences, we can offer promotions and deals that are most relevant and beneficial to you.
        Tailored Promotions: We aim to send you promotions that suit your individual needs, 
        making your experience with us more rewarding.''')
    st.subheader('How It Works')
    st.write('''  
        The survey will take just a few minutes to complete. We will ask you some questions about your age, income, spending habits, 
        and marital status. This information will be used to categorize customers into different segments, 
        allowing us to tailor our services to better meet your needs.''')
    st.subheader('We Value Your Privacy')
    st.write('''  
        Rest assured, all your responses will be kept confidential and will be used solely for the purpose of improving our services. 
        Your participation is entirely voluntary, and you can choose to skip any question you are not comfortable answering.''')
    st.subheader('Thank You!')
    st.write(''' 
        We appreciate your time and effort in helping us serve you better. Your feedback is invaluable to us, 
        and we are excited to use it to enhance your experience with EnC Retail.

        Thank you for being a valued customer!
                     ''')

#make us user interface
## bikin code untuk input data
    sex                 = st.radio('Gender: Male = 0;--  Female = 1', ['1', '0'])
    marital_status      = st.radio('Married: Single = 0;--  Non-Single = 1', ['1', '0'])
    age                 = st.number_input('Customer Age', 18, 78)
    education           = st.radio('''
                            Level of Education:
                                   No education = 0;--
                                   Highscool = 1;--
                                   Bachelor = 2;--
                                   Master = 3
                            ''', ['0', '1', '2', '3'])
    income              = st.number_input('Yearly Income in Thousand Dollar(k$)',1 , 100)
    occupation          = st.radio('''
                            Job Status:
                                   Unemployed = 0;--  
                                   Employed = 1;--
                                   Self-Employed or Management = 2
                            ''', ['0', '1', '2'])
    settlement_size     = st.radio('''
                            Size of Customer City:
                                   Small Size: 0;--
                                   Mid Size: 1;--
                                   Big Size: 2
                            ''', ['0', '1', '2'])
    
    with st.expander('Your Proflie'):
        result = {
            'Sex': sex,
            'Marital Status': marital_status,
            'Education': education,
            'Age': age,
            'Income': income,
            'Occupation': occupation,
            'Settlement Size': settlement_size
        }
    st.write(result)

    customer_segment = pd.DataFrame([result])
    segment = segmentor_model.predict(customer_segment)

    st.write("Predicted Segment:", segment[0])

    
