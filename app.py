import streamlit as st
import streamlit.components.v1 as stc

from promotion_evaluator import run_promotion_evaluator
from hoax_detector import run_hoax_detector
from segmentor_customer import run_segmentor_customer
from ml_app import run_ml_app


def main():

    st.sidebar.title("Navigation")
    selected_menu = st.sidebar.radio('Select Menu', ['Home', 'Promotion Evaluation', 'Hoax Detector', 'Customer Segmentor', 
                                                     'Stock Price Predictor', 'Voice Clonner', 'Chat WA Anonymizer', 
                                                     'Shopee Machine Affiliator', 'ml_app', 'Image Generator'])

    if selected_menu == 'Home':
        st.subheader("PORTOFOLIO OF EDO LAHARTU")
        st.write('''Passionate Data Scientist | Financial Analyst | Seasoned Leader | Business Owner | I help 
                 company transforming data into powerfull insights for informed decision-Making| Learning is adventure. 
                 I learn, so i live''')
        st.write('''Enthusiast Data Scientist that full of passion and committed to helping companies advance by 
                 helping them to develop strategic business plans based on predictive machine learning. 
                 Have been build several predict machine that already launched to web with many function 
                 like clustering customers segmentation, detecting fake news, valuating house price, predicting 
                 customer churning chance. Have a well-educated background at finance, so data science and finance 
                 competency could make magical combo at work. Most importantly, Have 13-years experience at leading 
                 people to solve many business problems.''')

    elif selected_menu == 'Promotion Evaluation':
        run_promotion_evaluator()

    elif selected_menu == 'Hoax Detector':
        run_hoax_detector()

    elif selected_menu =='Customer Segmentor':
        run_segmentor_customer()

    elif selected_menu == 'Stock Price Predictor':
        st.subheader('Which Stock You Want To valuates?')

    elif selected_menu == 'Voice Clonner':
        st.subheader('Whose Voice Do You Want To Use?')

    elif selected_menu == 'Chat WA Anonymizer':
        st.subheader('Now You Can Send Secret Chat To Your Side Chick')

    elif selected_menu == 'Shopee Machine Affiliator':
        st.subheader('Time To Make Money, Pal')

    elif selected_menu == 'ml_app':
        run_ml_app()

    elif selected_menu == 'Image Generator':
        st.markdown('(https://generateimages.streamlit.app//)')

if __name__ == '__main__':
    main()
