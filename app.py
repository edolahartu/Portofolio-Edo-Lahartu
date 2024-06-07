import streamlit as st
import streamlit.components.v1 as stc

from promotion_evaluator import run_promotion_evaluator
from hoax_detector import run_hoax_detector


def main():

    st.sidebar.title("Navigation")

    if st.sidebar.button("Home"):
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

    if st.sidebar.button('Promotion Evaluation'):
        st.subheader('Welcome To Promotion Evaluation')
        run_promotion_evaluator()

    if st.sidebar.button('Hoax Detector'):
        st.subheader('Welcome To Hoax Detector')
        run_hoax_detector()

    
if __name__ == '__main__':
    main()
