
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import re
import matplotlib.pyplot as plt

'''
Tahapan pembuatan code deployment adalah:
1. import libraries yg dibutuhkan
2. bikin code untuk load pickle
3. bikin code untuk terima input dari user
4. bikin code untuk preprocessing dan encoder
5. bikin code untuk jalankan model
6. bikin code untuk output 
'''
# fungsi bantuan untuk meng-import model machine learning buat prediksi
def load_pickle(pickle_file):
    loaded_pickle = joblib.load(open(os.path.join(pickle_file), 'rb'))
    return loaded_pickle

def run_hoax_detector():

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
    
    st.subheader('Input The News')
    news = st.text_input('The News')
    st.write(news)

    if news:
        #preprocess the input news

        # Agar kinerja mesin lebih baik, hapus tanda baca dalam berita
        def penghapus_tanda_baca(news):
            return re.sub(r'[^\w\s]', '', news)
        
        df = pd.DataFrame({'text': [news]})
        df['text'] = df['text'].apply(penghapus_tanda_baca)

        independent = df['text']
        
        #load vectorizer and model from pickle
        peng_vector = load_pickle('PENG_VECTOR_TEXT.pkl')
        membara_anti_hoax = load_pickle('MEMBARA_ANTI_HOAX.pkl')
        
        # vectorize the input
        news_tervector = peng_vector.transform(independent)
        pd.DataFrame(news_tervector)

        #predict using model
        hoax_detector = membara_anti_hoax.predict(news_tervector)[0]

        # # Function to plot pie chart
        # def plot_pie_chart(data, title):
        #     plt.figure(figsize=(8, 6))
        #     plt.pie(
        #         data.value_counts(),
        #         autopct="%1.1f%%",
        #         colors=["#99ff99", "#66b3ff", "#ff9999"],
        #         startangle=90,
        #         explode=(0.1, 0, 0),
        #         shadow=True,
        #     )
        #     plt.legend(labels=["Neutral", "Positive", "Negative"])
        #     plt.title(title)
        #     st.pyplot(plt)  # Display the plot in Streamlit
        #     plt.clf()  # Clear the plot    

        if hoax_detector == 1:
            st.success('''
                    **The news has been verified as authentic** by our advanced machine learning algorithms, 
                    providing you with confidence in its accuracy and reliability. 
                    Nonetheless, we urge you to remain vigilant and critical while consuming news, 
                    as even authentic information can sometimes be misleading or contain elements of disinformation. 
                    Always cross-check facts and consider the broader context to ensure a well-rounded understanding of the news you read.
                    ''')
        else:
            st.warning("""
                        **Caution: This news has been identified as potentially hoax.** Before believing or spreading it, make sure to verify the information from reliable sources.
            
            Here are some reasons why people create and spread fake news:
            1. Sensation and Attention: Fake news is often created to attract attention and create sensation on social media or online platforms.
            2. Financial Gain: Some individuals or groups may create and spread fake news to gain financial benefits through web traffic or advertisements.
            3. Political or Ideological Agenda: Fake news is frequently used as a tool to influence public opinion or advance specific political or ideological agendas.
            4. Dividing Communities: Fake news can be used to divide communities by spreading information that creates tension or conflict between groups.
            5. Personal Pleasure: Some individuals may create or spread fake news solely for personal amusement or as a joke.
                    
            It's important to note that these are just some common reasons, and there are many factors that may drive someone to create and spread fake news. It's crucial for all of us to always verify information before believing or sharing it further.
            """)

