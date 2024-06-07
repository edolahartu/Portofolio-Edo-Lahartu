
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import re


# fungsi bantuan untuk meng-import model machine learning buat prediksi
def load_pickle(pickle_file):
    loaded_pickle = joblib.load(open(os.path.join(pickle_file), 'rb'))
    return loaded_pickle

def run_hoax_detector():

    news = st.text_input('Input The News')
    st.write(news)

    df = pd.DataFrame({'text': [news]})

    # Agar kinerja mesin lebih baik, hapus tanda baca dalam berita
    def penghapus_tanda_baca(news):
        return re.sub(r'[^\w\s]', '', news)
    
    df['text'] = df['text'].apply(penghapus_tanda_baca)

    independent = df['text']
    
    peng_vector = load_pickle('PENG_VECTOR_TEXT.pkl')
    news_tervector = peng_vector.transform(independent)
    pd.DataFrame(news_tervector)

    membara_anti_hoax = load_pickle('MEMBARA_ANTI_HOAX.pkl')


    hoax_detector = membara_anti_hoax.predict(news_tervector)

    if hoax_detector == 1:
        st.success(256* ' REAL')
    else:
        st.warning(256* ' HOAX')

