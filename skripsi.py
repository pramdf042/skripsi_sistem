from streamlit_option_menu import option_menu
import joblib
import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
import nltk
from nltk.corpus import stopwords
import time
from math import log2
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Download stopwords from nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import CountVectorizer
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

@st.cache_resource
def get_driver():
    return webdriver.Chrome(
        service=Service(
            ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        ),
        options=options,
    )
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon='https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Klasifikasi Berita Pariwisata Menggunakan Metode SVM Multikelas</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset","Implementation"], 
            icons=['house', 'bar-chart', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://lh5.googleusercontent.com/p/AF1QipNgmGyncJl5jkHg6gnxdTSTqOKtIBpy-kl9PgDz=w540-h312-n-k-no" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    if selected == "Dataset":
        st.write("Data Sebelum Preprocessing")
        file_path = 'dataskripsi.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data['Konten'].head(10))
        st.write("Data Setelah Preprocessing")
        file_path2 = 'processed_text.csv'  # Ganti dengan path ke file Anda
        data2 = pd.read_csv(file_path2)
        st.write(data2['processed_text'].head(10))
    if selected == "Implementation":
        def cleaning(text):
            text = re.sub(r'_x000D_+', ' ', text)
            text = re.sub(r'SCROLL TO CONTINUE WITH CONTENT', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'[-()\"#/@;:<>{}\'+=~|.!?,_]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def case_folding(text):
            return text.lower()
        
        def tokenization(text):
            return nltk.tokenize.word_tokenize(text)
        
        def remove_stopwords(text):
            return [word for word in text if word not in stop_words]

        # Load the CountVectorizer and TfidfTransformer
        count_vectorizer = joblib.load("count_vectorizer.pkl")
        tfidf_transformer = joblib.load("tfidf_transformer.pkl")
        #Load Seleksi Fitur dan Model
        feature_selection = joblib.load("feature_selection.pkl")
        model = joblib.load("model_fold_4.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        # Ambil fitur yang dipilih dari file feature_selection.pkl
        selected_features = feature_selection['selected_features']
        
        with st.form("my_form"):
            new_text = st.text_input('Masukkan Berita')
            submit = st.form_submit_button("Prediksi")
            if submit:
                if new_text.strip():
                    #Preprocessing Berita Baru
                    clean_text = cleaning(new_text)
                    folded_text = case_folding(clean_text)
                    tokenized_text = tokenization(folded_text)
                    filtered_text = remove_stopwords(tokenized_text)
                    processed_text = ' '.join(filtered_text)
                    # Transformasi TF-IDF
                    text_counts = count_vectorizer.transform([processed_text])
                    text_tfidf = tfidf_transformer.transform(text_counts)
                    # Pilih hanya fitur yang relevan
                    X_new_selected = text_tfidf[:, selected_features]
        
                    # Prediksi Kategori
                    prediction = model.predict(X_new_selected)[0]
        
                    # Konversi ke label asli
                    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        
                    st.success(f"Prediksi Kategori Berita: **{predicted_label}**")
                else:
                    st.error("Masukkan ulasan terlebih dahulu!")

        
          


        
