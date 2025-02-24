from streamlit_option_menu import option_menu
import joblib
import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
        ["Home", "Dataset","Preprocessing Data", "TF-IDF", "Seleksi Fitur", "Modeling", "Implementation"], 
            icons=['house', 'bar-chart', 'database', 'list', 'trash', 'folder', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="data:image/jpeg;base64,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" width="500" height="300">
        </h3>""", unsafe_allow_html=True)

    if selected == "Dataset":
        st.write("Data Sebelum Preprocessing")
        file_path = 'dataskripsi.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data)

    if selected == "Preprocessing Data":
        file_path = 'dataskripsi.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
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
        
        data['cleaned'] = data['Konten'].apply(cleaning)
        st.write("Data Setelah Proses Cleaning")
        st.write(data['cleaned'])
        data['case_folding'] = data['cleaned'].apply(case_folding)
        st.write("Data Setelah Proses Case Folding")
        st.write(data['case_folding'])
        data['tokenization'] = data['case_folding'].apply(tokenization)
        st.write("Data Setelah Proses Tokenizing")
        st.write(data['tokenization'])
        data['remove_stopwords'] = data['tokenization'].apply(remove_stopwords)
        st.write("Data Setelah Proses Stopwords Removal")
        st.write(data['remove_stopwords'])
        data['processed_text'] = data['remove_stopwords'].apply(lambda x: ' '.join(x))
        st.write("Data Setelah Preproceesing Data")
        st.write(data['processed_text'])

    if selected == "TF-IDF":
        file_path = 'processed_text.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        # Load the CountVectorizer and TfidfTransformer
        count_vectorizer = joblib.load("count_vectorizer.pkl")
        tfidf_transformer = joblib.load("tfidf_transformer.pkl")
        #Frekuensi Kata
        X_count = count_vectorizer.fit_transform(data['processed_text'])
        # #TF-IDF
        X_tfidf = tfidf_transformer.fit_transform(X_count)
        df_tfidf = pd.DataFrame(X_tfidf.toarray(),columns=count_vectorizer.get_feature_names_out())
        st.write("TF-IDF")
        st.write(df_tfidf.head(10))

    if selected == "Seleksi Fitur":
        file_path = 'processed_text.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        file_path2 = 'dataskripsi.csv'  # Ganti dengan path ke file Anda
        data2 = pd.read_csv(file_path2)
        y = data2['Kategori']
        # Load the CountVectorizer and TfidfTransformer
        count_vectorizer = joblib.load("count_vectorizer.pkl")
        tfidf_transformer = joblib.load("tfidf_transformer.pkl")
        #Frekuensi Kata
        X_count = count_vectorizer.fit_transform(data['processed_text'])
        # #TF-IDF
        X_tfidf = tfidf_transformer.fit_transform(X_count)
        information_gain = joblib.load("count_ig.pkl")
        # Menampilkan seluruh fitur beserta nilai Information Gain-nya
        feature_names = np.array(count_vectorizer.get_feature_names_out())
        feature_ig_df = pd.DataFrame({
            'Feature': feature_names,
            'Information Gain': information_gain
        })
        
        # Menampilkan hasilnya
        st.write("Fitur Sebelum Diseleksi")
        st.write("Jumlah Fitur :", len(feature_names))
        st.write(feature_ig_df.sort_values(by='Information Gain', ascending=False))

        threshold = 2
        cutoff = (threshold / 100) * information_gain.max()
        selected_features = np.where(information_gain > cutoff)[0]
        selected_feature_names = np.array(count_vectorizer.get_feature_names_out())[selected_features]
        selected_ig_values = information_gain[selected_features]
        
        # Buat DataFrame
        st.write("Fitur Setelah Diseleksi")
        st.write("Jumlah Fitur :", len(selected_feature_names))
        df_selected_features = pd.DataFrame({
            "Feature": selected_feature_names,
            "Information Gain": selected_ig_values
        })
        st.write(df_selected_features.sort_values(by='Information Gain', ascending=False))

    if selected == "Modeling":
        file_path = 'processed_text.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        file_path2 = 'dataskripsi.csv'  # Ganti dengan path ke file Anda
        data2 = pd.read_csv(file_path2)
        y = data2['Kategori']
        # Encode label kategori menjadi angka
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        # Load the CountVectorizer and TfidfTransformer
        count_vectorizer = joblib.load("count_vectorizer.pkl")
        tfidf_transformer = joblib.load("tfidf_transformer.pkl")
        #Frekuensi Kata
        X_count = count_vectorizer.fit_transform(data['processed_text'])
        # #TF-IDF
        X_tfidf = tfidf_transformer.fit_transform(X_count)
        #Seleksi Fitur
        information_gain = joblib.load("count_ig.pkl")
        threshold = 2
        cutoff = (threshold / 100) * information_gain.max()
        selected_features = np.where(information_gain > cutoff)[0]
        X_selected = X_tfidf[:, selected_features]
        #Modeling
        model = joblib.load("model_fold_4_baru.pkl")
        # Tentukan jumlah fold yang sama dengan saat pelatihan
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Ambil data dari fold ke-4 sebagai data uji
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_selected), start=1):
            if fold_idx == 4:
                X_test, y_test = X_selected[test_idx], y_encoded[test_idx]
                break
        
        # Lakukan prediksi
        y_pred = model.predict(X_test)
        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().round(4)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        st.title("Evaluasi Model SVM")
        st.write(f"### Akurasi: {accuracy * 100:.2f}%")
        st.write("### Classification Report")
        st.dataframe(report_df)
        # Plot Confusion Matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        
    if selected == "Implementation":
        # Load the CountVectorizer and TfidfTransformer
        count_vectorizer = joblib.load("count_vectorizer.pkl")
        tfidf_transformer = joblib.load("tfidf_transformer.pkl")
        #Load Seleksi Fitur dan Model
        feature_selection = joblib.load("feature_selection.pkl")
        model = joblib.load("model_fold_4.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        # Ambil fitur yang dipilih dari file feature_selection.pkl
        selected_features = feature_selection['selected_features']
        st.write("seleksi fitur :",selected_features.shape)
        
        with st.form("my_form"):
            new_text = st.text_area('Masukkan Berita')
            submit = st.form_submit_button("Klasifikasi")
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
                    st.write("terseleksi :",X_new_selected.shape)
        
                    # Prediksi Kategori
                    prediction = model.predict(X_new_selected)[0]
                    # Konversi ke label asli
                    predicted_label = label_encoder.inverse_transform([prediction])[0]
                    st.success(f"Prediksi Kategori Berita: **{predicted_label}**")
                else:
                    st.error("Masukkan ulasan terlebih dahulu!")

        
          


        
