from streamlit_option_menu import option_menu
import joblib
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
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
    page_title="Sentimen Analysis",
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
<center><h2 style = "text-align: justify;">Analisi Sentimen Wisata Madura Dengan Maximum Entropy</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset","prediksi ulasan","Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
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
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        st.write(data['Ulasan'].head(10))
        st.write("Data Setelah Preprocessing")
        file_path2 = 'data preprocessing.csv'  # Ganti dengan path ke file Anda
        data2 = pd.read_csv(file_path2)
        st.write(data2['Ulasan'].head(10))
    if selected == "prediksi ulasan":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')
        with st.form("my_form"):
            st.subheader("Implementasi")
            ulasan = st.text_input('Masukkan ulasan')  # Input ulasan dari pengguna
            submit = st.form_submit_button("Prediksi")
            if submit:
                if ulasan.strip():  # Validasi input tidak kosong
                    # Transformasikan ulasan ke bentuk vektor
                    new_X = vectorizer.transform([ulasan]).toarray()
        
                    # Membuat dictionary dengan nama feature sesuai format model
                    new_data_features = {f"feature_{j}": new_X[0][j] for j in range(new_X.shape[1])}
                    
                    # Prediksi menggunakan model
                    new_pred = loaded_model.classify(new_data_features)
        
                    # Tampilkan hasil prediksi
                    st.subheader('Hasil Prediksi')
                    st.write(f"Prediction for New Data: {new_pred}")
                else:
                    st.error("Masukkan ulasan terlebih dahulu!")

    if selected == "Implementation":
        import joblib
        # Menggunakan pandas untuk membaca file CSV
        file_path = 'data stopword tes.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(data['stopword']).toarray()
        loaded_model = joblib.load('final_maxent_model.pkl')
        loaded_vectorizer = joblib.load('tfidf (1).pkl')


    
        st.subheader("Implementasi Menggunakan Data Baru")
        url = "https://colab.research.google.com/drive/1im2fPYWSGElnmKdOR8ysApZsiG_jOlRR?usp=sharing"
        st.write("Code implementasi scrapping dengan selenium [link](%s)" % url)
        url2 = "https://github.com/feb11fff/sistem-skripsi/tree/main"
        st.write("source code implemnatasi streamlit pada github [link](%s)" % url2)
        st.title("pilih sentimen wisata")
        
      
        if st.button("Bukit Jaddih"):
            try:
                import requests
            
                # URL dari selenium di vps
                api_url = "http://157.66.54.50:8501/scrape"
                
                # URL yang ingin di-scrape
                payload = {"url": "https://www.google.com/maps/place/Wisata+Bukit+Jaddhih/@-7.082283,112.7569647,17z/data=!4m8!3m7!1s0x2dd8045eb0acb79d:0x4a24af02fd796f55!8m2!3d-7.082283!4d112.7595396!9m1!1b1!16s%2Fg%2F11c2r8kctr?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"}
                
                # Mengirimkan POST request ke API FastAPI
                response = requests.post(api_url, json=payload)
                
                # Memeriksa status dan menampilkan hasilnya
                if response.status_code == 200:
                    result = response.json()
                    reviews = result.get('reviews', [])
                    print("Review Results:")
                    for review in reviews:
                        print(review)
                else:
                    print("Error:", response.status_code)
                reviews = response.json().get('reviews', [])
                    # Mengambil 5 data pertama dari kolom 'ulasan'
                top_10_reviews = reviews[-5:]
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        if st.button("Pantai Slopeng"):
            try:
                import requests
            
                # URL dari selenium di vps
                api_url = "http://157.66.54.50:8501/scrape"
                
                # URL yang ingin di-scrape
                payload = {"url": "https://www.google.com/maps/place/Pantai+Slopeng/@-6.8861093,113.7820433,15z/data=!4m8!3m7!1s0x2dd9ea23fabac2df:0x8550176773c06614!8m2!3d-6.8861095!4d113.792343!9m1!1b1!16s%2Fg%2F112yfwt6c?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"}
                
                # Mengirimkan POST request ke API FastAPI
                response = requests.post(api_url, json=payload)
                
                # Memeriksa status dan menampilkan hasilnya
                if response.status_code == 200:
                    result = response.json()
                    reviews = result.get('reviews', [])
                    print("Review Results:")
                    for review in reviews:
                        print(review)
                else:
                    print("Error:", response.status_code)
                reviews = response.json().get('reviews', [])
                    # Mengambil 5 data pertama dari kolom 'ulasan'
                top_10_reviews = reviews[-5:]
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        if st.button("Pantai Sembilan"):
            try:
                import requests
            
                # URL dari selenium di vps
                api_url = "http://157.66.54.50:8501/scrape"
                
                # URL yang ingin di-scrape
                payload = {"url": "https://www.google.com/maps/place/Pantai+Sembilan+Sumenep/@-7.1751703,113.919241,17z/data=!4m8!3m7!1s0x2dd759ba4659b12b:0x5818009169d7abb7!8m2!3d-7.1751703!4d113.9218159!9m1!1b1!16s%2Fg%2F11c5339dr4?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"}
                
                # Mengirimkan POST request ke API FastAPI
                response = requests.post(api_url, json=payload)
                
                # Memeriksa status dan menampilkan hasilnya
                if response.status_code == 200:
                    result = response.json()
                    reviews = result.get('reviews', [])
                    print("Review Results:")
                    for review in reviews:
                        print(review)
                else:
                    print("Error:", response.status_code)
                reviews = response.json().get('reviews', [])
                    # Mengambil 5 data pertama dari kolom 'ulasan'
                top_10_reviews = reviews[-5:]
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("Air Terjun Toroan"):
            try:
                import requests
            
                # URL dari selenium di vps
                api_url = "http://157.66.54.50:8501/scrape"
                
                # URL yang ingin di-scrape
                payload = {"url": "https://www.google.com/maps/place/Air+Terjun+Toroan/@-6.8928844,113.3097483,17z/data=!4m8!3m7!1s0x2e0518178cebfebb:0xefcf0aa128f79400!8m2!3d-6.8928897!4d113.3123232!9m1!1b1!16s%2Fg%2F11b6pkts1w?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"}
                
                # Mengirimkan POST request ke API FastAPI
                response = requests.post(api_url, json=payload)
                
                # Memeriksa status dan menampilkan hasilnya
                if response.status_code == 200:
                    result = response.json()
                    reviews = result.get('reviews', [])
                    print("Review Results:")
                    for review in reviews:
                        print(review)
                else:
                    print("Error:", response.status_code)
                reviews = response.json().get('reviews', [])
                    # Mengambil 5 data pertama dari kolom 'ulasan'
                top_10_reviews = reviews[-5:]
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        if st.button("Pantai Lombang "):
            try:
                import requests
            
                # URL dari selenium di vps
                api_url = "http://157.66.54.50:8501/scrape"
                
                # URL yang ingin di-scrape
                payload = {"url": "https://www.google.com/maps/place/Pantai+Lombang/@-6.9178648,114.0599177,16z/data=!4m8!3m7!1s0x2dd9f7276ab8c685:0xe6566e3638889a6!8m2!3d-6.9155738!4d114.0586496!9m1!1b1!16s%2Fg%2F112yfp277?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"}
                
                # Mengirimkan POST request ke API FastAPI
                response = requests.post(api_url, json=payload)
                
                # Memeriksa status dan menampilkan hasilnya
                if response.status_code == 200:
                    result = response.json()
                    reviews = result.get('reviews', [])
                    print("Review Results:")
                    for review in reviews:
                        print(review)
                else:
                    print("Error:", response.status_code)
                reviews = response.json().get('reviews', [])
                    # Mengambil 5 data pertama dari kolom 'ulasan'
                top_10_reviews = reviews[-5:]
                
                # Transformasi data ulasan ke fitur
                new_X = vectorizer.transform(top_10_reviews).toarray()
                
                # Membuat dictionary fitur (jika model membutuhkan format dictionary)
                features_list = [
                    {f"feature_{j}": new_X[i][j] for j in range(new_X.shape[1])} 
                    for i in range(new_X.shape[0])
                ]
                
                # Prediksi sentimen untuk setiap ulasan
                predictions = [loaded_model.classify(features) for features in features_list]
                
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen")
                hasil_prediksi = pd.DataFrame({
                    "Ulasan": top_10_reviews,
                    "Prediksi Sentimen": predictions
                })
                st.write(hasil_prediksi)
                # Hitung mayoritas kelas
                from collections import Counter
                class_counts = Counter(predictions)
                majority_class = class_counts.most_common(1)[0][0]  # Kelas dengan frekuensi tertinggi

                st.write("Kelas mayoritas (kesimpulan):", majority_class)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


        
          


        
