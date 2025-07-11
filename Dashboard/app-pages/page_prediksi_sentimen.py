import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import time

# --------------------------------------
# Data Processing Code 
# --------------------------------------

# Ambil model dari session_state
if "vectorizer" not in st.session_state or "model" not in st.session_state:
    st.error("Data belum dimuat! Silakan jalankan sentimen_app.py terlebih dahulu.")
else:
    vectorizer = st.session_state.vectorizer
    model = st.session_state.model

# Palet warna berdasarkan kelas sentimen
custom_colors = {'Negatif': '#FF5959', 'Netral': '#FACF5A', 'Positif': '#4F9DA6'}

# Peta label numerik ke string
label_mapping = {-1: "Negatif", 0: "Netral", 1: "Positif"}

# Dictionary normalisasi
normalization_dict = {
    "aja": "saja", "aj": "saja", "bbrp": "beberapa", "bgt": "banget",
    "bgtu": "begitu", "bikin": "membuat", "bkin": "bikin", "blm": "belum",
    "brp": "berapa", "bs": "bisa", "btw": "omong-omong", "dgn": "dengan",
    "dlm": "dalam", "dpt": "dapat", "dr": "dari", "dah": "sudah", "emang": "memang",
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "kalo": "kalau",
    "klu": "kalau", "klo": "kalau", "km": "kamu", "kmrn": "kemarin",
    "krn": "karena", "liat": "lihat", "lg": "lagi", "lho": "loh",
    "makasih": "terima kasih", "mksh": "terima kasih", "nah": "",
    "ngga": "tidak", "nggak": "tidak", "nih": "ini", "ny": "nya",
    "ok": "oke", "oke": "oke", "okey": "oke", "org": "orang",
    "pdhl": "padahal", "pls": "tolong", "sampe": "sampai",
    "sdh": "sudah", "sih": "", "sm": "sama", "smua": "semua",
    "sy": "saya", "td": "tadi", "tdk": "tidak", "thx": "terima kasih",
    "tp": "tapi", "trs": "terus", "udh": "sudah", "udah": "sudah",
    "utk": "untuk", "y": "ya", "yaampun": "ya ampun", "yg": "yang",
}

# Custom stopwords
custom_stopwords = set([
    "dan", "atau", "tetapi", "serta", "lalu", "kemudian", "namun", "sehingga", "agar",
    "di", "ke", "dari", "pada", "dengan", "tanpa", "untuk", "bagi", "dalam", "antara",
    "itu", "ini", "sebuah", "seorang", "para", "sang",
    "yang", "apa", "siapa", "mana",
    "lah", "kah", "pun", "dong", "deh", "loh", "kok", "ya", "nah", "hmm", "oh", "eh",
    "sekarang", "kemarin", "besok", "nanti",
    "saja", "hanya", "bahkan"
])

# Membuat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk Melakukan Preprocessing pada teks input
def text_preparation(text):
    """
    Membersihkan dan memproses teks: cleaning, case folding, normalisasi,
    tokenisasi, stopword removal, stemming, dan menghapus kata satu huruf.

    Parameters:
    text (str): Teks mentah yang akan diproses.

    Returns:
    str: Teks yang telah diproses sepenuhnya.
    """

    # 1. Cleaning
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)   # Hapus URL
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)         # Hapus HTML entities
    text = re.sub(r'<[^>]+>', ' ', text)                 # Hapus tag HTML
    text = re.sub(r'(?<=\w)\.(?=\w)', ' ', text)         # Tambah spasi di antara huruf dan titik
    text = text.replace('\xa0', ' ')                     # Hapus karakter non-breaking space
    emoji_pattern = re.compile(                          # Pola untuk menghapus emoji
        "[" u"\U0001F1E0-\U0001F1FF"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F600-\U0001F64F"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)                  # Hapus emoji
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)          # Hapus mention
    text = re.sub(r'#\w+', ' ', text)                    # Hapus hashtag
    text = re.sub(r'^RT[\s]+', '', text)                 # Hapus retweet (RT)
    text = re.sub(r'[0-9]', ' ', text)                   # Hapus angka
    text = re.sub(r'[^A-Za-z ]', ' ', text)              # Hapus karakter non-alfabet
    text = re.sub(r'[\n\r]', ' ', text)                  # Hapus karakter newline
    text = re.sub(r'\s+', ' ', text).strip()             # Normalisasi spasi ganda

    # 2. Case folding
    text = text.lower()                                  # Ubah teks menjadi huruf kecil

    # 3. Normalisasi kata tidak baku
    for key, value in normalization_dict.items():        # Ganti kata sesuai kamus normalisasi
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
    text = re.sub(r'\s+', ' ', text).strip()             # Bersihkan spasi ganda sisa

    # 4. Tokenisasi
    tokens = text.split()                                # Pisahkan teks menjadi kata-kata

    # 5. Stopword removal
    tokens = [word for word in tokens if word not in custom_stopwords]  # Hapus stopword

    # 6. Stemming
    tokens = [stemmer.stem(word) for word in tokens]     # Ubah kata ke bentuk dasar

    # 7. Hapus kata satu huruf
    tokens = [word for word in tokens if len(word) > 1]  # Hapus kata yang hanya satu huruf

    # Gabungkan kembali token menjadi kalimat
    final_text = ' '.join(tokens)

    return final_text


# Fungsi untuk menampilkan progress bar berwarna
def colored_progress(label, value, color):
    st.markdown(f"""
    <div style="margin-bottom: 0.5rem;"><strong>{label}: {value:.2f}%</strong></div>
    <div style="background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px; margin-bottom: 1rem;">
        <div style="width: {value}%; background-color: {color}; height: 100%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------
# Streamlit UI Code 
# --------------------------------------

st.title("Prediksi Sentimen")

# Layout: input teks + tombol sejajar
col1, col2 = st.columns([6, 1], vertical_alignment="bottom")
with col1:
    user_input = st.text_input(
        "Masukkan teks di sini:",
        placeholder="Contoh: Saya sangat bangga dengan prestasi atlet di PON 2024",
    )

with col2:
    predict_clicked = st.button("Prediksi Sentimen", use_container_width=True)

# Jika tombol diklik dan input tidak kosong
if predict_clicked:
    if user_input.strip():
        clean_input = text_preparation(user_input)

        with st.spinner("Memproses prediksi, mohon tunggu..."):
            time.sleep(1.5)

            # Transformasi dan prediksi
            vectorized_input = vectorizer.transform([clean_input])
            predicted_label_num = model.predict(vectorized_input)[0]
            predicted_label = label_mapping[predicted_label_num]

            # Ambil skor probabilitas jika tersedia
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vectorized_input)[0]
                class_labels = model.classes_
                confidence_scores = {
                    label_mapping[int(lbl)]: round(prob * 100, 2)
                    for lbl, prob in zip(class_labels, probas)
                }
            else:
                confidence_scores = {predicted_label: 100.0}

            # Warna latar belakang untuk label
            bg_color = custom_colors.get(predicted_label, "#4F9DA6")

            # Tampilkan hasil prediksi
            st.markdown(f"""
                <div style="background-color: {bg_color}; border-radius: 10px; padding: 1.2rem; text-align: center; line-height: 1.2; margin-bottom: 2rem">
                    <div style="color: white; font-weight: bold; font-size: 1.5rem;">Prediksi Sentimen:</div>
                    <div style="color: white; font-size: 2.5rem; font-weight: bold; margin-top: 0.3rem;">{predicted_label}</div>
                </div>
            """, unsafe_allow_html=True)

            # Confidence score
            sorted_scores = dict(sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True))

            st.subheader("📊 Confidence Score per Kelas", help="Confidence Score menunjukkan tingkat keyakinan model terhadap setiap kelas. Semakin tinggi nilainya, semakin yakin model terhadap prediksi tersebut.")
            for label, score in sorted_scores.items():
                colored_progress(label, score, custom_colors[label])
    else:
        st.warning("Silakan masukkan teks terlebih dahulu sebelum memprediksi.")
