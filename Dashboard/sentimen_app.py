import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pickle

# Konfigurasi awal Streamlit
st.set_page_config(page_title="Analisis Sentimen Pelaksanaan PON 2024", page_icon="📊", layout="wide")

# --------------------------------------
# Data Processing Code 
# --------------------------------------

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Data/data_baru_analisis.csv")
    return df

# Load model
@st.cache_resource 
def load_model():
    with open("Model/saved_tfidf_vectorizer.pkl", "rb") as f_vec, open("Model/saved_svm_model.pkl", "rb") as f_model:
        vectorizer = pickle.load(f_vec)
        model = pickle.load(f_model)
    return vectorizer, model

# Inisialisasi session_state untuk data
if "df_sentimen" not in st.session_state:
    st.session_state.df_sentimen = load_data()

# Inisialisasi session_state untuk model
if "vectorizer" not in st.session_state or "model" not in st.session_state:
    vectorizer, model = load_model()
    st.session_state.vectorizer = vectorizer
    st.session_state.model = model

# Inisialisasi session_state untuk filter
if "label_sentimen" not in st.session_state:
    st.session_state.label_sentimen = "All"

if "top_number" not in st.session_state:
    st.session_state.top_number = 20

# Gunakan data dari session state, tanpa memuat ulang
df_sentimen = st.session_state.df_sentimen

# --------------------------------------
# Streamlit UI Code 
# --------------------------------------

# Page Setup
pages = {
    "Menu Navigasi": [
        st.Page(page="app-pages/page_dashboard_sentimen.py", title="Dashboard", icon="📊", default=True),
        st.Page(page="app-pages/page_prediksi_sentimen.py", title="Prediksi Sentimen", icon="🤖")
    ]
}

# Navigation setup
pg = st.navigation(pages)

# Global filter
# Sidebar: Pilih filter
st.sidebar.subheader("🔍 Filters")

# Sidebar: Filter label sentimen
sentimen_cat = ["All"] + sorted(df_sentimen["Sentimen"].unique())
label_sentimen = st.sidebar.selectbox("Jenis Sentimen:", options=sentimen_cat, index=0)

# Sidebar: Filter top data
top_number = st.sidebar.number_input("Jumlah Data Teratas", min_value=5, max_value=50, value=20)

# Update session_state jika ada perubahan & refresh halaman
if (
    label_sentimen != st.session_state.label_sentimen or
    top_number != st.session_state.top_number
):
    st.session_state.label_sentimen = label_sentimen
    st.session_state.top_number = top_number
    st.rerun()  # Refresh agar filter berlaku


pg.run()
