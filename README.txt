# Sentimen PON Dashboard

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun dashboard interaktif yang dapat memprediksi sentimen dari teks yang berhubungan dengan Pekan Olahraga Nasional (PON) 2024. Model yang digunakan untuk prediksi adalah model klasifikasi sentimen berbasis Support Vector Machine (SVM), yang telah dilatih menggunakan data terkait sentimen dari media sosial.

## Struktur Folder
📦Sentimen PON  
 ┣ 📂Dashboard  
 ┃ ┣ 📂app-pages  
 ┃ ┃ ┣ 📜page_dashboard_sentimen.py  - Halaman utama dashboard yang menampilkan ringkasan sentimen terkait PON.  
 ┃ ┃ ┗ 📜page_prediksi_sentimen.py - Halaman untuk memprediksi sentimen dari input teks pengguna.  
 ┃ ┗ 📜sentimen_app.py - File utama yang menjalankan aplikasi Streamlit dan menghubungkan semua halaman.  
 ┣ 📂Data  
 ┃ ┗ 📜data_baru_analisis.csv - Dataset terbaru untuk hasil preprocessing di python notebook.  
 ┣ 📂Model  
 ┃ ┣ 📜saved_svm_model.pkl - Model SVM yang sudah dilatih dan disimpan untuk prediksi sentimen.  
 ┃ ┗ 📜saved_tfidf_vectorizer.pkl - Vectorizer TF-IDF yang digunakan untuk mengubah teks input menjadi fitur yang dapat diproses oleh model.  
 ┣ 📂Python Notebook  
 ┃ ┗ 📜Python Notebook_Sentimen Pon 2024 SVM.ipynb - Notebook yang digunakan untuk eksplorasi data dan pelatihan model.  
 ┃ 📜README.txt - File ini.
 ┗ 📜requirement.txt - Berisi daftar dan versi library yang digunakan pada proyek.

## Instalasi dan Pengaturan
1. **Instalasi Dependensi:**  
Pastikan Anda sudah menginstal semua dependensi yang dibutuhkan (cek file requirement.txt), Daftar dependensi utama:
- `pandas` (untuk manipulasi data)
- `scikit-learn` (untuk pemrosesan ML)
- `sastrawi` (untuk stemming bahasa Indonesia)
- `matplotlib` (untuk visualisasi dasar)
- `plotly` (untuk visualisasi interaktif)
- `wordcloud` (untuk visualisasi teks)
- `streamlit` (untuk antarmuka web aplikasi)


2. **Menjalankan Aplikasi:**  
> Untuk menjalankan aplikasi:
- Buka terminal/command prompt
- Navigasi ke direktori utama proyek
- Jalankan perintah: ``streamlit run Dashboard/sentimen_app.py`` Aplikasi akan otomatis terbuka di browser default (biasanya di http://localhost:8501)

> Untuk menghentikan server Streamlit:
- Kembali ke terminal tempat aplikasi dijalankan
- Tekan kombinasi tombol: Windows/Linux: ``Ctrl + C`` | MacOS: ``Command + C``
- Tunggu hingga proses benar-benar berhenti (terminal akan menampilkan pesan "Server stopped")

3. **Memuat Model dan Data:**  
Aplikasi ini sudah terintegrasi dengan model SVM yang disimpan dalam format `.pkl` bersama dengan TF-IDF vectorizer untuk pemrosesan teks. Pastikan Anda menyimpan file `saved_svm_model.pkl` dan `saved_tfidf_vectorizer.pkl` di dalam folder `Model`.


4. **Penggunaan:**  
- **Halaman Dashboard:** Menampilkan informasi visualisasi dan statistik terkait sentimen PON 2024 (Disertai filter yang dapat digunakan).
- **Halaman Prediksi Sentimen:** Pengguna dapat memasukkan teks dan mengklik tombol "Prediksi Sentimen". Aplikasi akan memberikan hasil prediksi sentimen dan confidence score untuk setiap kelas (Positif, Netral, Negatif).