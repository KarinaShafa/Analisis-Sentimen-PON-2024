import re
import random
import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------------------
# Data Processing Code 
# --------------------------------------

# Ambil data dari session_state
if "df_sentimen" not in st.session_state:
    st.error("Data belum dimuat! Silakan jalankan sentimen_app.py terlebih dahulu.")
else:
    df_sentimen = st.session_state.df_sentimen

# Ambil filter dari session_state
label_sentimen = st.session_state.get("label_sentimen")
top_number = st.session_state.get("top_number")

# Warna untuk masing-masing kelas sentimen
custom_colors = {'Negatif': '#FF5959', 'Netral': '#FACF5A', 'Positif': '#4F9DA6'}

# Menghitung jumlah data pada masing-masing kelas sentimen
def hitung_jumlah(df):
    total = df.shape[0]
    sentimen_counts = df['Sentimen'].value_counts()
    positif_count = sentimen_counts.get('Positif', 0)
    netral_count = sentimen_counts.get('Netral', 0)
    negatif_count = sentimen_counts.get('Negatif', 0)

    return total, positif_count, netral_count, negatif_count

def plot_sentiment_bar(df, custom_colors=custom_colors):
    """
    Menampilkan horizontal bar chart dengan label 'Sentimen (Persentase%)' di dalam batang.

    Parameters:
    - df: DataFrame yang memiliki kolom 'Sentimen'
    - custom_colors: Dict warna untuk tiap label sentimen

    Returns:
    - fig: Figure dari plotly.express
    - sentimen_counts: Data ringkasan
    """
    # Hitung jumlah dan persentase
    sentimen_counts = df['Sentimen'].value_counts().reset_index()
    sentimen_counts.columns = ['Sentimen', 'Jumlah']
    total = sentimen_counts['Jumlah'].sum()
    sentimen_counts['Persentase'] = sentimen_counts['Jumlah'] / total * 100

    # Format label: Sentimen (xx.x%)
    sentimen_counts['Label'] = sentimen_counts.apply(
        lambda row: f"{row['Sentimen']} ({row['Persentase']:.1f}%)", axis=1
    )

    # Buat bar chart horizontal
    fig = px.bar(
        sentimen_counts,
        x='Jumlah',
        y='Sentimen',
        orientation='h',
        color='Sentimen',
        text='Label',
        color_discrete_map=custom_colors
    )

    # Konfigurasi tampilan
    fig.update_traces(
        textposition='inside',
        insidetextanchor='start',
        textfont_size=18
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig, sentimen_counts

def plot_sentiment_wordcloud(df, label_sentimen, column='swremove_text', custom_colors=custom_colors):
    """
    Mengembalikan WordCloud untuk tweet berdasarkan label sentimen tertentu
    dalam bentuk matplotlib figure untuk ditampilkan di Streamlit.

    Parameters:
    - df: DataFrame yang berisi tweet dan label sentimen
    - label_sentimen: Label sentimen yang ingin dianalisis ('Negatif', 'Positif', 'Netral')
    - column: Kolom teks token yang akan digunakan (default: 'swremove_text')
    - custom_colors: Dict warna per sentimen (default: None)

    Returns:
    - fig (matplotlib.figure.Figure): Figure matplotlib dari WordCloud
    """

    # Filter data
    filtered_df = df[df['Sentimen'] == label_sentimen][column]

    # Bersihkan teks token (hapus tanda kurung siku, kutip, koma)
    cleaned_text = filtered_df.apply(lambda x: re.sub(r"[\[\]']", '', x))  # koma dibiarkan agar tetap memisahkan kata

    # Gabungkan menjadi satu string besar
    text = ' '.join(cleaned_text.astype(str))

    # Buat WordCloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        contour_color='black',
        contour_width=1
    ).generate(text)

    # Pewarnaan sesuai sentimen
    wordcloud.recolor(color_func=lambda *args, **kwargs: custom_colors.get(label_sentimen, '#000000'))

    # Buat figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    return fig

def plot_sentiment_pie_chart(df, custom_colors=custom_colors):
    """
    Menampilkan pie chart distribusi sentimen dari DataFrame dengan hole (donut chart).

    Parameters:
    - df (pd.DataFrame): DataFrame yang sudah memiliki kolom 'Sentimen' dengan nilai 'Positif', 'Netral', atau 'Negatif'.
    - custom_colors (dict): Dictionary yang memetakan nama sentimen ke kode warna hex.

    Returns:
    - fig (plotly.graph_objects.Figure): Visualisasi pie chart dengan hole.
    - sentimen_counts (pd.DataFrame): Data jumlah masing-masing kategori sentimen.
    """
    # Hitung jumlah per sentimen
    sentimen_counts = df['Sentimen'].value_counts().reset_index()
    sentimen_counts.columns = ['Sentimen', 'Jumlah']

    # Plot pie chart
    fig = px.pie(
        sentimen_counts,
        names='Sentimen',
        values='Jumlah',
        color='Sentimen',
        color_discrete_map=custom_colors,
        hole=0.4  # Membuat hole pada pie chart (donut chart)
    )

    # Menampilkan label persentase dan nama sentimen
    fig.update_traces(textinfo='percent+label')

    # Menyembunyikan legend dan mengatur layout
    fig.update_layout(
        showlegend=False,  # Menyembunyikan legend
        plot_bgcolor='white'
    )

    return fig, sentimen_counts

def plot_ngram_frequencies(df, sentimen_label='Positif', column='swremove_text', n=1, top_n=20):

    # Tentukan sentimen yang akan digunakan
    sentimen_list = ['Positif', 'Netral', 'Negatif'] if sentimen_label == 'All' else [sentimen_label]

    # List untuk simpan hasil
    all_data = []

    for label in sentimen_list:
        sub_df = df[df['Sentimen'] == label][column]
        cleaned = sub_df.apply(lambda x: re.sub(r"[\[\]',]", '', x) if isinstance(x, str) else x)
        text = ' '.join(cleaned.astype(str))
        tokens = text.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngram_list = [' '.join(ng) for ng in ngrams]
        counts = Counter(ngram_list)
        for ngram, freq in counts.items():
            all_data.append({'n-gram': ngram, 'Frekuensi': freq, 'Sentimen': label})

    ngram_df = pd.DataFrame(all_data)

    # Ambil top_n n-gram berdasarkan total frekuensi
    top_ngrams = (
        ngram_df.groupby('n-gram')['Frekuensi']
        .sum().nlargest(top_n).index
    )
    ngram_df = ngram_df[ngram_df['n-gram'].isin(top_ngrams)]

    # Tentukan urutan y berdasarkan total frekuensi (dari besar ke kecil)
    y_order = (
        ngram_df.groupby('n-gram')['Frekuensi']
        .sum().sort_values(ascending=False).index.tolist()
    )

    # Gunakan nama kolom dinamis
    y_col = f'{n}-gram' if sentimen_label != 'All' else 'n-gram'
    ngram_df[y_col] = ngram_df['n-gram']

    # Buat grafik
    # Buat grafik
    fig = px.bar(
        ngram_df,
        x='Frekuensi',
        y=y_col,
        color='Sentimen',
        orientation='h',
        text='Frekuensi',
        color_discrete_map=custom_colors,
        category_orders={y_col: y_order}
    )

    fig.update_traces(textposition='inside')
    fig.update_layout(
        barmode='stack' if sentimen_label == 'All' else 'relative',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(
            title='',       # Ini menyembunyikan judul y-axis
            showticklabels=True,  # Pastikan label tetap tampil
            ticks='outside'       # Opsional, jika ingin tampilan rapi
        ),
        margin=dict(l=0, r=0, t=5, b=0),
        autosize=True
    )

    return ngram_df, fig

def plot_top_users_by_sentiment(df, user_column='username', sentiment_column='Sentimen', top_n=10, sentiment_filter='All'):
    """
    Menampilkan diagram stacked bar pengguna yang paling sering muncul beserta distribusi sentimennya.
    """
    # Filter jika hanya ingin satu jenis sentimen
    if sentiment_filter != 'All':
        df = df[df[sentiment_column] == sentiment_filter]

    # Ambil top_n pengguna berdasarkan jumlah tweet
    top_users = df[user_column].value_counts().head(top_n).index
    filtered_df = df[df[user_column].isin(top_users)]

    # Hitung jumlah sentimen per pengguna
    user_sentiment_counts = filtered_df.groupby([user_column, sentiment_column]).size().reset_index(name='Jumlah')

    # Tambahkan '@' ke nama pengguna
    user_sentiment_counts[user_column] = '@' + user_sentiment_counts[user_column].astype(str)

    # Pivot dan isi NaN
    pivot_df = user_sentiment_counts.pivot(index=user_column, columns=sentiment_column, values='Jumlah').fillna(0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=True).drop(columns='Total').reset_index()

    # Melt untuk keperluan plotly
    melted_df = pivot_df.melt(id_vars=user_column, var_name='Sentimen', value_name='Jumlah')

    # Plot
    fig = px.bar(
        melted_df,
        x='Jumlah',
        y=user_column,
        color='Sentimen',
        orientation='h',
        title='',
        color_discrete_map=custom_colors,
        text='Jumlah'
    )

    # Update layout: sembunyikan axis tapi tampilkan label y (usernames)
    fig.update_layout(
        barmode='stack' if sentiment_filter == 'All' else 'group',
        yaxis=dict(
            title='',
            showticklabels=True,
            ticks='outside',
            showline=False,
            showgrid=False
        ),
        xaxis=dict(
            visible=False  # x-axis disembunyikan sepenuhnya
        ),
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False
    )

    fig.update_traces(
        textposition='inside',
        insidetextanchor='start',
        cliponaxis=False,
        textfont_size=12
    )

    return fig, user_sentiment_counts

def plot_top_mentions_by_sentiment(df, mention_column='mention', sentiment_column='Sentimen',
                                   sentimen_filter='All', top_n=10):
    """
    Menampilkan diagram bar pengguna yang paling sering dimention dari kolom string mention,
    dengan pembersihan simbol terlebih dahulu.
    """
    # Filter berdasarkan sentimen jika tidak memilih 'All'
    if sentimen_filter != 'All':
        df = df[df[sentiment_column] == sentimen_filter]

    # Bersihkan teks token (hapus tanda kurung siku, kutip, koma)
    cleaned_text = df[mention_column].apply(lambda x: re.sub(r"[\[\]',]", '', x) if isinstance(x, str) else '')

    # Ekstrak mention dan sentimennya
    mention_data = []
    for text, sentimen in zip(cleaned_text, df[sentiment_column]):
        mentions = text.split()
        for mention in mentions:
            if mention.strip():  # Abaikan kosong
                mention_data.append((mention.strip(), sentimen))

    if not mention_data:
        return None, pd.DataFrame()

    # Buat DataFrame dan hitung frekuensi
    mention_df = pd.DataFrame(mention_data, columns=['Mention', 'Sentimen'])
    mention_counts = mention_df.groupby(['Mention', 'Sentimen']).size().reset_index(name='Jumlah')

    # Ambil top_n mention berdasarkan total
    top_mentions = (mention_counts.groupby('Mention')['Jumlah']
                    .sum().sort_values(ascending=False).head(top_n).index)

    # Filter hanya top_n mention
    mention_counts = mention_counts[mention_counts['Mention'].isin(top_mentions)]

    # Tambahkan '@' di depan mention
    mention_counts['Mention'] = mention_counts['Mention'].astype(str)

    # Urutan mention dari paling banyak ke sedikit
    mention_order = (mention_counts.groupby('Mention')['Jumlah']
                     .sum().sort_values(ascending=False).index)

    # Buat plot
    fig = px.bar(
        mention_counts,
        x='Jumlah',
        y='Mention',
        color='Sentimen',
        orientation='h',
        title='',
        color_discrete_map=custom_colors,
        category_orders={'Mention': list(mention_order)}
    )

    fig.update_layout(
        barmode='stack',
        xaxis=dict(visible=False),
        yaxis=dict(title='', showticklabels=True, ticks='outside'),
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False
    )
    fig.update_traces(
        texttemplate='%{x}',
        textposition='inside',
        insidetextanchor='start',
        cliponaxis=False
    )

    return fig, mention_counts


def plot_hashtag_wordcloud_by_sentiment(df, sentiment_filter='All', column='hashtag', 
                                        custom_colors=None, width=800, height=400):
    """
    Menggabungkan tokenisasi hashtag dan visualisasi WordCloud berdasarkan sentimen.

    Parameters:
    - df: DataFrame
    - sentiment_filter: 'All' atau salah satu dari ['Positif', 'Negatif', 'Netral']
    - column: Nama kolom hashtag
    - custom_colors: Dict warna berdasarkan sentimen (misal: {'Positif': 'green', ...})
    - width, height: Ukuran wordcloud

    Returns:
    - fig: matplotlib.figure
    - df_freq: DataFrame frekuensi hashtag
    """

    # Filter berdasarkan sentimen
    if sentiment_filter != 'All':
        filtered_df = df[df['Sentimen'] == sentiment_filter]
        color = custom_colors.get(sentiment_filter, 'black') if custom_colors else 'black'
    else:
        filtered_df = df.copy()
        color = None

    # Tokenisasi hashtag (mirip fungsi anjascek)
    all_tokens = []
    for item in filtered_df[column]:
        if isinstance(item, list):
            tokens = [str(tag).strip().lower() for tag in item if isinstance(tag, str)]
            all_tokens.extend(tokens)
        elif isinstance(item, str):
            cleaned = re.sub(r"[\[\]',]", '', item)
            tokens = [tag.strip().lower() for tag in cleaned.split() if tag.strip()]
            all_tokens.extend(tokens)

    # Hitung frekuensi
    hashtag_counts = Counter(all_tokens)
    if not hashtag_counts:
        return None, pd.DataFrame(columns=['Hashtag', 'Frekuensi'])

    # Warna dinamis
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if sentiment_filter != 'All':
            return color
        else:
            return random.choice(list(custom_colors.values())) if custom_colors else 'black'

    # Buat WordCloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        color_func=color_func,
        prefer_horizontal=1.0,
        collocations=False,
        min_font_size=10
    ).generate_from_frequencies(hashtag_counts)

    # Visualisasi matplotlib
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    # DataFrame frekuensi
    df_freq = pd.DataFrame(hashtag_counts.items(), columns=['Hashtag', 'Frekuensi']).sort_values(by='Frekuensi', ascending=False)

    return fig, df_freq


# --------------------------------------
# Streamlit UI Code 
# --------------------------------------

st.title("Dashboard Analisis Sentimen Pelaksanaan PON 2024")

# Warna untuk masing-masing kelas sentimen
custom_colors = {'Negatif': '#FF5959', 'Netral': '#FACF5A', 'Positif': '#4F9DA6'}

col1a, col2a = st.columns([1,3], border=True)
with col1a:
    total_data, jumlah_positif, jumlah_netral, jumlah_negatif = hitung_jumlah(df_sentimen)
    st.metric("Total Data", f"📊 {total_data:,}", border=True)
    st.metric("Sentimen Positif", f"😊 {jumlah_positif:,}", border=True)
    st.metric("Sentimen Netral", f"😐 {jumlah_netral:,}", border=True)
    st.metric("Sentimen Negatif", f"😠 {jumlah_negatif:,}", border=True)
with col2a:
    st.subheader("Distribusi Sentimen")
    fig_bar, df_bar_counts = plot_sentiment_bar(df_sentimen)
    st.plotly_chart(fig_bar, use_container_width=True)

with st.container(border=True):
    st.subheader("Wordcloud pada Masing-Masing Sentimen")
    
    # Membuat kolom untuk menampilkan WordCloud
    col1b, col2b, col3b = st.columns(3, border=True)
    with col1b:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Sentimen Negatif</div>", unsafe_allow_html=True)
        fig_wc_neg = plot_sentiment_wordcloud(df_sentimen, 'Negatif')
        st.pyplot(fig_wc_neg, use_container_width=True)
    with col2b:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Sentimen Netral</div>", unsafe_allow_html=True)
        fig_wc_net = plot_sentiment_wordcloud(df_sentimen, 'Netral')
        st.pyplot(fig_wc_net, use_container_width=True)    
    with col3b:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Sentimen Positif</div>", unsafe_allow_html=True)
        fig_wc_pos = plot_sentiment_wordcloud(df_sentimen, 'Positif')
        st.pyplot(fig_wc_pos, use_container_width=True)

with st.container(border=True):
    st.subheader("Frekuensi Penggunaan Kata Berdasarkan Sentimen")
    
    # Membuat kolom
    col1c, col2c, col3c = st.columns(3, border=True)
    with col1c:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Unigram (1-kata)</div>", unsafe_allow_html=True)
        ngram_df_1, fig_1 = plot_ngram_frequencies(df_sentimen, sentimen_label=label_sentimen, n=1, top_n=top_number)
        st.plotly_chart(fig_1, use_container_width=True)
    with col2c:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Bigram (2-kata)</div>", unsafe_allow_html=True)  
        ngram_df_2, fig_2 = plot_ngram_frequencies(df_sentimen, sentimen_label=label_sentimen, n=2, top_n=top_number)
        st.plotly_chart(fig_2, use_container_width=True)
    with col3c:
        st.markdown("<div style='text-align: center; font-size: 1rem'>Frekuensi Trigram (3-kata)</div>", unsafe_allow_html=True)
        ngram_df_3, fig_3 = plot_ngram_frequencies(df_sentimen, sentimen_label=label_sentimen, n=3, top_n=top_number)
        st.plotly_chart(fig_3, use_container_width=True)

# with st.container(border=True):
#     st.subheader("Pengguna Paling Aktif & Sering Dimention Berdasarkan Sentimen")
    
#     # Membuat kolom
#     col1d, col2d = st.columns(2, border=True)
#     with col1d:
#         st.markdown("<div style='text-align: center; font-size: 1rem'>Pengguna Paling Aktif</div>", unsafe_allow_html=True)
#         fig_aktif, df_user_sentiment = plot_top_users_by_sentiment(df_sentimen, top_n=top_number, sentiment_filter=label_sentimen)
#         st.plotly_chart(fig_aktif, use_container_width=True)
#     with col2d:
#         st.markdown("<div style='text-align: center; font-size: 1rem'>Pengguna Sering Dimention</div>", unsafe_allow_html=True)
#         fig_mention, mention_df = plot_top_mentions_by_sentiment(df_sentimen, sentimen_filter=label_sentimen, top_n=top_number)
#         st.plotly_chart(fig_mention, use_container_width=True)

with st.container(border=True):
    st.subheader("Hashtag Terpopuler Berdasarkan Sentimen")
    with st.container(border=True):
        col1e, col2e = st.columns([3,1], vertical_alignment="center")
        with col1e:
            fig_hastag, freq_df = plot_hashtag_wordcloud_by_sentiment(df_sentimen, sentiment_filter=label_sentimen, column='hashtag', custom_colors=custom_colors)
            st.pyplot(fig_hastag, use_container_width=True)
        with col2e:
            st.dataframe(freq_df, hide_index=True, use_container_width=True)

with st.container(border=True):
    st.subheader("Pencarian Data")

    # Input pencarian
    search_text = st.text_input("🔍 Cari Teks atau Username:", 
                                placeholder="Masukkan kata kunci atau username...")

    # Salin dan siapkan dataframe
    filtered_df = df_sentimen.copy()

    # Filter berdasarkan teks pencarian (dari kolom 'full_text' atau 'username')
    if search_text:
        filtered_df = filtered_df[
            filtered_df['full_text'].str.contains(search_text, case=False, na=False) |
            filtered_df['username'].str.contains(search_text, case=False, na=False)
        ]

        # Menentukan jenis pesan berdasarkan jumlah hasil
        result_count = len(filtered_df)
        if result_count == 0:
            st.info("Tidak ada data yang cocok ditemukan.")
        else:
            st.success(f"Ditemukan {result_count} data yang cocok.")

    # Pilih dan ubah nama kolom yang ingin ditampilkan
    display_df = filtered_df[['username', 'full_text', 'Sentimen']].rename(
        columns={
            'username': 'Username',
            'full_text': 'Full Text'
        }
    )

    # Tampilkan dalam Streamlit
    st.dataframe(display_df, hide_index=True, use_container_width=True,
                 column_config={"Username": st.column_config.Column(width="small"),
                                "Full Text": st.column_config.Column(width="large"),
                                "Sentimen": st.column_config.Column(width="small")})



