import streamlit as st
import pandas as pd
import zipfile
import os
import joblib
from collections import deque
import math
import urllib.parse
import re

# Konfigurasi halaman
st.set_page_config(initial_sidebar_state="expanded")

# --- Konfigurasi Batas Tampilan ---
DISPLAY_LIMIT = 10

# --- Konfigurasi Path ---
DATA_DIR = "data"
ZIP_FILE_NAME = "Dataset.zip"
SVM_MODEL_GENRE = "svm_model.pkl"
SVM_MODEL_TAG = "svm_model_tags.pkl"
SVM_MODEL_CATEGORY = "svm_model_categories.pkl"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/180x100.png?text=No+Image"

# --- Custom CSS ---
# Blok CSS final untuk tema terang yang konsisten
hide_streamlit_style = """
    <style>
    /* 1. ATURAN BACKGROUND TERANG */
    .stApp {
        background-color: #F0F2F6; /* Warna background utama (abu-abu terang) */
    }
    .main { 
        padding-left: 300px !important; 
    }

    /* 2. MEMBUAT SEMUA TEKS MENJADI HITAM */
    h1, h2, h3, h4, h5, h6, p, label, li, span,
    .stAlert p
    {
        color: #000000 !important; 
    }

    /* 3. MENYAMAKAN TEMA SELECTBOX MENJADI TERANG */
    div[data-testid="stSelectbox"] > div {
        background-color: #FFFFFF;
        color: #000000 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #FFFFFF;
    }
    div[data-baseweb="popover"] ul li {
        color: #000000 !important;
    }
    div[data-baseweb="popover"] ul li:hover {
        background-color: #F0F2F6;
    }

    /* 4. [PERUBAHAN] PENGECUALIAN UNTUK KARTU GAME */
    /* Memastikan teks di kartu game tetap putih dengan background abu-abu */
    div[style*="background-color: #4A5568"] * {
        color: white !important;
    }
    div[style*="background-color: #4A5568"] p[style*="color: #ccc"] {
        color: #E2E8F0 !important; /* Abu-abu lebih terang untuk kontras */
    }

    /* 5. ATURAN SIDEBAR PUTIH */
    section[data-testid="stSidebar"] {
        visibility: visible !important; 
        display: block !important; 
        width: 300px !important;
        left: 0px !important; 
        transform: none !important; 
        z-index: 9999 !important;
        background-color: #FFFFFF !important;
    }
    
    /* 6. MENYEMBUNYIKAN ELEMEN BAWAAN STREAMLIT */
    #MainMenu, footer, header { visibility: hidden !important; display: none !important; }
    .stAlert { display: none !important; }
    .stApp > header, .css-1lcbmhc, .css-1d391kg, .css-1f198p6 {
        display: none !important; 
        margin-top: 0px !important; 
        padding-top: 0px !important;
    }
    button[data-testid="stSidebarCollapseButton"] { 
        visibility: hidden !important; 
        display: none !important; 
    }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- FUNGSI-FUNGSI ---

@st.cache_data
def load_data():
    if not os.path.exists(DATA_DIR):
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        except Exception as e:
            st.error(f"Error saat mengekstrak file: {e}")
            return pd.DataFrame()

    df = pd.DataFrame()
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".csv") and ("dataset" in file.lower() or "data" in file.lower()):
                try:
                    df = pd.read_csv(os.path.join(root, file), encoding='utf-8')
                    df.columns = [col.strip().lower() for col in df.columns]
                    break
                except Exception as e:
                    st.error(f"Error saat memuat CSV '{file}': {e}")
                    return pd.DataFrame()
        if not df.empty:
            break

    if df.empty:
        st.error("Tidak ada file CSV dataset yang ditemukan.")
        return pd.DataFrame()

    if 'name' in df.columns:
        df.drop_duplicates(subset=['name'], inplace=True, keep='first')

    df['device'] = df.get('device', pd.Series(dtype='str')).fillna('N/A').astype(str).apply(lambda x: x.strip() if x.strip() and x.lower() not in ['nan', 'none'] else 'N/A')

    # --- BLOK KODE YANG DIPERBARUI UNTUK HARGA ---
    def format_price(price_input):
        if pd.isna(price_input): return "Gratis"
        price_str = str(price_input).strip().lower()
        if not price_str or price_str in ['0', '0.0', 'free', 'gratis', 'nan', 'none']: return "Gratis"
        try:
            # Hanya mengubah simbol ke USD dan formatnya, tanpa konversi kurs
            return f"${float(price_str):,.2f}"
        except ValueError:
            return str(price_input).strip()
    df['price'] = df.get('price', pd.Series(dtype='str')).apply(format_price)
    # --- AKHIR BLOK KODE YANG DIPERBARUI ---

    other_cols = {'short description': 'Deskripsi tidak tersedia', 'genre': 'N/A', 'tags': 'N/A', 'categories': 'N/A', 'platforms': 'N/A'}
    for col, default in other_cols.items():
        df[col] = df.get(col, pd.Series(dtype='str')).fillna(default).astype(str).apply(lambda x: x.strip() if x.strip() and x.lower() not in ['nan', 'none'] else default)

    df['header image'] = df.get('header image', pd.Series(dtype='str')).fillna('').apply(lambda x: x if isinstance(x, str) and x.startswith("http") else "")
    df['positive reviews'] = pd.to_numeric(df.get('positive reviews', pd.Series(dtype='float')), errors='coerce').fillna(0)

    return df

@st.cache_resource
def load_svm_models():
    try:
        return (joblib.load(SVM_MODEL_GENRE), joblib.load(SVM_MODEL_TAG), joblib.load(SVM_MODEL_CATEGORY))
    except Exception:
        return None

def get_recommendations_based_on_preferences(data_df):
    history = st.session_state.history
    if any(history.values()):
        df_temp = data_df.copy()
        df_temp["score"] = 0
        if history["genre"]:
            for pref in history["genre"]: df_temp.loc[df_temp["genre"].str.contains(pref, na=False), "score"] += 3
        if history["tag"]:
            for pref in history["tag"]: df_temp.loc[df_temp["tags"].str.contains(pref, na=False), "score"] += 2
        if history["category"]:
            for pref in history["category"]: df_temp.loc[df_temp["categories"].str.contains(pref, na=False), "score"] += 1
        return df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False).head(DISPLAY_LIMIT)
    return pd.DataFrame()

def display_game_card(game_row):
    nama = game_row.get('name', 'N/A')
    short_description = game_row.get('short description', '')
    price = game_row.get('price', 'N/A')
    device = game_row.get('device', 'N/A')
    gambar = game_row.get('header image', '') or PLACEHOLDER_IMAGE

    genres = ", ".join(g.strip() for g in str(game_row.get('genre', '')).split(',') if g.strip()) or 'N/A'
    tags = ", ".join(t.strip() for t in str(game_row.get('tags', '')).split(',') if t.strip()) or 'N/A'
    kategoris = ", ".join(k.strip() for k in str(game_row.get('categories', '')).split(',') if k.strip()) or 'N/A'

    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(f'{nama} gameplay')}"

    # --- [PERUBAHAN] Mengubah background-color kartu game menjadi abu-abu ---
    st.markdown(f"""
    <div style="display: flex; gap: 20px; padding: 15px; border: 1px solid #718096; border-radius: 10px; margin-bottom: 20px; background-color: #4A5568; color: white;">
        <div style="flex-shrink: 0;"><img src="{gambar}" style="width: 180px; height: 100%; border-radius: 10px; object-fit: cover;"></div>
        <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; margin-bottom: 5px;">{nama}</h4>
                <p style="font-size: 14px; margin-bottom: 10px; color: #ccc;">{short_description}</p>
                <p style="font-size: 13px; line-height: 1.6;">
                    <strong>Genre:</strong> {genres} <br> <strong>Tags:</strong> {tags} <br> <strong>Kategori:</strong> {kategoris} <br>
                    <strong>Price:</strong> {price} <br> <strong>Device:</strong> {device}
                </p>
            </div>
            <div style="margin-top: 15px; text-align: right;">
                <a href="{youtube_url}" target="_blank" style="text-decoration: none; color: white; background-color: #c4302b; padding: 8px 16px; border-radius: 5px; font-size: 14px; font-weight: bold;">🎬 Lihat Gameplay</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if nama != 'N/A' and nama not in st.session_state.viewed_games:
        st.session_state.viewed_games.append(nama)

def display_recommendations(recs_df):
    if recs_df.empty:
        st.info("Tidak ada game yang ditemukan berdasarkan kriteria ini.")
    else:
        for _, row in recs_df.loc[:, ~recs_df.columns.duplicated()].iterrows():
            display_game_card(row)

# --- STRUKTUR UTAMA APLIKASI ---

df = load_data()
models = load_svm_models()

if "history" not in st.session_state:
    st.session_state.history = {"genre": [], "tag": [], "category": []}
if "viewed_games" not in st.session_state:
    st.session_state.viewed_games = deque(maxlen=DISPLAY_LIMIT)

with st.sidebar:
    st.title("Dashboard")
    halaman = st.radio("Pilih Halaman:", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Rekomendasi Harga", "Rekomendasi Device", "Histori"])

if halaman == "Beranda":
    st.title("🎮 Rekomendasi Game untuk Anda")
    st.write("Dapatkan rekomendasi game berdasarkan histori pilihan Anda.")
    st.markdown("---")

    if df.empty or models is None:
        st.error("Gagal memuat data atau model. Aplikasi tidak dapat menampilkan rekomendasi.")
    else:
        is_history_empty = not any(st.session_state.history.values())
        rekomendasi = get_recommendations_based_on_preferences(df) if not is_history_empty else df.sort_values(by='positive reviews', ascending=False).head(DISPLAY_LIMIT)
        if not is_history_empty:
            st.info("Berikut adalah rekomendasi game berdasarkan preferensi Anda:")
        display_recommendations(rekomendasi)

elif halaman == "Penjelasan Metode":
    st.title("📚 Penjelasan Metode")
    st.write("""Aplikasi ini menggunakan metode **Content-Based Filtering** untuk merekomendasikan game. Ini berarti rekomendasi didasarkan pada karakteristik game itu sendiri, seperti deskripsi, genre, tag, dan kategorinya, serta preferensi Anda yang tercatat dari interaksi sebelumnya.
    Bagaimana Cara Kerjanya?
    Model utama yang digunakan adalah **Support Vector Machine (SVM)**. SVM adalah algoritma Machine Learning yang sangat efektif untuk tugas klasifikasi. Dalam konteks ini, SVM dilatih untuk "memahami" hubungan antara teks (seperti deskripsi game) dan atribut-atribut seperti genre, tag, atau kategori.
    Anda mungkin bertanya, "Mengapa hanya SVM, tidak termasuk TF-IDF?"
    **TF-IDF (Term Frequency-Inverse Document Frequency) sebenarnya adalah bagian integral dari proses ini, meskipun tidak secara eksplisit dimuat sebagai model terpisah di sini.**
    """)

    st.write("""
    * **TF-IDF** adalah teknik *ekstraksi fitur* yang digunakan untuk mengubah teks mentah (seperti deskripsi game) menjadi representasi numerik yang dapat dipahami oleh algoritma Machine Learning. Tanpa mengubah teks menjadi angka, model seperti SVM tidak akan bisa memprosesnya.
    * Prosesnya adalah sebagai berikut:
        1.  **Pra-pemrosesan Teks:** Pembersihan duplikat data adalah langkah pra-pemrosesan data yang sangat penting dalam hampir setiap proyek data, termasuk sistem rekomendasi melakukan nya agar tidak ada data duplikat.
        2.  **Vektorisasi dengan TF-IDF:** TF-IDF menghitung seberapa penting sebuah kata dalam sebuah dokumen (deskripsi game) relatif terhadap koleksi semua dokumen (semua deskripsi game). Kata-kata yang unik untuk suatu game akan memiliki skor TF-IDF yang tinggi, sementara kata-kata umum (seperti "dan", "atau") akan memiliki skor rendah. Hasilnya adalah vektor numerik untuk setiap deskripsi game.
        3.  **Pelatihan SVM:** Vektor-vektor numerik ini kemudian digunakan sebagai input untuk melatih model SVM. SVM belajar untuk mengidentifikasi pola dalam vektor-vektor ini yang membedakan satu genre dari genre lainnya, satu tag dari tag lainnya, dan seterusnya.
    * Ketika Anda memilih sebuah genre atau tag, aplikasi ini akan mencari game yang memiliki karakteristik serupa berdasarkan representasi numerik ini yang telah dipelajari oleh model SVM.
    * Dalam implementasi nyata, seringkali TF-IDF Vectorizer dan model SVM disimpan bersama dalam satu objek `pipeline` (misalnya, menggunakan `scikit-learn` Pipeline) dan kemudian di-pickle menjadi satu file (`.pkl`). Jadi, ketika Anda memuat `svm_model.pkl`, Anda sebenarnya memuat seluruh alur kerja yang sudah termasuk TF-IDF Vectorizer di dalamnya. Ini menyederhanakan penyebaran model karena Anda tidak perlu memuat dua objek terpisah.

    Dengan demikian, TF-IDF adalah tahap penting yang memungkinkan SVM bekerja dengan data tekstual. Aplikasi ini menggunakan tiga model SVM terpisah, masing-masing khusus untuk memproses dan merekomendasikan berdasarkan Genre, Tag, dan Kategori, memungkinkan rekomendasi yang lebih spesifik dan akurat.
    """)

elif halaman in ["Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori"]:
    if df.empty or models is None:
        st.error("Gagal memuat data atau model. Tidak dapat menampilkan filter.")
    else:
        page_map = {
            "Rekomendasi Genre": ("genre", "Genre", "genre"),
            "Rekomendasi Tag": ("tags", "Tag", "tag"),
            "Rekomendasi Kategori": ("categories", "Kategori", "category")
        }
        col_name, title_name, key_name = page_map[halaman]
        st.title(f"🎯 Rekomendasi Berdasarkan {title_name}")

        items = sorted(list(set(item.strip() for sublist in df[col_name].dropna() for item in sublist.split(',') if item.strip() and item.strip() != 'N/A')))

        if not items:
            st.warning(f"Tidak ada {title_name.lower()} yang ditemukan.")
        else:
            pilihan = st.selectbox(f"Pilih {title_name.lower()} sebagai filter:", [f"Pilih {title_name}"] + items)
            if pilihan != f"Pilih {title_name}":
                if pilihan not in st.session_state.history[key_name]:
                    st.session_state.history[key_name].append(pilihan)
                hasil = df[df[col_name].str.contains(pilihan, case=False, na=False)].head(DISPLAY_LIMIT)
                st.subheader(f"Rekomendasi Game untuk {title_name}: {pilihan}")
                display_recommendations(hasil)
            else:
                st.info(f"Pilih {title_name.lower()} dari daftar untuk melihat rekomendasi.")

elif halaman == "Rekomendasi Harga":
    st.title("💰 Rekomendasi Berdasarkan Harga")
    if df.empty:
        st.error("Gagal memuat data. Tidak dapat menampilkan filter.")
    else:
        prices = sorted(list(df['price'].unique()))
        if "Gratis" in prices:
            prices.remove("Gratis")
            prices.insert(0, "Gratis")

        pilihan = st.selectbox("Pilih harga sebagai filter:", ["Pilih Harga"] + prices)
        if pilihan != "Pilih Harga":
            hasil = df[df['price'] == pilihan].head(DISPLAY_LIMIT)
            st.subheader(f"Rekomendasi Game dengan Harga: {pilihan}")
            display_recommendations(hasil)
        else:
            st.info("Pilih harga dari daftar untuk melihat rekomendasi.")

elif halaman == "Rekomendasi Device":
    st.title("💻 Rekomendasi Berdasarkan Spesifikasi Device")
    st.write("Filter game berdasarkan Platform, CPU, dan RAM. Opsi filter dibuat dari data yang tersedia.")

    if df.empty:
        st.error("Gagal memuat data. Tidak dapat menampilkan filter.")
    else:
        hasil = df.copy()

        platforms = sorted(list(set(item.strip() for sublist in df['platforms'].dropna() for item in sublist.split(',') if item.strip() and item.strip() != 'N/A')))
        pilihan_platform = st.selectbox("1. Pilih Platform (Wajib)", ["Pilih Platform"] + platforms)

        if pilihan_platform != "Pilih Platform":
            hasil = hasil[hasil['platforms'].str.contains(pilihan_platform, case=False, na=False)]

            cpu_list = hasil['device'].str.extract(r'CPU:\s*([^;]+)')[0].dropna().unique()
            cleaned_cpus = sorted([cpu.strip() for cpu in cpu_list if cpu.strip()])
            pilihan_cpu = st.selectbox("2. Pilih CPU (Opsional)", ["Semua CPU"] + cleaned_cpus)

            ram_list = hasil['device'].str.extract(r'(\d+)\s*GB RAM')[0].dropna()
            if not ram_list.empty:
                unique_rams = sorted(ram_list.astype(int).unique())
                ram_options = [f"{ram} GB" for ram in unique_rams]
                pilihan_ram = st.selectbox("3. Pilih RAM (Opsional)", ["Semua RAM"] + ram_options)
            else:
                pilihan_ram = "Semua RAM"
                st.markdown("<small>_Tidak ada data RAM spesifik untuk pilihan saat ini._</small>", unsafe_allow_html=True)


            filters_applied = [f"Platform: {pilihan_platform}"]

            if pilihan_cpu != "Semua CPU":
                hasil = hasil[hasil['device'].str.contains(re.escape(pilihan_cpu), case=False, na=False)]
                filters_applied.append(f"CPU: {pilihan_cpu}")

            if pilihan_ram != "Semua RAM":
                ram_search_term = pilihan_ram + " RAM"
                hasil = hasil[hasil['device'].str.contains(ram_search_term, case=False, na=False)]
                filters_applied.append(f"RAM: {pilihan_ram}")

            st.markdown("---")
            subheader_title = "Rekomendasi Game untuk " + ", ".join(filters_applied)
            st.subheader(subheader_title)
            display_recommendations(hasil.head(DISPLAY_LIMIT))

        else:
            st.info("Silakan pilih platform terlebih dahulu untuk menampilkan filter dan rekomendasi.")


elif halaman == "Histori":
    st.title("🕒 Histori Game yang Dilihat")
    st.write(f"Berikut adalah daftar hingga {DISPLAY_LIMIT} game yang baru saja Anda lihat.")
    if df.empty:
        st.error("Data game tidak termuat, tidak dapat menampilkan histori.")
    elif st.session_state.viewed_games:
        for game_name in reversed(list(dict.fromkeys(st.session_state.viewed_games))):
            game_details = df[df['name'] == game_name]
            if not game_details.empty:
                display_game_card(game_details.iloc[0])
    else:
        st.info("Anda belum melihat game apa pun.")

    st.markdown("---")
    if st.button("Bersihkan Histori Game yang Dilihat"):
        st.session_state.viewed_games.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("Preferensi Tersimpan")

    col1, col2, col3 = st.columns(3)

    def display_preference_list(column, title):
        with column:
            st.markdown(f"**{title}:**")
            pref_list = st.session_state.history.get(title.lower())
            if pref_list:
                for item in pref_list:
                    st.markdown(f"- {item}")
            else:
                st.markdown("- Tidak ada")

    display_preference_list(col1, "Genre")
    display_preference_list(col2, "Tag")
    display_preference_list(col3, "Kategori")

    if st.button("Bersihkan Preferensi", key="clear_prefs"):
        st.session_state.history = {"genre": [], "tag": [], "category": []}
        st.rerun()
