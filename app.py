import streamlit as st
import pandas as pd
import zipfile
import os
import joblib
from collections import deque
import math
import urllib.parse

# Konfigurasi halaman
st.set_page_config(initial_sidebar_state="expanded")

# --- Konfigurasi ---
DATA_DIR = "data"
ZIP_FILE_NAME = "Dataset.zip"
SVM_MODEL_GENRE = "svm_model.pkl"
SVM_MODEL_TAG = "svm_model_tags.pkl"
SVM_MODEL_CATEGORY = "svm_model_categories.pkl"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/180x100.png?text=No+Image"
DISPLAY_LIMIT = 10
VIEWED_HISTORY_LIMIT = 20

# --- Custom CSS ---
hide_streamlit_style = """
    <style>
    #MainMenu, footer, header { visibility: hidden !important; display: none !important; }
    .stAlert { display: none !important; }
    section[data-testid="stSidebar"] {
        visibility: visible !important; display: block !important; width: 300px !important;
        left: 0px !important; transform: none !important; z-index: 9999 !important;
    }
    .main { padding-left: 300px !important; }
    .stApp > header, .css-1lcbmhc, .css-1d391kg, .css-1f198p6 {
        display: none !important; margin-top: 0px !important; padding-top: 0px !important;
    }
    button[data-testid="stSidebarCollapseButton"] { visibility: hidden !important; display: none !important; }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNGSI-FUNGSI ---

# =========================================================================================
# === PERUBAHAN UTAMA DI SINI: FUNGSI load_data() DIBUAT JAUH LEBIH ANDAL ===
# =========================================================================================
@st.cache_data
def load_data():
    """Memuat dan melakukan pra-pemrosesan dataset dari file ZIP dengan penanganan data yang andal."""
    if not os.path.exists(DATA_DIR):
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        except Exception as e:
            st.error(f"Error saat mengekstrak file: {e}")
            st.stop()

    df = pd.DataFrame()
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".csv") and ("dataset" in file.lower() or "data" in file.lower()):
                try:
                    df = pd.read_csv(os.path.join(root, file))
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

    # Hapus duplikat berdasarkan nama game
    if 'name' in df.columns:
        df.drop_duplicates(subset=['name'], inplace=True, keep='first')

    # --- Pemrosesan Kolom 'device' yang Andal ---
    if 'device' in df.columns:
        # Isi nilai kosong (NaN) dengan string 'N/A', lalu ubah semua jadi string
        df['device'] = df['device'].fillna('N/A').astype(str)
        # Ganti string kosong atau string 'nan' menjadi 'N/A'
        df['device'] = df['device'].apply(lambda x: x.strip() if x.strip() and x.lower() not in ['nan', 'none'] else 'N/A')
    else:
        df['device'] = 'N/A'

    # --- Pemrosesan Kolom 'price' yang Andal ---
    if 'price' in df.columns:
        def format_price_robustly(price_input):
            if pd.isna(price_input):
                return "Gratis"
            price_str = str(price_input).strip().lower()
            if not price_str or price_str in ['0', '0.0', 'free', 'gratis', 'nan', 'none']:
                return "Gratis"
            try:
                price_num = float(price_str)
                return f"Rp{price_num:,.0f}"
            except ValueError:
                return str(price_input).strip() # Kembalikan teks asli jika bukan angka
        df['price'] = df['price'].apply(format_price_robustly)
    else:
        df['price'] = 'N/A'

    # Proses kolom lainnya
    other_cols = {
        'short description': 'Deskripsi tidak tersedia', 'genre': 'N/A',
        'tags': 'N/A', 'categories': 'N/A'
    }
    for col, default in other_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str).apply(lambda x: x.strip() if x.strip() and x.lower() not in ['nan', 'none'] else default)
    
    if 'header image' in df.columns:
        df['header image'] = df['header image'].fillna('').apply(lambda x: x if isinstance(x, str) and x.startswith("http") else "")
    
    if 'positive reviews' in df.columns:
        df['positive reviews'] = pd.to_numeric(df['positive reviews'], errors='coerce').fillna(0)

    return df


@st.cache_resource
def load_svm_models():
    """Memuat model SVM."""
    try:
        models = (joblib.load(SVM_MODEL_GENRE), joblib.load(SVM_MODEL_TAG), joblib.load(SVM_MODEL_CATEGORY))
        return models
    except Exception as e:
        st.error(f"Error saat memuat model SVM: {e}.")
        st.stop()

def get_recommendations_based_on_preferences(data_df):
    """Menghasilkan rekomendasi berdasarkan preferensi."""
    history = st.session_state.history
    if any(history.values()):
        df_temp = data_df.copy()
        df_temp["score"] = 0
        if history["genre"] and 'genre' in df_temp.columns:
            for pref in history["genre"]: df_temp.loc[df_temp["genre"].str.contains(pref, na=False), "score"] += 3
        if history["tag"] and 'tags' in df_temp.columns:
            for pref in history["tag"]: df_temp.loc[df_temp["tags"].str.contains(pref, na=False), "score"] += 2
        if history["category"] and 'categories' in df_temp.columns:
            for pref in history["category"]: df_temp.loc[df_temp["categories"].str.contains(pref, na=False), "score"] += 1
        return df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False).head(DISPLAY_LIMIT)
    return pd.DataFrame()

def display_game_card(game_row):
    """Menampilkan kartu informasi game."""
    nama = game_row.get('name', 'N/A')
    short_description = game_row.get('short description', '')
    price = game_row['price'] if 'price' in game_row and pd.notna(game_row['price']) else 'N/A'
    device = game_row['device'] if 'device' in game_row and pd.notna(game_row['device']) else 'N/A'
    gambar = game_row.get('header image', '') or PLACEHOLDER_IMAGE
    
    genres = ", ".join(g for g in str(game_row.get('genre', '')).split(',') if g.strip()) or 'N/A'
    tags = ", ".join(t for t in str(game_row.get('tags', '')).split(',') if t.strip()) or 'N/A'
    kategoris = ", ".join(k for k in str(game_row.get('categories', '')).split(',') if k.strip()) or 'N/A'
    
    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(f'{nama} gameplay')}"

    st.markdown(f"""
    <div style="display: flex; gap: 20px; padding: 15px; border: 1px solid #444; border-radius: 10px; margin-bottom: 20px; background-color: #222; color: white;">
        <div style="flex-shrink: 0;">
            <img src="{gambar}" style="width: 180px; height: 100%; border-radius: 10px; object-fit: cover;">
        </div>
        <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="margin-top: 0; margin-bottom: 5px;">{nama}</h4>
                <p style="font-size: 14px; margin-bottom: 10px; color: #ccc;">{short_description}</p>
                <p style="font-size: 13px; line-height: 1.6;">
                    <strong>Genre:</strong> {genres} <br>
                    <strong>Tags:</strong> {tags} <br>
                    <strong>Kategori:</strong> {kategoris} <br>
                    <strong>Price:</strong> {price} <br>
                    <strong>Device:</strong> {device}
                </p>
            </div>
            <div style="margin-top: 15px; text-align: right;">
                <a href="{youtube_url}" target="_blank" style="text-decoration: none; color: white; background-color: #c4302b; padding: 8px 16px; border-radius: 5px; font-size: 14px; font-weight: bold; display: inline-block;">
                    🎬 Lihat Gameplay
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if nama != 'N/A' and nama not in st.session_state.viewed_games:
        st.session_state.viewed_games.append(nama)

def display_recommendations(recs_df):
    """Menampilkan daftar rekomendasi game."""
    if recs_df.empty:
        st.info("Tidak ada game yang ditemukan berdasarkan kriteria ini.")
    else:
        for _, row in recs_df.loc[:, ~recs_df.columns.duplicated()].iterrows():
            display_game_card(row)

# --- Logika Utama Aplikasi ---
df = load_data()
if not df.empty:
    model_genre, model_tag, model_category = load_svm_models()

    if "history" not in st.session_state:
        st.session_state.history = {"genre": [], "tag": [], "category": []}
    if "viewed_games" not in st.session_state:
        st.session_state.viewed_games = deque(maxlen=VIEWED_HISTORY_LIMIT)

    with st.sidebar:
        st.title("Dashboard")
        halaman = st.radio("Pilih Halaman:", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"])

    if halaman == "Beranda":
        st.title("🎮 Rekomendasi Game untuk Anda")
        st.write("Dapatkan rekomendasi game berdasarkan histori pilihan Anda.")
        st.markdown("---")
        is_history_empty = not any(st.session_state.history.values())
        rekomendasi = get_recommendations_based_on_preferences(df) if not is_history_empty else df.sort_values(by='positive reviews', ascending=False).head(DISPLAY_LIMIT)
        if not is_history_empty:
            st.info("Berikut adalah rekomendasi game berdasarkan preferensi Anda:")
        display_recommendations(rekomendasi)

    elif halaman == "Penjelasan Metode":
        st.title("📚 Penjelasan Metode")
        st.write("""
        Aplikasi ini menggunakan metode **Content-Based Filtering** untuk merekomendasikan game. Ini berarti rekomendasi didasarkan pada karakteristik game itu sendiri, seperti deskripsi, genre, tag, dan kategorinya, serta preferensi Anda yang tercatat dari interaksi sebelumnya.

        ### Bagaimana Cara Kerjanya?
        Model utama yang digunakan adalah **Support Vector Machine (SVM)**. SVM adalah algoritma Machine Learning yang sangat efektif untuk tugas klasifikasi. Dalam konteks ini, SVM dilatih untuk "memahami" hubungan antara teks (seperti deskripsi game) dan atribut-atribut seperti genre, tag, atau kategori.
        
        Prosesnya melibatkan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengubah teks deskripsi menjadi vektor angka yang dapat diproses oleh SVM. Vektor ini kemudian digunakan untuk melatih model SVM agar dapat mengklasifikasikan game berdasarkan genre, tag, dan kategorinya, yang memungkinkan sistem untuk memberikan rekomendasi yang relevan.
        """)

    elif halaman in ["Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori"]:
        page_map = {
            "Rekomendasi Genre": ("genre", "Genre"),
            "Rekomendasi Tag": ("tags", "Tag"),
            "Rekomendasi Kategori": ("categories", "Kategori")
        }
        col_name, title_name = page_map[halaman]
        
        st.title(f"🎯 Rekomendasi Berdasarkan {title_name}")
        
        items = sorted(list(set(item.strip() for sublist in df[col_name].dropna() for item in sublist.split(',') if item.strip() and item.strip() != 'N/A')))
        
        if not items:
            st.warning(f"Tidak ada {title_name.lower()} yang ditemukan.")
        else:
            pilihan = st.selectbox(f"Pilih {title_name.lower()} sebagai filter:", [f"Pilih {title_name}"] + items)
            if pilihan != f"Pilih {title_name}":
                key_name = 'genre' if col_name == 'genre' else col_name.replace('s', '')
                if pilihan not in st.session_state.history[key_name]:
                    st.session_state.history[key_name].append(pilihan)
                hasil = df[df[col_name].str.contains(pilihan, na=False)]
                st.subheader(f"Rekomendasi Game untuk {title_name}: {pilihan}")
                display_recommendations(hasil)
            else:
                st.info(f"Pilih {title_name.lower()} dari daftar untuk melihat rekomendasi.")
    
    elif halaman == "Histori":
        st.title("🕒 Histori Game yang Dilihat")
        st.write("Berikut adalah daftar game yang baru saja Anda lihat.")
        if st.session_state.viewed_games:
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
        # Tampilan preferensi...
