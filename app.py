import streamlit as st
import pandas as pd
import zipfile
import os
import joblib
from collections import deque
import math
import urllib.parse

# Tambahkan baris ini tepat di sini:
st.set_page_config(initial_sidebar_state="expanded")

# --- Konfigurasi ---
DATA_DIR = "data"
ZIP_FILE_NAME = "Dataset.zip"
SVM_MODEL_GENRE = "svm_model.pkl"
SVM_MODEL_TAG = "svm_model_tags.pkl"
SVM_MODEL_CATEGORY = "svm_model_categories.pkl"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/180x100.png?text=No+Image"
DISPLAY_LIMIT = 10  # Batas untuk game yang ditampilkan di satu halaman
VIEWED_HISTORY_LIMIT = 20  # Batas untuk berapa banyak game unik yang disimpan dalam histori tampilan

# --- Custom CSS: SEMBUNYIKAN KETERANGAN UI ATAS & PESAN INFO/SUCCESS, BIARKAN SIDEBAR UTUH & LEBARKAN, DAN SEMBUNYIKAN TOMBOL LIPAT SIDEBAR ---
hide_streamlit_style = """
    <style>
    /* 1. Menyembunyikan Menu Utama (hamburger), Footer, dan Header Streamlit */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    header { /* Ini adalah header Streamlit default di bagian paling atas */
        visibility: hidden !important;
        display: none !important;
    }

    /* 2. MENJAMIN SEMUA PESAN INFO/SUCCESS/WARNING (ALERTS/BALLOONS) DISEMBUNYIKAN */
    .stAlert {
        display: none !important;
    }

    /* 3. MENJAMIN SIDEBAR TETAP MUNCUL DAN MELEBARKANNYA */
    section[data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
        width: 300px !important;      /* Lebar sidebar (sesuaikan jika perlu) */
        left: 0px !important;
        transform: none !important;
        z-index: 9999 !important;
    }

    /* 4. Pastikan konten utama tidak tumpang tindih dengan sidebar */
    .main {
        padding-left: 300px !important; /* Sesuaikan dengan lebar sidebar */
    }

    /* 5. Mengatasi potensi elemen yang menyebabkan padding/margin tambahan di atas konten utama/sidebar */
    .stApp > header {
        display: none !important;
    }
    .css-1lcbmhc {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    .css-1d391kg {
        padding-top: 0px !important;
    }
    .css-1f198p6 {
        padding-top: 0px !important;
    }

    /* 6. Menyembunyikan tombol lipat sidebar (panah ganda <<) */
    button[data-testid="stSidebarCollapseButton"] {
        visibility: hidden !important;
        display: none !important;
    }

    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNGSI-FUNGSI ---

# =========================================================================================
# === PERUBAHAN DI SINI: FUNGSI load_data DIMODIFIKASI UNTUK MENANGANI DATA KOSONG ===
# =========================================================================================
@st.cache_data
def load_data():
    """Memuat dan melakukan pra-pemrosesan dataset dari file ZIP."""
    if not os.path.exists(DATA_DIR):
        st.info(f"Mengekstrak {ZIP_FILE_NAME}...")
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            st.success("Dataset berhasil diekstrak!")
        except FileNotFoundError:
            st.error(f"Error: {ZIP_FILE_NAME} tidak ditemukan.")
            st.stop()
        except Exception as e:
            st.error(f"Error saat mengekstrak file zip: {e}")
            st.stop()

    df = pd.DataFrame()
    csv_found = False
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".csv") and ("dataset" in file.lower() or "data" in file.lower()):
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    df.columns = [col.strip().lower() for col in df.columns]
                    csv_found = True
                    break
                except Exception as e:
                    st.sidebar.error(f"Error saat memuat CSV '{file}': {e}")
        if csv_found:
            break

    if not csv_found:
        st.error(f"Tidak ada file CSV dataset yang cocok ditemukan di direktori '{DATA_DIR}'.")
        return pd.DataFrame()

    if not df.empty:
        if 'name' in df.columns:
            df.drop_duplicates(subset=['name'], inplace=True, keep='first')
        
        # Kolom yang akan diproses
        cols_to_process = {
            'short description': 'Deskripsi tidak tersedia.',
            'genre': 'N/A',
            'tags': 'N/A',
            'categories': 'N/A',
            'header image': '',
            'device': 'N/A',
            'platforms': 'N/A'
        }

        for col, default_value in cols_to_process.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_value)
                # Pastikan string kosong juga diubah menjadi nilai default
                df[col] = df[col].apply(lambda x: default_value if isinstance(x, str) and not x.strip() else x)
            else:
                st.sidebar.warning(f"Kolom '{col}' tidak ditemukan.")

        # Pemrosesan khusus untuk 'price'
        if 'price' in df.columns:
            # Ubah ke numerik, yang error jadi NaN, lalu isi NaN dengan 0
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            # Format harga: 0 jadi 'Gratis', selain itu format ke Rupiah
            df['price'] = df['price'].apply(lambda x: "Gratis" if x == 0 else f"Rp{float(x):,.0f}")
        else:
            st.sidebar.warning("Kolom 'price' tidak ditemukan.")

        # Pemrosesan khusus untuk 'positive reviews'
        if 'positive reviews' in df.columns:
            df['positive reviews'] = pd.to_numeric(df['positive reviews'], errors='coerce').fillna(0)

        # Validasi URL gambar
        if 'header image' in df.columns:
            df['header image'] = df['header image'].apply(lambda x: x if isinstance(x, str) and x.startswith("http") else "")

    return df

@st.cache_resource
def load_svm_models():
    """Memuat model SVM yang sudah dilatih."""
    try:
        model_genre = joblib.load(SVM_MODEL_GENRE)
        model_tag = joblib.load(SVM_MODEL_TAG)
        model_category = joblib.load(SVM_MODEL_CATEGORY)
        st.sidebar.success("Model SVM berhasil dimuat.")
        return model_genre, model_tag, model_category
    except FileNotFoundError:
        st.error(f"Satu atau lebih file model ({SVM_MODEL_GENRE}, {SVM_MODEL_TAG}, {SVM_MODEL_CATEGORY}) tidak ditemukan.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model SVM: {e}. Periksa file model Anda.")
        st.stop()

def get_recommendations_based_on_preferences(data_df):
    """Menghasilkan rekomendasi game berdasarkan histori preferensi genre/tag/kategori pengguna."""
    preferensi_genre = st.session_state.history["genre"]
    preferensi_tag = st.session_state.history["tag"]
    preferensi_kat = st.session_state.history["category"]

    if preferensi_genre or preferensi_tag or preferensi_kat:
        df_temp = data_df.copy()
        df_temp["score"] = 0

        if preferensi_genre and 'genre' in df_temp.columns:
            for pref_g in preferensi_genre:
                df_temp.loc[df_temp["genre"].str.contains(pref_g, case=False, na=False), "score"] += 3
        if preferensi_tag and 'tags' in df_temp.columns:
            for pref_t in preferensi_tag:
                df_temp.loc[df_temp["tags"].str.contains(pref_t, case=False, na=False), "score"] += 2
        if preferensi_kat and 'categories' in df_temp.columns:
            for pref_k in preferensi_kat:
                df_temp.loc[df_temp["categories"].str.contains(pref_k, case=False, na=False), "score"] += 1

        hasil = df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False)
        return hasil.head(DISPLAY_LIMIT)
    else:
        return pd.DataFrame()

def display_game_card(game_row):
    """Menampilkan informasi game tunggal dalam format kartu terstruktur dengan tombol YouTube."""
    nama = game_row.get('name', 'Tidak ada nama').strip()
    short_description = game_row.get('short description', 'Deskripsi tidak tersedia.').strip()
    
    genre_str = str(game_row.get('genre', '')).strip()
    tag_str = str(game_row.get('tags', '')).strip()
    kategori_str = str(game_row.get('categories', '')).strip()

    price = game_row.get('price', 'N/A')
    device = game_row.get('device', 'N/A')
    
    genres_formatted = ", ".join([g.strip() for g in genre_str.split(',') if g.strip()]) if genre_str != 'N/A' else 'N/A'
    tags_formatted = ", ".join([t.strip() for t in tag_str.split(',') if t.strip()]) if tag_str != 'N/A' else 'N/A'
    kategoris_formatted = ", ".join([k.strip() for k in kategori_str.split(',') if k.strip()]) if kategori_str != 'N/A' else 'N/A'

    gambar = game_row.get('header image', '')
    if not isinstance(gambar, str) or not gambar.startswith("http") or not gambar.strip():
        gambar = PLACEHOLDER_IMAGE

    Youtube_query = f"{nama} gameplay"
    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(Youtube_query)}"

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
                    <strong>Genre:</strong> {genres_formatted} <br>
                    <strong>Tags:</strong> {tags_formatted} <br>
                    <strong>Kategori:</strong> {kategoris_formatted} <br>
                    <strong>Harga:</strong> {price} <br>
                    <strong>Perangkat:</strong> {device}
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

    if nama != 'Tidak ada nama' and nama not in st.session_state.viewed_games:
        st.session_state.viewed_games.append(nama)


def display_recommendations(data_df, title, recommendations_df):
    """Menampilkan judul dan daftar rekomendasi game, menerapkan batas tampilan."""
    if title:
        st.subheader(title)

    if recommendations_df.empty:
        st.info("Tidak ada game yang ditemukan berdasarkan kriteria ini.")
    else:
        recommendations_df = recommendations_df.loc[:,~recommendations_df.columns.duplicated()]
        for _, row in recommendations_df.iterrows():
            display_game_card(row)

# --- Logika Utama Aplikasi ---

df = load_data()
if not df.empty:
    df = df.loc[:,~df.columns.duplicated()]

    model_genre, model_tag, model_category = load_svm_models()

    if "history" not in st.session_state:
        st.session_state.history = {"genre": [], "tag": [], "category": []}
    if "viewed_games" not in st.session_state:
        st.session_state.viewed_games = deque(maxlen=VIEWED_HISTORY_LIMIT)

    with st.sidebar:
        st.title("Dashboard")
        halaman = st.radio("Pilih Halaman:", [
            "Beranda", "Penjelasan Metode", "Rekomendasi Genre",
            "Rekomendasi Tag", "Rekomendasi Kategori", "Histori"
        ])
    
    if halaman == "Beranda":
        st.title("🎮 Rekomendasi Game untuk Anda")
        st.write("Selamat datang! Dapatkan rekomendasi game berdasarkan histori pilihan Anda.")
        st.markdown("---")
        st.header("Rekomendasi Game")

        is_preference_history_empty = not (st.session_state.history["genre"] or
                                           st.session_state.history["tag"] or
                                           st.session_state.history["category"])

        if is_preference_history_empty:
            if 'positive reviews' in df.columns:
                rekomendasi = df.sort_values(by='positive reviews', ascending=False).head(DISPLAY_LIMIT)
            else:
                rekomendasi = pd.DataFrame()
        else:
            st.info("Berikut adalah rekomendasi game berdasarkan preferensi Anda sebelumnya:")
            rekomendasi = get_recommendations_based_on_preferences(df)

        display_recommendations(df, "", rekomendasi)

    elif halaman == "Penjelasan Metode":
        st.title("📚 Penjelasan Metode")
        st.write("""
        Aplikasi ini menggunakan metode **Content-Based Filtering** untuk merekomendasikan game...
        """) # Penjelasan metode tidak diubah

    elif halaman == "Rekomendasi Genre":
        st.title("🎯 Rekomendasi Berdasarkan Genre")
        all_genres = set()
        if 'genre' in df.columns:
            for genres_str in df['genre'].dropna().unique():
                if genres_str != 'N/A':
                    for g in str(genres_str).split(','):
                        all_genres.add(g.strip())
        daftar_genre = sorted(list(all_genres))
        
        if not daftar_genre:
            st.warning("Tidak ada genre yang ditemukan di dataset.")
        else:
            genre_pilihan = st.selectbox("Pilih genre sebagai filter awal:", ["Pilih Genre"] + daftar_genre)
            if genre_pilihan != "Pilih Genre":
                if genre_pilihan not in st.session_state.history['genre']:
                    st.session_state.history['genre'].append(genre_pilihan)
                hasil = df[df['genre'].str.contains(genre_pilihan, case=False, na=False)]
                display_recommendations(df, f"Rekomendasi Game untuk Genre: {genre_pilihan}", hasil)
            else:
                st.info("Pilih genre dari daftar di atas untuk melihat rekomendasi.")

    elif halaman == "Rekomendasi Tag":
        st.title("🏷️ Rekomendasi Berdasarkan Tag")
        all_tags = set()
        if 'tags' in df.columns:
            for tags_str in df['tags'].dropna().unique():
                if tags_str != 'N/A':
                    for t in str(tags_str).split(','):
                        all_tags.add(t.strip())
        daftar_tag = sorted(list(all_tags))
        
        if not daftar_tag:
            st.warning("Tidak ada tag yang ditemukan di dataset.")
        else:
            tag_pilihan = st.selectbox("Pilih tag sebagai filter awal:", ["Pilih Tag"] + daftar_tag)
            if tag_pilihan != "Pilih Tag":
                if tag_pilihan not in st.session_state.history['tag']:
                    st.session_state.history['tag'].append(tag_pilihan)
                hasil = df[df['tags'].str.contains(tag_pilihan, case=False, na=False)]
                display_recommendations(df, f"Rekomendasi Game untuk Tag: {tag_pilihan}", hasil)
            else:
                st.info("Pilih tag dari daftar di atas untuk melihat rekomendasi.")

    elif halaman == "Rekomendasi Kategori":
        st.title("📂 Rekomendasi Berdasarkan Kategori")
        all_categories = set()
        if 'categories' in df.columns:
            for categories_str in df['categories'].dropna().unique():
                 if categories_str != 'N/A':
                    for c in str(categories_str).split(','):
                        all_categories.add(c.strip())
        daftar_kategori = sorted(list(all_categories))

        if not daftar_kategori:
            st.warning("Tidak ada kategori yang ditemukan di dataset.")
        else:
            kategori_pilihan = st.selectbox("Pilih kategori sebagai filter awal:", ["Pilih Kategori"] + daftar_kategori)
            if kategori_pilihan != "Pilih Kategori":
                if kategori_pilihan not in st.session_state.history['category']:
                    st.session_state.history['category'].append(kategori_pilihan)
                hasil = df[df['categories'].str.contains(kategori_pilihan, case=False, na=False)]
                display_recommendations(df, f"Rekomendasi Game untuk Kategori: {kategori_pilihan}", hasil)
            else:
                st.info("Pilih kategori dari daftar di atas untuk melihat rekomendasi.")

    elif halaman == "Histori":
        st.title("🕒 Histori Game yang Dilihat")
        st.write("Berikut adalah daftar game yang baru saja Anda lihat dari berbagai halaman rekomendasi.")
        if st.session_state.viewed_games:
            unique_viewed_games = list(dict.fromkeys(reversed(list(st.session_state.viewed_games))))
            for game_name in unique_viewed_games:
                game_details = df[df['name'] == game_name]
                if not game_details.empty:
                    display_game_card(game_details.iloc[0])
                else:
                    st.markdown(f"- **{game_name}** (Detail tidak ditemukan)")
        else:
            st.info("Anda belum melihat game apa pun. Jelajahi rekomendasi untuk memulai!")
        
        # Opsi untuk membersihkan histori tidak diubah
        st.markdown("---")
        st.subheader("Bersihkan Histori Game")
        if st.button("Bersihkan Semua Histori Game yang Dilihat"):
            st.session_state.viewed_games.clear()
            st.success("Histori game yang dilihat telah dibersihkan!")
            st.rerun()

        st.markdown("---")
        st.subheader("Preferensi Rekomendasi Tersimpan (untuk algoritma)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Genre:**")
            st.write(st.session_state.history.get("genre", ["- Tidak ada"]))
        with col2:
            st.markdown("**Tag:**")
            st.write(st.session_state.history.get("tag", ["- Tidak ada"]))
        with col3:
            st.markdown("**Kategori:**")
            st.write(st.session_state.history.get("category", ["- Tidak ada"]))
        if st.button("Bersihkan Preferensi Rekomendasi", key="clear_pref_history"):
            st.session_state.history = {"genre": [], "tag": [], "category": []}
            st.success("Preferensi rekomendasi telah dibersihkan!")
            st.rerun()
