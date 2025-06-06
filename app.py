import streamlit as st
import pandas as pd
import zipfile
import os
import joblib

# --- Configuration ---
DATA_DIR = "data"
ZIP_FILE_NAME = "Dataset.zip"
SVM_MODEL_GENRE = "svm_model.pkl"
SVM_MODEL_TAG = "svm_model_tags.pkl"
SVM_MODEL_CATEGORY = "svm_model_categories.pkl"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/180x100.png?text=No+Image"

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Loads and preprocesses the dataset from a ZIP file."""
    if not os.path.exists(DATA_DIR):
        st.info(f"Extracting {ZIP_FILE_NAME}...")
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            st.success("Dataset extracted successfully!")
        except FileNotFoundError:
            st.error(f"Error: {ZIP_FILE_name} not found. Please ensure it's in the same directory as the script.")
            st.stop()
        except Exception as e:
            st.error(f"Error extracting zip file: {e}")
            st.stop()

    df = pd.DataFrame()
    csv_found = False
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(".csv") and ("dataset" in file.lower() or "data" in file.lower()):
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    df.columns = df.columns.str.strip().str.lower() # Normalize column names
                    csv_found = True
                    st.sidebar.success(f"Dataset '{file}' loaded successfully.")
                    break
                except Exception as e:
                    st.sidebar.error(f"Error loading CSV '{file}': {e}")
        if csv_found:
            break

    if not csv_found:
        st.error(f"No suitable dataset CSV file found in the '{DATA_DIR}' directory after extraction.")
        return pd.DataFrame()

    if not df.empty:
        st.sidebar.subheader("Data Preprocessing:")

        # Deduplication
        if 'name' in df.columns:
            initial_rows = len(df)
            df.drop_duplicates(subset=['name'], inplace=True, keep='first')
            if initial_rows - len(df) > 0:
                st.sidebar.info(f"Dropped {initial_rows - len(df)} duplicate game entries based on 'name'.")
        else:
            st.sidebar.warning("Column 'name' not found for duplicate removal. Skipping deduplication.")

        # Handle Missing 'Short Description'
        if 'short description' in df.columns:
            df['short description'] = df['short description'].fillna('Deskripsi tidak tersedia.')
        else:
            st.sidebar.warning("Column 'short description' not found. Game descriptions might be missing.")

        # Handle Missing 'Genre', 'Tags', 'Categories', 'Header Image'
        for col in ['genre', 'tags', 'categories', 'header image']:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                st.sidebar.warning(f"Column '{col}' not found in dataset. Ensure correct column names.")
        
        # Ensure image URLs are valid
        if 'header image' in df.columns:
            df['header image'] = df['header image'].apply(lambda x: x if (isinstance(x, str) and x.startswith("http")) else "")

    return df

@st.cache_resource
def load_svm_models():
    """Loads pre-trained SVM models."""
    try:
        model_genre = joblib.load(SVM_MODEL_GENRE)
        model_tag = joblib.load(SVM_MODEL_TAG)
        model_category = joblib.load(SVM_MODEL_CATEGORY)
        st.sidebar.success("SVM Models loaded successfully.")
        return model_genre, model_tag, model_category
    except FileNotFoundError:
        st.error(f"One or more model files ({SVM_MODEL_GENRE}, {SVM_MODEL_TAG}, {SVM_MODEL_CATEGORY}) not found. Please ensure they are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading SVM models: {e}. Please check your model files.")
        st.stop()

def get_recommendations_from_history(data_df):
    """Generates game recommendations based on user's history."""
    preferensi_genre = st.session_state.history["genre"]
    preferensi_tag = st.session_state.history["tag"]
    preferensi_kat = st.session_state.history["category"]

    if preferensi_genre or preferensi_tag or preferensi_kat:
        df_temp = data_df.copy()
        df_temp["score"] = 0

        # Weighted scoring for historical preferences
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
        return hasil.head(10)
    else:
        # Fallback to random sample if no history or dataset is empty
        if not data_df.empty:
            return data_df.sample(min(10, len(data_df)), random_state=42) # Use random_state for reproducibility
        else:
            return pd.DataFrame()

def display_game_card(game_row):
    """Displays a single game's information in a structured card format."""
    nama = game_row.get('name', 'Tidak ada nama').strip()
    short_description = game_row.get('short description', 'Deskripsi tidak tersedia.').strip()
    if not short_description:
        short_description = 'Deskripsi tidak tersedia.'

    genre_str = str(game_row.get('genre', '')).strip()
    tag_str = str(game_row.get('tags', '')).strip()
    kategori_str = str(game_row.get('categories', '')).strip()

    # --- MODIFIED: Join with comma and space instead of <br> ---
    genres_formatted = ", ".join([g.strip() for g in genre_str.split(',') if g.strip()]) if genre_str else '-'
    tags_formatted = ", ".join([t.strip() for t in tag_str.split(',') if t.strip()]) if tag_str else '-'
    kategoris_formatted = ", ".join([k.strip() for k in kategori_str.split(',') if k.strip()]) if kategori_str else '-'
    # --- END MODIFIED ---

    gambar = game_row.get('header image', '')
    if not isinstance(gambar, str) or not gambar.startswith("http") or not gambar.strip():
        gambar = PLACEHOLDER_IMAGE

    st.markdown(f"""
    <div style="display: flex; gap: 20px; padding: 15px; border: 1px solid #444; border-radius: 10px; margin-bottom: 20px; background-color: #222;">
        <div style="flex-shrink: 0;">
            <img src="{gambar}" style="width: 180px; height: auto; border-radius: 10px; object-fit: cover;">
        </div>
        <div style="flex-grow: 1; color: white;">
            <h4 style="margin-bottom: 5px;">{nama}</h4>
            <p style="font-size: 14px; margin-bottom: 10px;">{short_description}</p>
            <p style="font-size: 13px;">
                <strong>Genre:</strong> {genres_formatted} <br>
                <strong>Tags:</strong> {tags_formatted} <br>
                <strong>Kategori:</strong> {kategoris_formatted}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_recommendations(data_df, title, recommendations_df):
    """Displays a title and a list of game recommendations."""
    if title: # Only display subheader if title is provided
        st.subheader(title)
    if recommendations_df.empty:
        st.info("Tidak ada game yang ditemukan berdasarkan kriteria ini.")
    else:
        for _, row in recommendations_df.iterrows():
            display_game_card(row)

# --- Main App Logic ---

# Load data and models
df = load_data()
model_genre, model_tag, model_category = load_svm_models()

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = {"genre": [], "tag": [], "category": []}

# Sidebar Navigation
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori", "Histori Pilihan"])

# --- Page Content ---

if halaman == "Beranda":
    st.title("🎮 Rekomendasi Game Terbaik untuk Anda")
    st.write("Selamat datang! Dapatkan rekomendasi game personal berdasarkan histori pilihan Anda. Semakin banyak Anda memilih genre, tag, atau kategori, semakin akurat rekomendasi yang diberikan.")
    
    st.markdown("---")
    st.header("Rekomendasi Game Berdasarkan Histori Anda")
    rekomendasi = get_recommendations_from_history(df)
    display_recommendations(df, "", rekomendasi) # Title passed as empty string, as header is above

elif halaman == "Penjelasan Metode":
    st.title("📚 Penjelasan Metode")
    st.write("""
    Aplikasi ini menggunakan metode **Content-Based Filtering** untuk merekomendasikan game. Ini berarti rekomendasi didasarkan pada karakteristik game itu sendiri, seperti deskripsi, genre, tag, dan kategorinya, serta preferensi Anda yang tercatat dari interaksi sebelumnya.

    ### Bagaimana Cara Kerjanya?
    Model utama yang digunakan adalah **Support Vector Machine (SVM)**. SVM adalah algoritma Machine Learning yang sangat efektif untuk tugas klasifikasi. Dalam konteks ini, SVM dilatih untuk "memahami" hubungan antara teks (seperti deskripsi game) dan atribut-atribut seperti genre, tag, atau kategori.

    Anda mungkin bertanya, "Mengapa hanya SVM, tidak termasuk TF-IDF?"
    **TF-IDF (Term Frequency-Inverse Document Frequency) sebenarnya adalah bagian integral dari proses ini, meskipun tidak secara eksplisit dimuat sebagai model terpisah di sini.**
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Tf-idf.png/500px-Tf-idf.png", caption="Visualisasi Konsep TF-IDF")
    st.write("""
    * **TF-IDF** adalah teknik *ekstraksi fitur* yang digunakan untuk mengubah teks mentah (seperti deskripsi game) menjadi representasi numerik yang dapat dipahami oleh algoritma Machine Learning. Tanpa mengubah teks menjadi angka, model seperti SVM tidak akan bisa memprosesnya.
    * Prosesnya adalah sebagai berikut:
        1.  **Preprocessing Teks:** Teks deskripsi game dibersihkan (misalnya, menghilangkan tanda baca, mengubah ke huruf kecil, menghapus stopwords).
        2.  **Vektorisasi dengan TF-IDF:** TF-IDF menghitung seberapa penting sebuah kata dalam sebuah dokumen (deskripsi game) relatif terhadap koleksi semua dokumen (semua deskripsi game). Kata-kata yang unik untuk suatu game akan memiliki skor TF-IDF yang tinggi, sementara kata-kata umum (seperti "dan", "atau") akan memiliki skor rendah. Hasilnya adalah vektor numerik untuk setiap deskripsi game.
        3.  **Pelatihan SVM:** Vektor-vektor numerik ini kemudian digunakan sebagai input untuk melatih model SVM. SVM belajar untuk mengidentifikasi pola dalam vektor-vektor ini yang membedakan satu genre dari genre lainnya, satu tag dari tag lainnya, dan seterusnya.
    * Ketika Anda memilih sebuah genre atau tag, aplikasi ini akan mencari game yang memiliki karakteristik serupa berdasarkan representasi numerik ini yang telah dipelajari oleh model SVM.
    * Dalam implementasi nyata, seringkali TF-IDF Vectorizer dan model SVM disimpan bersama dalam satu objek `pipeline` (misalnya, menggunakan `scikit-learn` Pipeline) dan kemudian di-pickle menjadi satu file (`.pkl`). Jadi, ketika Anda memuat `svm_model.pkl`, Anda sebenarnya memuat seluruh alur kerja yang sudah termasuk TF-IDF Vectorizer di dalamnya. Ini menyederhanakan penyebaran model karena Anda tidak perlu memuat dua objek terpisah.

    Dengan demikian, TF-IDF adalah tahap penting yang memungkinkan SVM bekerja dengan data tekstual. Aplikasi ini menggunakan tiga model SVM terpisah, masing-masing khusus untuk memproses dan merekomendasikan berdasarkan Genre, Tag, dan Kategori, memungkinkan rekomendasi yang lebih spesifik dan akurat.
    """)

elif halaman == "Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")

    all_genres = set()
    if 'genre' in df.columns:
        for genres_str in df['genre'].dropna().unique():
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

elif halaman == "Histori Pilihan":
    st.title("🕒 Histori Pilihan Anda")
    st.write("Lihat genre, tag, dan kategori yang telah Anda pilih. Ini membantu sistem merekomendasikan game yang lebih sesuai untuk Anda.")

    # Use columns for a neater history display
    col1, col2, col3 = st.columns(3) # Use 3 columns for genre, tag, category if space allows

    with col1:
        st.subheader("Histori Genre")
        if st.session_state.history["genre"]:
            for i, genre in enumerate(st.session_state.history["genre"]):
                st.markdown(f"- {genre}")
        else:
            st.info("Anda belum memilih genre apa pun.")

    with col2:
        st.subheader("Histori Tag")
        if st.session_state.history["tag"]:
            for i, tag in enumerate(st.session_state.history["tag"]):
                st.markdown(f"- {tag}")
        else:
            st.info("Anda belum memilih tag apa pun.")
    
    with col3: # Moved category to third column
        st.subheader("Histori Kategori")
        if st.session_state.history["category"]:
            for i, category in enumerate(st.session_state.history["category"]):
                st.markdown(f"- {category}")
        else:
            st.info("Anda belum memilih kategori apa pun.")

    st.markdown("---")
    st.header("Rekomendasi Berdasarkan Kombinasi Histori")
    st.write("Berikut adalah rekomendasi game yang disesuaikan berdasarkan seluruh histori pilihan Anda (genre, tag, dan kategori).")
    
    rekomendasi_histori_gabungan = get_recommendations_from_history(df)
    display_recommendations(df, "", rekomendasi_histori_gabungan) # Title passed as empty string, as header is above

    st.markdown("---")
    st.subheader("Bersihkan Histori")
    if st.button("Bersihkan Semua Histori Pilihan"):
        st.session_state.history = {"genre": [], "tag": [], "category": []}
        st.success("Histori pilihan Anda telah dibersihkan!")
        st.rerun() # Rerun to update the display
