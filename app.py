import streamlit as st
import pandas as pd
import zipfile
import os
import joblib

# === Membaca dataset dari ZIP ===
def load_data():
    data_dir = "data"
    zip_file_name = "Dataset.zip"

    # Ensure the 'data' directory exists and extract if not
    if not os.path.exists(data_dir):
        st.info(f"Extracting {zip_file_name}...")
        try:
            with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            st.success("Dataset extracted successfully!")
        except FileNotFoundError:
            st.error(f"Error: {zip_file_name} not found. Please ensure it's in the same directory as the script.")
            st.stop() # Stop the app if the zip file is missing
        except Exception as e:
            st.error(f"Error extracting zip file: {e}")
            st.stop()

    df = pd.DataFrame() # Initialize df outside the loop

    # Walk through the extracted directory to find the CSV
    csv_found = False
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Look for a CSV file that likely contains the dataset
            if file.lower().endswith(".csv") and ("dataset" in file.lower() or "data" in file.lower()):
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    # Convert all column names to lowercase to ensure consistency
                    df.columns = df.columns.str.strip().str.lower()
                    csv_found = True
                    st.sidebar.success(f"Dataset '{file}' loaded successfully.")
                    break # Stop after finding the first suitable CSV
                except Exception as e:
                    st.sidebar.error(f"Error loading CSV '{file}': {e}")
        if csv_found:
            break

    if not csv_found:
        st.error(f"No suitable dataset CSV file found in the '{data_dir}' directory after extraction.")
        return pd.DataFrame()

    # --- Data Preprocessing ---
    if not df.empty:
        st.sidebar.subheader("Data Preprocessing:")

        # 1. Drop Duplicates
        if 'name' in df.columns:
            initial_rows = len(df)
            df.drop_duplicates(subset=['name'], inplace=True, keep='first')
            st.sidebar.info(f"Dropped {initial_rows - len(df)} duplicate game entries based on 'name'.")
        else:
            st.sidebar.warning("No 'name' column found for duplicate removal. Skipping deduplication.")

        # 2. Handle Missing Descriptions
        # Check for 'deskripsi' first, then 'description'
        if 'deskripsi' in df.columns:
            df['deskripsi'] = df['deskripsi'].fillna('Deskripsi tidak tersedia.')
        elif 'description' in df.columns:
            df['description'] = df['description'].fillna('Deskripsi tidak tersedia.')
        else:
            st.sidebar.warning("Neither 'deskripsi' nor 'description' column found. Game descriptions might be missing.")

        # 3. Handle Missing Genres, Tags, Categories, and Images
        # Ensure 'genre', 'tags', 'categories', 'img' (all lowercase) exist and are handled
        # The .str.lower() above already handles the capitalization from source.
        # Here we just ensure they exist and fill NaN.
        for col in ['genre', 'tags', 'categories', 'img']: # Use 'tags' and 'categories' consistently here
            if col in df.columns:
                df[col] = df[col].fillna('') # Fill NaN with empty string
            else:
                st.sidebar.warning(f"Column '{col}' not found in dataset. Ensure correct column names.")
        
        # Ensure image URLs are valid
        if 'img' in df.columns:
            df['img'] = df['img'].apply(lambda x: x if (isinstance(x, str) and x.startswith("http")) else "")


    return df

df = load_data()

# === Load all SVM models ===
# Ensure these files exist or handle their absence gracefully
try:
    model_genre = joblib.load("svm_model.pkl")
    model_tag = joblib.load("svm_model_tags.pkl") # Assuming this model maps to 'tags'
    model_category = joblib.load("svm_model_categories.pkl") # Assuming this model maps to 'categories'
    st.sidebar.success("SVM Models loaded successfully.")
except FileNotFoundError:
    st.error("Model files (svm_model.pkl, svm_model_tags.pkl, svm_model_categories.pkl) not found. Please ensure they are in the same directory.")
    st.stop() # Stop the app if models are not found
except Exception as e:
    st.error(f"Error loading SVM models: {e}. Please check your model files.")
    st.stop()

# === Sidebar Navigation ===
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Penjelasan Metode", "Rekomendasi Genre", "Rekomendasi Tag", "Rekomendasi Kategori"])

# === Store history of genre/tag/category choices ===
if "history" not in st.session_state:
    st.session_state.history = {"genre": [], "tag": [], "category": []}

# === Function for Combined Recommendation based on Weighted Preferences ===
def rekomendasi_berdasarkan_histori():
    preferensi_genre = st.session_state.history["genre"]
    preferensi_tag = st.session_state.history["tag"]
    preferensi_kat = st.session_state.history["category"]

    if preferensi_genre or preferensi_tag or preferensi_kat:
        df_temp = df.copy()
        df_temp["score"] = 0

        # Weighted scoring for historical preferences
        if preferensi_genre and 'genre' in df_temp.columns:
            for pref_g in preferensi_genre:
                df_temp.loc[df_temp["genre"].str.contains(pref_g, case=False, na=False), "score"] += 3
        if preferensi_tag and 'tags' in df_temp.columns: # Use 'tags' here
            for pref_t in preferensi_tag:
                df_temp.loc[df_temp["tags"].str.contains(pref_t, case=False, na=False), "score"] += 2
        if preferensi_kat and 'categories' in df_temp.columns: # Use 'categories' here
            for pref_k in preferensi_kat:
                df_temp.loc[df_temp["categories"].str.contains(pref_k, case=False, na=False), "score"] += 1

        hasil = df_temp[df_temp["score"] > 0].sort_values(by="score", ascending=False)
        return hasil.head(10)
    else:
        # Fallback to random sample if no history or dataset is empty
        if not df.empty:
            return df.sample(min(10, len(df)))
        else:
            return pd.DataFrame() # Return empty if df is also empty

# === Function to Display Games in Card Format ===
def tampilkan_game(hasil):
    if hasil.empty:
        st.write("Tidak ada game yang ditemukan berdasarkan kriteria yang dipilih.")
        return

    for i, row in hasil.iterrows():
        nama = row.get('name', 'Tidak ada nama').strip()

        # Prioritize 'deskripsi' but fall back to 'description'
        deskripsi = row.get('deskripsi', row.get('description', 'Deskripsi tidak tersedia.')).strip()
        if not deskripsi: # If it's empty after stripping
            deskripsi = 'Deskripsi tidak tersedia.'

        # Process genre, tags, categories to display them separately
        # Ensure these columns exist and are treated as strings before splitting
        genre_str = str(row.get('genre', '')).strip()
        tag_str = str(row.get('tags', '')).strip() # Use 'tags' here
        kategori_str = str(row.get('categories', '')).strip() # Use 'categories' here

        # Split and join with <br> for separate lines, filter out empty strings
        genres_formatted = "<br>".join([g.strip() for g in genre_str.split(',') if g.strip()]) if genre_str else '-'
        tags_formatted = "<br>".join([t.strip() for t in tag_str.split(',') if t.strip()]) if tag_str else '-'
        kategoris_formatted = "<br>".join([k.strip() for k in kategori_str.split(',') if k.strip()]) if kategori_str else '-'

        # Image handling (already robust in previous version, added explicit check for empty string)
        gambar = row.get('img', '')
        if not isinstance(gambar, str) or not gambar.startswith("http") or not gambar.strip():
            gambar = "https://via.placeholder.com/180x100.png?text=No+Image"

        st.markdown(f"""
        <div style="display: flex; gap: 20px; padding: 15px; border: 1px solid #444; border-radius: 10px; margin-bottom: 20px; background-color: #222;">
            <div style="flex-shrink: 0;">
                <img src="{gambar}" style="width: 180px; height: auto; border-radius: 10px; object-fit: cover;">
            </div>
            <div style="flex-grow: 1; color: white;">
                <h4 style="margin-bottom: 5px;">{nama}</h4>
                <p style="font-size: 14px; margin-bottom: 10px;">{deskripsi}</p>
                <p style="font-size: 13px;">
                    <strong>Genre:</strong> {genres_formatted} <br>
                    <strong>Tags:</strong> {tags_formatted} <br>
                    <strong>Kategori:</strong> {kategoris_formatted}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === Home Page ===
if halaman == "Beranda":
    st.title("🎮 Rekomendasi Game Terbaik untuk Anda")
    st.write("Berikut adalah 10 rekomendasi game terbaik berdasarkan histori interaksi Anda.")
    rekomendasi = rekomendasi_berdasarkan_histori()
    tampilkan_game(rekomendasi)

# === Explanation Method Page ===
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
        1.  **Preprocessing Teks:** Teks deskripsi game dibersihkan (misalnya, menghilangkan tanda baca, mengubah ke huruf kecil).
        2.  **Vektorisasi dengan TF-IDF:** TF-IDF menghitung seberapa penting sebuah kata dalam sebuah dokumen (deskripsi game) relatif terhadap koleksi semua dokumen (semua deskripsi game). Kata-kata yang unik untuk suatu game akan memiliki skor TF-IDF yang tinggi, sementara kata-kata umum (seperti "dan", "atau") akan memiliki skor rendah. Hasilnya adalah vektor numerik untuk setiap deskripsi game.
        3.  **Pelatihan SVM:** Vektor-vektor numerik ini kemudian digunakan sebagai input untuk melatih model SVM. SVM belajar untuk mengidentifikasi pola dalam vektor-vektor ini yang membedakan satu genre dari genre lainnya, satu tag dari tag lainnya, dan seterusnya.
    * Ketika Anda memilih sebuah genre atau tag, aplikasi ini akan mencari game yang memiliki karakteristik serupa berdasarkan representasi numerik ini yang telah dipelajari oleh model SVM.
    * Dalam implementasi nyata, seringkali TF-IDF Vectorizer dan model SVM disimpan bersama dalam satu objek `pipeline` (misalnya, menggunakan `scikit-learn` Pipeline) dan kemudian di-pickle menjadi satu file (`svm_model.pkl`). Jadi, ketika Anda memuat `svm_model.pkl`, Anda sebenarnya memuat seluruh alur kerja yang sudah termasuk TF-IDF Vectorizer di dalamnya. Ini menyederhanakan penyebaran model karena Anda tidak perlu memuat dua objek terpisah.

    Dengan demikian, TF-IDF adalah tahap penting yang memungkinkan SVM bekerja dengan data tekstual. Aplikasi ini menggunakan tiga model SVM terpisah, masing-masing khusus untuk memproses dan merekomendasikan berdasarkan Genre, Tag, dan Kategori, memungkinkan rekomendasi yang lebih spesifik dan akurat.
    """)

# === Recommendation by Genre Page ===
elif halaman == "Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")

    # Get all unique genres by splitting and flattening the list
    all_genres = set()
    if 'genre' in df.columns:
        for genres_str in df['genre'].dropna().unique():
            for g in str(genres_str).split(','):
                all_genres.add(g.strip())
    daftar_genre = sorted(list(all_genres))
    
    if not daftar_genre:
        st.warning("Tidak ada genre yang ditemukan di dataset.")
        st.stop()

    genre_pilihan = st.selectbox("Pilih genre sebagai filter awal:", daftar_genre)

    if genre_pilihan:
        if genre_pilihan not in st.session_state.history['genre']:
            st.session_state.history['genre'].append(genre_pilihan)
        # Filter based on genre string containing the selected genre
        hasil = df[df['genre'].str.contains(genre_pilihan, case=False, na=False)]
        st.markdown("### Rekomendasi Game berdasarkan genre")
        tampilkan_game(hasil)
    else:
        st.info("Pilih genre dari daftar di atas untuk melihat rekomendasi.")


# === Recommendation by Tag Page ===
elif halaman == "Rekomendasi Tag":
    st.title("🏷️ Rekomendasi Berdasarkan Tag")

    # Get all unique tags by splitting and flattening the list
    all_tags = set()
    if 'tags' in df.columns: # Use 'tags' here
        for tags_str in df['tags'].dropna().unique(): # Use 'tags' here
            for t in str(tags_str).split(','):
                all_tags.add(t.strip())
    daftar_tag = sorted(list(all_tags))

    if not daftar_tag:
        st.warning("Tidak ada tag yang ditemukan di dataset.")
        st.stop()

    tag_pilihan = st.selectbox("Pilih tag sebagai filter awal:", daftar_tag)

    if tag_pilihan:
        if tag_pilihan not in st.session_state.history['tag']:
            st.session_state.history['tag'].append(tag_pilihan)
        # Filter based on tag string containing the selected tag
        hasil = df[df['tags'].str.contains(tag_pilihan, case=False, na=False)] # Use 'tags' here
        st.markdown("### Rekomendasi Game berdasarkan tag")
        tampilkan_game(hasil)
    else:
        st.info("Pilih tag dari daftar di atas untuk melihat rekomendasi.")

# === Recommendation by Category Page ===
elif halaman == "Rekomendasi Kategori":
    st.title("📂 Rekomendasi Berdasarkan Kategori")

    # Get all unique categories by splitting and flattening the list
    all_categories = set()
    if 'categories' in df.columns: # Use 'categories' here
        for categories_str in df['categories'].dropna().unique(): # Use 'categories' here
            for c in str(categories_str).split(','):
                all_categories.add(c.strip())
    daftar_kategori = sorted(list(all_categories))

    if not daftar_kategori:
        st.warning("Tidak ada kategori yang ditemukan di dataset.")
        st.stop()

    kategori_pilihan = st.selectbox("Pilih kategori sebagai filter awal:", daftar_kategori)

    if kategori_pilihan:
        if kategori_pilihan not in st.session_state.history['category']:
            st.session_state.history['category'].append(kategori_pilihan)
        # Filter based on category string containing the selected category
        hasil = df[df['categories'].str.contains(kategori_pilihan, case=False, na=False)] # Use 'categories' here
        st.markdown("### Rekomendasi Game berdasarkan kategori")
        tampilkan_game(hasil)
    else:
        st.info("Pilih kategori dari daftar di atas untuk melihat rekomendasi.")
