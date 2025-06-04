import streamlit as st
import pandas as pd
import os
import json
import zipfile
import requests
import io
import joblib
from datetime import datetime
from sklearn.metrics.pairwise import linear_kernel

# Configuration
st.set_page_config(page_title="Game Recommendation System", layout="wide")

# Constants
HISTORY_FILE = "history.json"
GITHUB_REPO = "https://github.com/Skripsiade123/Skripsi/raw/main/"
DATASET_ZIP = "Dataset.zip"

# Helper functions
def clean_and_format_data(df):
    """Clean and format the game dataset"""
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    if 'Game Name' in df.columns:
        df.rename(columns={'Game Name': 'Name'}, inplace=True)
    
    # Process list-type columns
    list_columns = ['Tags', 'Categories', 'Genre']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].fillna('').apply(lambda x: [i.strip() for i in x.split(',') if isinstance(x, str) else [])

    # Ensure essential columns exist
    required_columns = ['Name', 'Header Image', 'Short Description']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    return df

def recommend_games(df, cosine_sim, index, n_recommendations=10):
    """Generate game recommendations based on cosine similarity"""
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]  # Exclude self and get top N
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices]

def load_csv_from_github_zip(zip_url):
    """Load CSV dataset from GitHub zip file"""
    try:
        response = requests.get(zip_url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = next(name for name in z.namelist() if name.endswith('.csv'))
            with z.open(csv_filename) as f:
                return pd.read_csv(f)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

# History functions
def load_saved_history():
    """Load recommendation history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_entry(entry):
    """Save a new entry to recommendation history"""
    history = load_saved_history()
    history.insert(0, entry)
    # Keep only the last 20 entries
    history = history[:20]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except:
        pass

# Model loading with caching
@st.cache_resource(ttl=3600)
def load_model_and_vectorizer(feature_type):
    """Load the appropriate model and vectorizer based on feature type"""
    model_files = {
        "Genre": ("svm_model.pkl", "tfidf_vectorizer.pkl"),
        "Tag": ("svm_model_tags.pkl", "tfidf_vectorizer_tags.pkl"),
        "Kategori": ("svm_model_categories.pkl", "tfidf_vectorizer_categories.pkl")
    }
    
    if feature_type not in model_files:
        return None, None
    
    model_file, vec_file = model_files[feature_type]
    
    try:
        model = joblib.load(io.BytesIO(requests.get(GITHUB_REPO + model_file).content))
        vectorizer = joblib.load(io.BytesIO(requests.get(GITHUB_REPO + vec_file).content))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load data
df = load_csv_from_github_zip(GITHUB_REPO + DATASET_ZIP)
df = clean_and_format_data(df)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = load_saved_history()

# Page navigation
st.sidebar.title("Game Recommendation System")
pages = {
    "Beranda": "Home",
    "Rekomendasi Genre": "Genre Recommendations",
    "Rekomendasi Tag": "Tag Recommendations",
    "Rekomendasi Kategori": "Category Recommendations",
    "Histori": "Recommendation History",
    "Tentang": "About"
}
page = st.sidebar.radio("Menu", list(pages.keys()))

def display_game_card(row):
    """Display a game card with information"""
    col1, col2 = st.columns([1, 3])
    with col1:
        if row['Header Image']:
            st.image(row['Header Image'], width=200)
        else:
            st.image("https://via.placeholder.com/200x100?text=No+Image", width=200)
    with col2:
        st.subheader(row['Name'])
        st.write(row['Short Description'])
        st.write(f"**Genre:** {', '.join(row.get('Genre', ['N/A']))}")
        st.write(f"**Tags:** {', '.join(row.get('Tags', ['N/A']))}")
        st.write(f"**Categories:** {', '.join(row.get('Categories', ['N/A']))}")
    st.markdown("---")

# Page rendering
if page == "Beranda":
    st.title("🎮 Sistem Rekomendasi Game")
    st.write("""
        Selamat datang di sistem rekomendasi game! Aplikasi ini membantu Anda menemukan game baru 
        berdasarkan preferensi Anda terhadap genre, tag, atau kategori tertentu.
    """)
    
    # Show recent recommendations from history
    if st.session_state.history:
        st.subheader("Rekomendasi Terakhir Anda")
        last_recommendations = st.session_state.history[0].get("recommended", [])
        if last_recommendations:
            recommended_games = df[df['Name'].isin(last_recommendations)]
            for _, row in recommended_games.iterrows():
                display_game_card(row)
        else:
            st.info("Belum ada rekomendasi yang tersimpan.")
    else:
        st.info("Belum ada riwayat rekomendasi. Silakan coba fitur rekomendasi kami!")

elif page == "Rekomendasi Genre":
    st.title("🎯 Rekomendasi Berdasarkan Genre")
    
    if not df.empty:
        genre_list = sorted({genre for sublist in df['Genre'] for genre in sublist})
        selected_genre = st.selectbox("Pilih Genre", genre_list, key="genre_select")
        
        df_filtered = df[df['Genre'].apply(lambda x: selected_genre in x)].reset_index(drop=True)
        
        if not df_filtered.empty:
            model, vectorizer = load_model_and_vectorizer("Genre")
            
            if model and vectorizer:
                df_filtered['features'] = df_filtered['Tags'].apply(lambda x: ' '.join(x))
                tfidf_matrix = vectorizer.transform(df_filtered['features'])
                cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                st.subheader(f"Rekomendasi Game dengan Genre: {selected_genre}")
                recommended = recommend_games(df_filtered, cosine_sim, 0)
                
                for _, row in recommended.iterrows():
                    display_game_card(row)
                
                # Save to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Genre",
                    "value": selected_genre,
                    "recommended": recommended['Name'].tolist()
                }
                save_history_entry(history_entry)
                st.session_state.history = load_saved_history()
            else:
                st.error("Model tidak dapat dimuat. Silakan coba lagi nanti.")
        else:
            st.warning(f"Tidak ditemukan game dengan genre: {selected_genre}")
    else:
        st.error("Dataset tidak tersedia. Silakan coba lagi nanti.")

elif page == "Rekomendasi Tag":
    st.title("🏷️ Rekomendasi Berdasarkan Tag")
    
    if not df.empty:
        tag_list = sorted({tag for sublist in df['Tags'] for tag in sublist})
        selected_tag = st.selectbox("Pilih Tag", tag_list, key="tag_select")
        
        df_filtered = df[df['Tags'].apply(lambda x: selected_tag in x)].reset_index(drop=True)
        
        if not df_filtered.empty:
            model, vectorizer = load_model_and_vectorizer("Tag")
            
            if model and vectorizer:
                df_filtered['features'] = df_filtered['Categories'].apply(lambda x: ' '.join(x))
                tfidf_matrix = vectorizer.transform(df_filtered['features'])
                cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                st.subheader(f"Rekomendasi Game dengan Tag: {selected_tag}")
                recommended = recommend_games(df_filtered, cosine_sim, 0)
                
                for _, row in recommended.iterrows():
                    display_game_card(row)
                
                # Save to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Tag",
                    "value": selected_tag,
                    "recommended": recommended['Name'].tolist()
                }
                save_history_entry(history_entry)
                st.session_state.history = load_saved_history()
            else:
                st.error("Model tidak dapat dimuat. Silakan coba lagi nanti.")
        else:
            st.warning(f"Tidak ditemukan game dengan tag: {selected_tag}")
    else:
        st.error("Dataset tidak tersedia. Silakan coba lagi nanti.")

elif page == "Rekomendasi Kategori":
    st.title("📦 Rekomendasi Berdasarkan Kategori")
    
    if not df.empty:
        cat_list = sorted({cat for sublist in df['Categories'] for cat in sublist})
        selected_cat = st.selectbox("Pilih Kategori", cat_list, key="cat_select")
        
        df_filtered = df[df['Categories'].apply(lambda x: selected_cat in x)].reset_index(drop=True)
        
        if not df_filtered.empty:
            model, vectorizer = load_model_and_vectorizer("Kategori")
            
            if model and vectorizer:
                df_filtered['features'] = df_filtered['Tags'].apply(lambda x: ' '.join(x))
                tfidf_matrix = vectorizer.transform(df_filtered['features'])
                cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                st.subheader(f"Rekomendasi Game dengan Kategori: {selected_cat}")
                recommended = recommend_games(df_filtered, cosine_sim, 0)
                
                for _, row in recommended.iterrows():
                    display_game_card(row)
                
                # Save to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "Kategori",
                    "value": selected_cat,
                    "recommended": recommended['Name'].tolist()
                }
                save_history_entry(history_entry)
                st.session_state.history = load_saved_history()
            else:
                st.error("Model tidak dapat dimuat. Silakan coba lagi nanti.")
        else:
            st.warning(f"Tidak ditemukan game dengan kategori: {selected_cat}")
    else:
        st.error("Dataset tidak tersedia. Silakan coba lagi nanti.")

elif page == "Histori":
    st.title("🕒 Histori Rekomendasi")
    
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history, 1):
            with st.expander(f"{i}. {entry['type']}: {entry['value']} ({entry['timestamp']})"):
                if entry.get("recommended"):
                    st.write("Game yang direkomendasikan:")
                    cols = st.columns(2)
                    for j, game in enumerate(entry['recommended']):
                        cols[j % 2].write(f"• {game}")
                else:
                    st.write("Tidak ada detail rekomendasi yang tersimpan.")
    else:
        st.info("Belum ada riwayat rekomendasi.")

elif page == "Tentang":
    st.title("ℹ️ Tentang Sistem Rekomendasi Game")
    st.write("""
        ### Metode yang Digunakan
        Sistem ini menggunakan pendekatan **Content-Based Filtering** dengan algoritma **Support Vector Machine (SVM)**.
        Fitur-fitur game seperti Genre, Tags, dan Categories diubah menjadi representasi vektor menggunakan **TF-IDF Vectorizer**.
        
        ### Cara Kerja
        1. Sistem menganalisis konten dari game-game yang ada
        2. Membangun profil preferensi berdasarkan pilihan pengguna
        3. Menghitung kesamaan (similarity) antar game menggunakan **cosine similarity**
        4. Merekomendasikan game yang paling mirip dengan preferensi pengguna
        
        ### Fitur Aplikasi
        - Rekomendasi berdasarkan Genre
        - Rekomendasi berdasarkan Tag
        - Rekomendasi berdasarkan Kategori
        - Penyimpanan histori rekomendasi
    """)
    
    st.markdown("---")
    st.write("""
        Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi tentang sistem rekomendasi game.
        Dataset yang digunakan berasal dari Steam Store.
    """)
