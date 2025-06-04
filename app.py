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

# Konfigurasi Streamlit
st.set_page_config(page_title="Game Recommendation System", layout="wide")

# Fungsi untuk membersihkan data
def clean_and_format_data(df):
    """Clean and format the game dataset"""
    try:
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        if 'Game Name' in df.columns:
            df.rename(columns={'Game Name': 'Name'}, inplace=True)
        
        # Process list columns safely
        list_columns = ['Tags', 'Categories', 'Genre']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').apply(
                    lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
                )
        
        # Ensure required columns exist
        for col in ['Name', 'Header Image', 'Short Description']:
            if col not in df.columns:
                df[col] = ''
        
        return df
    
    except Exception as e:
        st.error(f"Data cleaning error: {str(e)}")
        return df

# Fungsi rekomendasi
def recommend_games(df, cosine_sim, index, n_recommendations=10):
    """Generate game recommendations based on cosine similarity"""
    try:
        sim_scores = list(enumerate(cosine_sim[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        game_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        return df.iloc[game_indices]
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()

# Load data dari GitHub
@st.cache_resource
def load_data():
    try:
        zip_url = "https://github.com/Skripsiade123/Skripsi/raw/main/Dataset.zip"
        response = requests.get(zip_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file = next(f for f in z.namelist() if f.endswith('.csv'))
            with z.open(csv_file) as f:
                df = pd.read_csv(f)
        return clean_and_format_data(df)
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

# Load model dari GitHub
@st.cache_resource
def load_model(feature_type):
    model_files = {
        "Genre": ("svm_model.pkl", "tfidf_vectorizer.pkl"),
        "Tag": ("svm_model_tags.pkl", "tfidf_vectorizer_tags.pkl"),
        "Kategori": ("svm_model_categories.pkl", "tfidf_vectorizer_categories.pkl")
    }
    
    try:
        model_url = f"https://github.com/Skripsiade123/Skripsi/raw/main/{model_files[feature_type][0]}"
        vec_url = f"https://github.com/Skripsiade123/Skripsi/raw/main/{model_files[feature_type][1]}"
        
        model = joblib.load(io.BytesIO(requests.get(model_url).content)
        vectorizer = joblib.load(io.BytesIO(requests.get(vec_url).content)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

# UI Components
def display_game(game):
    """Display game card"""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(game['Header Image'] if game['Header Image'] else "https://via.placeholder.com/300x150?text=No+Image", 
                width=200)
    with col2:
        st.subheader(game['Name'])
        st.write(game['Short Description'])
        st.caption(f"**Genre:** {', '.join(game.get('Genre', ['N/A']))}")
        st.caption(f"**Tags:** {', '.join(game.get('Tags', ['N/A']))}")
        st.caption(f"**Categories:** {', '.join(game.get('Categories', ['N/A']))}")
    st.divider()

# Main App
def main():
    df = load_data()
    
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Pilih Halaman", [
        "Beranda", 
        "Rekomendasi Genre", 
        "Rekomendasi Tag", 
        "Rekomendasi Kategori"
    ])
    
    if page == "Beranda":
        st.title("🎮 Sistem Rekomendasi Game")
        st.write("Selamat datang! Pilih jenis rekomendasi di menu sidebar.")
        
    elif "Rekomendasi" in page:
        feature_type = page.split()[-1]
        st.title(f"Rekomendasi Berdasarkan {feature_type}")
        
        # Pilihan fitur
        feature_list = sorted({
            f for col in [feature_type] 
            for lst in df.get(col, pd.Series([])) 
            for f in lst
        })
        
        selected_feature = st.selectbox(f"Pilih {feature_type}", feature_list)
        
        # Filter data
        df_filtered = df[df[feature_type].apply(
            lambda x: selected_feature in x if isinstance(x, list) else False
        ).reset_index(drop=True)
        
        if not df_filtered.empty:
            model, vectorizer = load_model(feature_type)
            if model and vectorizer:
                feature_col = "Tags" if feature_type == "Genre" else "Categories"
                df_filtered['features'] = df_filtered[feature_col].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else ''
                )
                
                tfidf_matrix = vectorizer.transform(df_filtered['features'])
                cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
                
                for i, game in recommend_games(df_filtered, cosine_sim, 0).iterrows():
                    display_game(game)
            else:
                st.warning("Model tidak tersedia")
        else:
            st.warning(f"Tidak ditemukan game dengan {feature_type.lower()} ini")

if __name__ == "__main__":
    main()
