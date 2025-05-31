import streamlit as st
import pandas as pd
import zipfile

st.set_page_config(page_title="Sistem Rekomendasi Game", layout="wide")

# Upload file ZIP yang berisi Dataset.csv
uploaded_zip = st.file_uploader("Upload file ZIP yang berisi Dataset.csv", type="zip")

if uploaded_zip is not None:
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            if "Dataset.csv" in zip_ref.namelist():
                with zip_ref.open('Dataset.csv') as csv_file:
                    df = pd.read_csv(csv_file)

                # Preprocessing
                df['Genre'] = df['Genre'].fillna("").apply(lambda x: [i.strip() for i in x.split(",")])
                df['Tags'] = df['Tags'].fillna("").apply(lambda x: [i.strip().strip('"') for i in x.split(",")])
                df['Categories'] = df['Categories'].fillna("").apply(lambda x: [i.strip() for i in x.split(",")])

                # Sidebar
                st.sidebar.title("Navigasi")
                page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Genre", "Tags", "Categories"])

                # Halaman Beranda
                if page == "Beranda":
                    st.title("🎮 Sistem Rekomendasi Game")
                    st.write("Berikut adalah beberapa game dari Steam:")
                    for _, row in df.iterrows():
                        st.subheader(row['Name'])
                        st.image(row['Header Image'], width=300)
                        st.write(row['Short Description'])
                        st.write(f"**Genre:** {', '.join(row['Genre'])}")
                        st.markdown("---")

                # Halaman Genre
                elif page == "Genre":
                    st.title("🎯 Pilih Genre")
                    all_genres = sorted(set(g for genres in df['Genre'] for g in genres))
                    selected_genre = st.selectbox("Klik untuk memilih genre:", all_genres)

                    st.markdown("---")
                    st.subheader(f"Game dengan genre **{selected_genre}**:")
                    filtered = df[df['Genre'].apply(lambda x: selected_genre in x)]
                    if not filtered.empty:
                        for _, row in filtered.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game dengan genre ini.")

                # Halaman Tags
                elif page == "Tags":
                    st.title("🏷️ Pilih Tag")
                    all_tags = sorted(set(tag for tags in df['Tags'] for tag in tags))
                    selected_tag = st.selectbox("Klik untuk memilih tag:", all_tags)

                    st.markdown("---")
                    st.subheader(f"Game dengan tag **{selected_tag}**:")
                    filtered = df[df['Tags'].apply(lambda x: selected_tag in x)]
                    if not filtered.empty:
                        for _, row in filtered.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game dengan tag ini.")

                # Halaman Categories
                elif page == "Categories":
                    st.title("📂 Pilih Kategori")
                    all_categories = sorted(set(cat for cats in df['Categories'] for cat in cats))
                    selected_cat = st.selectbox("Klik untuk memilih kategori:", all_categories)

                    st.markdown("---")
                    st.subheader(f"Game dengan kategori **{selected_cat}**:")
                    filtered = df[df['Categories'].apply(lambda x: selected_cat in x)]
                    if not filtered.empty:
                        for _, row in filtered.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game dengan kategori ini.")
            else:
                st.error("File 'Dataset.csv' tidak ditemukan di dalam ZIP.")
    except zipfile.BadZipFile:
        st.error("File ZIP tidak valid.")
else:
    st.info("Silakan upload file ZIP yang berisi Dataset.csv.")
