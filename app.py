import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

                df['combined_features'] = df['Genre'].apply(lambda x: ' '.join(x)) + ' ' + \
                                          df['Tags'].apply(lambda x: ' '.join(x)) + ' ' + \
                                          df['Categories'].apply(lambda x: ' '.join(x))

                TARGET_GENRE = 'Action'
                df['is_target_genre'] = df['Genre'].apply(lambda genres: TARGET_GENRE in genres)

                tfidf_vectorizer = TfidfVectorizer()
                X = tfidf_vectorizer.fit_transform(df['combined_features'])
                y = df['is_target_genre']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                svm_model = SVC(kernel='linear', probability=True)
                svm_model.fit(X_train, y_train)

                df_cleaned = df.copy()

                # Sidebar
                st.sidebar.title("Navigasi")
                page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Genre", "Tags", "Categories", "Rekomendasi Genre (SVM)", "Rekomendasi Tags (SVM)", "Rekomendasi Categories (SVM)"])

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

                # Rekomendasi Genre SVM
                elif page == "Rekomendasi Genre (SVM)":
                    selected_genre = st.selectbox("Pilih genre sebagai filter awal:", sorted(set(g for genres in df['Genre'] for g in genres)))
                    filtered_games = df_cleaned[df_cleaned['Genre'].apply(lambda x: selected_genre in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        st.subheader(f"Rekomendasi Game berdasarkan genre '{selected_genre}' dan validasi SVM untuk target '{TARGET_GENRE}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game yang cocok.")

                # Rekomendasi Tags SVM
                elif page == "Rekomendasi Tags (SVM)":
                    TARGET_TAG = 'Indie'
                    selected_tag = st.selectbox("Pilih tag sebagai filter awal:", sorted(set(t for tags in df['Tags'] for t in tags)))
                    filtered_games = df_cleaned[df_cleaned['Tags'].apply(lambda x: selected_tag in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        st.subheader(f"Rekomendasi Game berdasarkan tag '{selected_tag}' dan validasi SVM untuk target '{TARGET_TAG}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game yang cocok.")

                # Rekomendasi Categories SVM
                elif page == "Rekomendasi Categories (SVM)":
                    selected_cat = st.selectbox("Pilih kategori sebagai filter awal:", sorted(set(c for cats in df['Categories'] for c in cats)))
                    filtered_games = df_cleaned[df_cleaned['Categories'].apply(lambda x: selected_cat in x)]
                    if not filtered_games.empty:
                        tfidf_filtered = tfidf_vectorizer.transform(filtered_games['combined_features'])
                        filtered_games['Predicted'] = svm_model.predict(tfidf_filtered)
                        recommended = filtered_games[filtered_games['Predicted'] == 1]
                        st.subheader(f"Rekomendasi Game berdasarkan kategori '{selected_cat}' dan validasi SVM untuk target '{TARGET_GENRE}':")
                        for _, row in recommended.iterrows():
                            st.subheader(row['Name'])
                            st.image(row['Header Image'], width=300)
                            st.write(row['Short Description'])
                            st.markdown("---")
                    else:
                        st.warning("Tidak ada game yang cocok.")

            else:
                st.error("File 'Dataset.csv' tidak ditemukan di dalam ZIP.")
    except zipfile.BadZipFile:
        st.error("File ZIP tidak valid.")
else:
    st.info("Silakan upload file ZIP yang berisi Dataset.csv.")
