# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# --- Surprise Library Imports ---
from surprise import Dataset, Reader, SVD
import surprise.dump
# --- End Surprise Library Imports ---

DATA_DIR = 'data/'
MODELS_DIR = 'models/'

# --- Configuration ---
CF_MODEL_FILENAME = 'surprise_svd_model_tuned.joblib' # Use your actual tuned model filename
DEFAULT_COLLAB_WEIGHT = 0.8
DEFAULT_CONTENT_WEIGHT = 0.2
DEFAULT_MIN_RATING_CONTENT_PROFILE = 3.5

# --- Load Core Data (Cached) ---
@st.cache_data
def load_core_data(data_path):
    movies_file = os.path.join(data_path, 'movies_processed.csv') # Assumes tmdbId is here
    ratings_file = os.path.join(data_path, 'ratings.csv')
    links_file = os.path.join(data_path, 'links.csv') # Load links.csv for imdbId
    cosine_sim_file = os.path.join(data_path, 'cosine_similarity_content.npy')

    # Check for essential files first
    essential_files = [movies_file, ratings_file, cosine_sim_file, links_file]
    if not all(os.path.exists(f) for f in essential_files):
        missing = [f for f in essential_files if not os.path.exists(f)]
        st.error(f"Essential data file(s) not found: {', '.join(missing)}. Please ensure they exist in {data_path}.")
        return None, None, None, None, None # Adjusted return for links_df

    movies_df = pd.read_csv(movies_file)
    ratings_df = pd.read_csv(ratings_file)
    links_df = pd.read_csv(links_file) # Load links
    try:
        cosine_sim_content = np.load(cosine_sim_file)
    except Exception as e:
        st.error(f"Error loading 'cosine_similarity_content.npy': {e}")
        return None, None, None, None, None

    # Merge imdbId from links_df into movies_df if not already present
    # movies_processed.csv from your notebook already includes tmdbId.
    # We'll ensure imdbId is also available.
    if 'imdbId' not in movies_df.columns and 'movieId' in links_df.columns:
        movies_df = pd.merge(movies_df, links_df[['movieId', 'imdbId']], on='movieId', how='left')
    
    # Ensure tmdbId is present as well, just in case it wasn't in movies_processed.csv
    if 'tmdbId' not in movies_df.columns and 'movieId' in links_df.columns:
         movies_df = pd.merge(movies_df, links_df[['movieId', 'tmdbId']], on='movieId', how='left')


    if not movies_df.empty:
        indices_map_movieId_to_df_idx = pd.Series(movies_df.index, index=movies_df['movieId']).drop_duplicates()
    else:
        st.error("Movies DataFrame is empty.")
        return None, None, None, None, None
        
    if cosine_sim_content.shape[0] != len(movies_df):
        st.warning(f"Mismatch: cosine_sim_content has {cosine_sim_content.shape[0]} rows, movies_df has {len(movies_df)} rows.")
        
    return movies_df, ratings_df, links_df, cosine_sim_content, indices_map_movieId_to_df_idx

# --- Load Surprise CF Model (Cached) ---
@st.cache_resource
def load_surprise_cf_model(model_filename, models_path):
    full_model_path = os.path.join(models_path, model_filename)
    try:
        _, loaded_algo = surprise.dump.load(full_model_path)
        return loaded_algo
    except FileNotFoundError:
        st.error(f"Surprise CF model not found at {full_model_path}. Train and save it first.")
        return None
    except Exception as e: st.error(f"Error loading Surprise model: {e}"); return None

# --- Build Surprise Trainset (Cached) ---
@st.cache_resource
def get_full_surprise_trainset(_ratings_df_for_trainset):
    if _ratings_df_for_trainset is None or _ratings_df_for_trainset.empty: return None
    reader = Reader(rating_scale=(_ratings_df_for_trainset['rating'].min(), _ratings_df_for_trainset['rating'].max()))
    data = Dataset.load_from_df(_ratings_df_for_trainset[['userId', 'movieId', 'rating']], reader)
    return data.build_full_trainset()

# --- Helper to add links to recommendation DataFrames ---
def add_movie_links_to_recommendations(recs_df, movies_with_links_df):
    if recs_df.empty or 'movieId' not in recs_df.columns:
        return recs_df
    
    # Ensure movies_with_links_df has the necessary ID columns
    if 'imdbId' not in movies_with_links_df.columns or 'tmdbId' not in movies_with_links_df.columns:
        st.warning("imdbId or tmdbId not found in movies data. Links cannot be generated fully.")
        # Add empty columns if they don't exist to prevent merge errors, but links will be empty
        if 'imdbId' not in movies_with_links_df.columns: movies_with_links_df['imdbId'] = pd.NA
        if 'tmdbId' not in movies_with_links_df.columns: movies_with_links_df['tmdbId'] = pd.NA

    recs_df_with_links = pd.merge(recs_df, 
                                  movies_with_links_df[['movieId', 'imdbId', 'tmdbId']], 
                                  on='movieId', 
                                  how='left')
    
    # Create actual link URLs
    recs_df_with_links['IMDb Link'] = recs_df_with_links['imdbId'].apply(
        lambda x: f"http://www.imdb.com/title/tt{str(x).zfill(7)}/" if pd.notna(x) else ""
    )
    recs_df_with_links['TMDb Link'] = recs_df_with_links['tmdbId'].apply(
        lambda x: f"https://www.themoviedb.org/movie/{int(x)}" if pd.notna(x) else ""
    )
    return recs_df_with_links.drop(columns=['imdbId', 'tmdbId'], errors='ignore')


# --- Recommendation Functions (Adapted for Surprise CF) ---
# (get_collaborative_recommendations_app, get_content_recommendations_app, get_hybrid_recommendations_app
#  remain the same as the previous version you provided, they already return DataFrames with 'movieId')

def get_collaborative_recommendations_app(user_id, surprise_algo_app, app_trainset, 
                                          movies_df_app, all_ratings_df_for_exclusion_app, top_n=10):
    if surprise_algo_app is None or app_trainset is None:
        return pd.DataFrame()

    all_movie_raw_ids = [app_trainset.to_raw_iid(inner_id) for inner_id in app_trainset.all_items()]
    rated_movie_ids = all_ratings_df_for_exclusion_app[all_ratings_df_for_exclusion_app['userId'] == user_id]['movieId'].unique().tolist()
    
    recommendations = []
    try: _ = app_trainset.to_inner_uid(user_id) 
    except ValueError: pass # User not in CF trainset

    for movie_id in all_movie_raw_ids:
        if movie_id not in rated_movie_ids:
            prediction = surprise_algo_app.predict(uid=user_id, iid=movie_id)
            movie_detail = movies_df_app[movies_df_app['movieId'] == movie_id]
            if not movie_detail.empty:
                recommendations.append({
                    'movieId': movie_id,
                    'title_clean': movie_detail['title_clean'].iloc[0],
                    'predicted_collaborative_score': prediction.est,
                    'genres_str': movie_detail.get('genres_str', pd.Series([""])).iloc[0]
                })
    
    recs_df = pd.DataFrame(recommendations)
    if not recs_df.empty:
        recs_df = recs_df.sort_values(by='predicted_collaborative_score', ascending=False).head(top_n)
    return recs_df

def get_content_recommendations_app(user_id, ratings_df_content_app, movies_df_content_app, 
                                     cosine_sim_matrix_app, indices_map_content_app, 
                                     top_n=10, min_rating_threshold=4.0):
    user_ratings = ratings_df_content_app[(ratings_df_content_app['userId'] == user_id) & (ratings_df_content_app['rating'] >= min_rating_threshold)]
    if user_ratings.empty: return pd.DataFrame()
    
    liked_movie_ids = user_ratings['movieId'].tolist()
    liked_movie_indices = [indices_map_content_app[mid] for mid in liked_movie_ids if mid in indices_map_content_app and 0 <= indices_map_content_app[mid] < cosine_sim_matrix_app.shape[0]]

    if not liked_movie_indices: return pd.DataFrame()
    
    try: user_profile_sim_vector = np.mean(cosine_sim_matrix_app[liked_movie_indices, :], axis=0)
    except: return pd.DataFrame()
    
    sim_scores_series = pd.Series(user_profile_sim_vector, index=movies_df_content_app.index).sort_values(ascending=False)
    rated_overall = ratings_df_content_app[ratings_df_content_app['userId'] == user_id]['movieId'].unique().tolist()
    
    recommendations = []
    for movie_df_idx, score in sim_scores_series.items(): 
        if len(recommendations) >= top_n: break
        movie_info = movies_df_content_app.loc[movie_df_idx] 
        movie_id_rec = movie_info['movieId']
        if movie_id_rec not in rated_overall:
            genres_val = movie_info.get('genres_str', "")
            if not isinstance(genres_val, str) or pd.isna(genres_val): genres_val = ""
            recommendations.append({
                'movieId': movie_id_rec, 'title_clean': movie_info['title_clean'],
                'predicted_content_score': score, 'genres_str': genres_val
            })
    return pd.DataFrame(recommendations)

def get_hybrid_recommendations_app(user_id, surprise_algo_hybrid_app, app_trainset_hybrid, 
                                   cosine_sim_content_hybrid_app, indices_map_content_hybrid_app, 
                                   ratings_df_hybrid_app, movies_df_hybrid_app, 
                                   top_n=10, collab_weight=0.5, content_weight=0.5, 
                                   min_rating_threshold_content_profile=4.0):
    num_initial_recs = top_n * 3 
    content_recs = get_content_recommendations_app(user_id, ratings_df_hybrid_app, movies_df_hybrid_app, cosine_sim_content_hybrid_app, indices_map_content_hybrid_app, top_n=num_initial_recs, min_rating_threshold=min_rating_threshold_content_profile)
    collab_recs = get_collaborative_recommendations_app(user_id, surprise_algo_hybrid_app, app_trainset_hybrid, movies_df_hybrid_app, ratings_df_hybrid_app, top_n=num_initial_recs)

    no_content = content_recs.empty
    no_collab = collab_recs.empty
    if no_content and no_collab: return pd.DataFrame()
    
    scaler = MinMaxScaler()
    if not no_content and 'predicted_content_score' in content_recs:
        if content_recs['predicted_content_score'].nunique() > 1: content_recs['normalized_score'] = scaler.fit_transform(content_recs[['predicted_content_score']])
        elif len(content_recs) > 0: content_recs['normalized_score'] = 0.5 if content_recs['predicted_content_score'].iloc[0] != 0 else 0.0
        else: content_recs['normalized_score'] = 0.0; no_content = True
    else:
        if 'normalized_score' not in content_recs.columns : content_recs = content_recs.assign(normalized_score=pd.NA)
        no_content = True
        
    if not no_collab and 'predicted_collaborative_score' in collab_recs:
        if collab_recs['predicted_collaborative_score'].nunique() > 1: collab_recs['normalized_score'] = scaler.fit_transform(collab_recs[['predicted_collaborative_score']])
        elif len(collab_recs) > 0: collab_recs['normalized_score'] = 0.5 if collab_recs['predicted_collaborative_score'].iloc[0] !=0 else 0.0
        else: collab_recs['normalized_score'] = 0.0; no_collab = True
    else:
        if 'normalized_score' not in collab_recs.columns : collab_recs = collab_recs.assign(normalized_score=pd.NA)
        no_collab = True

    if no_content and not no_collab: return collab_recs.head(top_n).rename(columns={'predicted_collaborative_score': 'hybrid_score'})[['movieId', 'title_clean', 'hybrid_score', 'genres_str']]
    if no_collab and not no_content: return content_recs.head(top_n).rename(columns={'predicted_content_score': 'hybrid_score'})[['movieId', 'title_clean', 'hybrid_score', 'genres_str']]
    if no_collab and no_content: return pd.DataFrame()

    merged_recs = pd.merge(content_recs[['movieId', 'title_clean', 'genres_str', 'normalized_score']], collab_recs[['movieId', 'normalized_score']], on='movieId', how='outer', suffixes=('_content', '_collab'))
    merged_recs['normalized_score_content'] = merged_recs['normalized_score_content'].fillna(0)
    merged_recs['normalized_score_collab'] = merged_recs['normalized_score_collab'].fillna(0)
    
    if 'title_clean_content' in merged_recs.columns:
        merged_recs['title_clean'] = merged_recs['title_clean_content'] 
        merged_recs['genres_str'] = merged_recs['genres_str_content']
        merged_recs.drop(columns=['title_clean_content', 'genres_str_content'], inplace=True, errors='ignore')

    merged_recs.dropna(subset=['title_clean'], inplace=True)
    merged_recs['hybrid_score'] = (collab_weight * merged_recs['normalized_score_collab'] + content_weight * merged_recs['normalized_score_content'])
    final_recs = merged_recs.sort_values(by='hybrid_score', ascending=False).drop_duplicates(subset=['movieId'], keep='first').head(top_n)
    return final_recs[['movieId', 'title_clean', 'hybrid_score', 'genres_str']].reset_index(drop=True)


# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Movie Recommender")
    st.title("🎬 Hybrid Movie Recommendation System")

    # Load core data, now including links_df implicitly through movies_df
    movies_df, ratings_df, _, cosine_sim_content, indices_map_movieId_to_df_idx = load_core_data(DATA_DIR)
    # The underscore discards the raw links_df as movies_df should now contain the merged IDs

    if movies_df is None or ratings_df is None or cosine_sim_content is None or indices_map_movieId_to_df_idx is None:
        st.error("Failed to load essential data. Application cannot proceed."); return
    if movies_df.empty or ratings_df.empty:
        st.error("Movies or Ratings data is empty. Application cannot proceed."); return

    surprise_cf_algo_app = load_surprise_cf_model(CF_MODEL_FILENAME, MODELS_DIR)
    app_trainset = get_full_surprise_trainset(ratings_df.copy())

    if surprise_cf_algo_app is None or app_trainset is None:
        st.error("CF model components not loaded. Check logs."); return

    st.sidebar.header("👤 User Preferences")
    available_user_ids = sorted(ratings_df['userId'].unique())
    selected_user_id = st.sidebar.selectbox("Select User ID:", available_user_ids, index=0)
    top_n_recs = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)
    
    st.sidebar.subheader("⚙️ Hybrid Model Weights")
    collab_w_ui = st.sidebar.slider("Collaborative Weight:", 0.0, 1.0, DEFAULT_COLLAB_WEIGHT, 0.05, "%.2f")
    content_w_ui = 1.0 - collab_w_ui
    st.sidebar.markdown(f"*Content-Based Weight: `{content_w_ui:.2f}`*")
    min_rating_thresh_ui = st.sidebar.slider("Min Rating for Content Profile:", 1.0, 5.0, DEFAULT_MIN_RATING_CONTENT_PROFILE, 0.5, "%.1f")

    if selected_user_id:
        st.header(f"✨ Recommendations for User ID: {selected_user_id}")

        if st.sidebar.button("🔄 Get Recommendations", type="primary", key="get_recs_button"):
            with st.spinner(f"Generating recommendations..."):
                hybrid_recs_raw = get_hybrid_recommendations_app(
                    selected_user_id, surprise_cf_algo_app, app_trainset, cosine_sim_content, 
                    indices_map_movieId_to_df_idx, ratings_df, movies_df, top_n_recs, 
                    collab_w_ui, content_w_ui, min_rating_thresh_ui
                )
                hybrid_recommendations = add_movie_links_to_recommendations(hybrid_recs_raw, movies_df)


            if not hybrid_recommendations.empty:
                st.subheader("🏆 Top Hybrid Movie Recommendations")
                # Define column configuration for links
                st.dataframe(
                    hybrid_recommendations[['title_clean', 'hybrid_score', 'genres_str', 'IMDb Link', 'TMDb Link']],
                    column_config={
                        "hybrid_score": st.column_config.NumberColumn(format="%.4f"),
                        "IMDb Link": st.column_config.LinkColumn("IMDb", display_text="Go to IMDb"),
                        "TMDb Link": st.column_config.LinkColumn("TMDb", display_text="Go to TMDb")
                    },
                    hide_index=True
                )
            else:
                st.info("No hybrid recommendations found.")

            with st.expander("🔍 Show Individual Model Recommendations"):
                st.markdown("---")
                st.subheader("🧩 Content-Based Only")
                with st.spinner("Generating content-based..."):
                    cb_recs_raw = get_content_recommendations_app(selected_user_id, ratings_df, movies_df, cosine_sim_content, indices_map_movieId_to_df_idx, top_n_recs, min_rating_thresh_ui)
                    content_recs_only = add_movie_links_to_recommendations(cb_recs_raw, movies_df)
                if not content_recs_only.empty:
                    st.dataframe(content_recs_only[['title_clean', 'predicted_content_score', 'genres_str', 'IMDb Link', 'TMDb Link']],
                                 column_config={
                                     "predicted_content_score": st.column_config.NumberColumn(format="%.4f"),
                                     "IMDb Link": st.column_config.LinkColumn("IMDb"), "TMDb Link": st.column_config.LinkColumn("TMDb")},
                                 hide_index=True)
                else: st.caption("No content-based recs.")

                st.markdown("---")
                st.subheader("🤝 Collaborative Filtering Only (Surprise)")
                with st.spinner("Generating collaborative..."):
                    cf_recs_raw = get_collaborative_recommendations_app(selected_user_id, surprise_cf_algo_app, app_trainset, movies_df, ratings_df, top_n_recs)
                    collab_recs_only = add_movie_links_to_recommendations(cf_recs_raw, movies_df)
                if not collab_recs_only.empty:
                    st.dataframe(collab_recs_only[['title_clean', 'predicted_collaborative_score', 'genres_str', 'IMDb Link', 'TMDb Link']],
                                 column_config={
                                     "predicted_collaborative_score": st.column_config.NumberColumn(format="%.4f"),
                                     "IMDb Link": st.column_config.LinkColumn("IMDb"), "TMDb Link": st.column_config.LinkColumn("TMDb")},
                                 hide_index=True)
                else: st.caption("No collaborative recs.")
        else: st.info("Click 'Get Recommendations' in the sidebar.")
    else: st.info("Please select a User ID.")
            
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Dataset Info:**")
    if not movies_df.empty: st.sidebar.markdown(f"- Movies: {len(movies_df):,}")
    if not ratings_df.empty: 
        st.sidebar.markdown(f"- Ratings: {len(ratings_df):,}")
        st.sidebar.markdown(f"- Unique Users: {ratings_df['userId'].nunique():,}")

if __name__ == '__main__':
    main()