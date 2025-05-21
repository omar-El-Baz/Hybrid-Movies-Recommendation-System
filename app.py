# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import os

DATA_DIR = 'data/' 

# --- Load Data ---
@st.cache_data # Cache data loading
def load_data(data_path):
    movies_file = os.path.join(data_path, 'movies_processed.csv')
    ratings_file = os.path.join(data_path, 'ratings.csv')
    cosine_sim_file = os.path.join(data_path, 'cosine_similarity_content.npy')

    if not all(os.path.exists(f) for f in [movies_file, ratings_file, cosine_sim_file]):
        st.error(f"One or more data files not found in {data_path}. Please ensure movies_processed.csv, ratings.csv, and cosine_similarity_content.npy exist.")
        return None, None, None, None

    movies_df = pd.read_csv(movies_file)
    ratings_df = pd.read_csv(ratings_file)
    try:
        cosine_sim_content = np.load(cosine_sim_file)
    except Exception as e:
        st.error(f"Error loading 'cosine_similarity_content.npy': {e}")
        return None, None, None, None
    
    # Create indices mapping: movieId to DataFrame row index (for cosine_sim_content alignment)
    # This assumes movies_df is the same one used to generate cosine_sim_content and its order is preserved.
    if not movies_df.empty:
        indices_map = pd.Series(movies_df.index, index=movies_df['movieId']).drop_duplicates()
    else:
        indices_map = pd.Series(dtype='int64')
        
    if cosine_sim_content.shape[0] != len(movies_df):
        st.warning(f"Mismatch: cosine_sim_content has {cosine_sim_content.shape[0]} rows, movies_df has {len(movies_df)} rows. Ensure consistency.")
        # Potentially return None or handle error appropriately
        
    return movies_df, ratings_df, cosine_sim_content, indices_map

# --- Collaborative Filtering Model Training ---
@st.cache_resource # Cache the trained model and related components
def train_collaborative_filtering_model(ratings_df_cf, n_components=50, random_state=42): # Reduced n_components for faster UI demo
    st.write("Cache miss: Training Collaborative Filtering model...") # For debugging cache
    # Create user-item matrix
    user_item_matrix = ratings_df_cf.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    user_ids_map = user_item_matrix.index
    movie_ids_map = user_item_matrix.columns # These are actual movieIds
    
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd.fit_transform(user_item_matrix)
    
    # Reconstruct predicted ratings matrix for all users and items
    predicted_ratings_matrix = np.dot(user_factors, svd.components_)
    
    return predicted_ratings_matrix, user_item_matrix, user_ids_map, movie_ids_map

# --- Recommendation Functions (Adapted from 03_collaborative_hybrid_dev.py) ---

def get_collaborative_recommendations(user_id, pred_ratings_matrix, user_item_matrix_cf, 
                                      user_ids_map_cf, movie_ids_map_cf, 
                                      movies_df_cf, ratings_df_cf, top_n=10):
    if user_id not in user_ids_map_cf:
        # st.warning(f"User {user_id} not found in ratings data for collaborative filtering.")
        return pd.DataFrame(columns=['movieId', 'title_clean', 'predicted_collaborative_score', 'genres_str'])
    
    user_idx = user_ids_map_cf.get_loc(user_id)
    user_pred_ratings_row = pred_ratings_matrix[user_idx, :] # Predicted ratings for this user for all movies in movie_ids_map_cf
    
    rated_movie_ids = ratings_df_cf[ratings_df_cf['userId'] == user_id]['movieId'].tolist()
    
    recommendations = []
    # movie_ids_map_cf are the actual movieIds that form the columns of the user_item_matrix
    for i, movie_id_col in enumerate(movie_ids_map_cf):
        if movie_id_col not in rated_movie_ids:
            movie_detail = movies_df_cf[movies_df_cf['movieId'] == movie_id_col]
            if not movie_detail.empty:
                recommendations.append({
                    'movieId': movie_id_col,
                    'title_clean': movie_detail['title_clean'].iloc[0],
                    'predicted_collaborative_score': user_pred_ratings_row[i], # Score corresponds to movie_id_col at index i
                    'genres_str': movie_detail['genres_str'].iloc[0]
                })
    
    recs_df = pd.DataFrame(recommendations)
    if not recs_df.empty:
        recs_df = recs_df.sort_values(by='predicted_collaborative_score', ascending=False).head(top_n)
    return recs_df

def get_content_recommendations_for_user(user_id, ratings_df_content, movies_df_content, 
                                         cosine_sim_matrix, indices_map_content, 
                                         top_n=10, min_rating_threshold=4.0):
    user_ratings = ratings_df_content[(ratings_df_content['userId'] == user_id) & (ratings_df_content['rating'] >= min_rating_threshold)]
    
    if user_ratings.empty:
        return pd.DataFrame(columns=['movieId', 'title_clean', 'predicted_content_score', 'genres_str'])
    
    liked_movie_ids = user_ratings['movieId'].tolist()
    
    liked_movie_indices_in_cosine_sim = []
    for movie_id_liked in liked_movie_ids:
        if movie_id_liked in indices_map_content:
            idx = indices_map_content[movie_id_liked] # This is the row index in movies_df_content / cosine_sim_matrix
            if 0 <= idx < cosine_sim_matrix.shape[0]:
                 liked_movie_indices_in_cosine_sim.append(idx)

    if not liked_movie_indices_in_cosine_sim:
        return pd.DataFrame(columns=['movieId', 'title_clean', 'predicted_content_score', 'genres_str'])
    
    try:
        # Ensure all indices are valid before slicing
        valid_indices = [i for i in liked_movie_indices_in_cosine_sim if i < cosine_sim_matrix.shape[0]]
        if not valid_indices:
             return pd.DataFrame(columns=['movieId', 'title_clean', 'predicted_content_score', 'genres_str'])
        user_profile_sim = np.mean(cosine_sim_matrix[valid_indices, :], axis=0)
    except (IndexError, ValueError) as e:
        st.error(f"Error calculating user profile for content-based: {e}")
        return pd.DataFrame(columns=['movieId', 'title_clean', 'predicted_content_score', 'genres_str'])
    
    # sim_scores_series maps movie_df_content's index to similarity score
    sim_scores_series = pd.Series(user_profile_sim, index=movies_df_content.index) 
    sorted_sim_scores = sim_scores_series.sort_values(ascending=False)
    
    rated_movie_ids_by_user = ratings_df_content[ratings_df_content['userId'] == user_id]['movieId'].tolist()
    
    recommendations = []
    # movie_df_idx is the actual index from movies_df_content (e.g., 0, 1, 2...)
    for movie_df_idx, score in sorted_sim_scores.items(): 
        if len(recommendations) >= top_n:
            break
        
        # Use .loc for safety if movies_df_content.index is not a simple RangeIndex
        # However, if it's from pd.read_csv without set_index, .index is RangeIndex, and .iloc is fine too.
        # Original code used .iloc[movie_idx], where movie_idx was from enumerate(cosine_sim[idx])
        # Here, movie_df_idx is directly from movies_df_content.index
        movie_info = movies_df_content.loc[movie_df_idx] 
        movie_id_rec = movie_info['movieId']

        if movie_id_rec not in rated_movie_ids_by_user:
            recommendations.append({
                'movieId': movie_id_rec,
                'title_clean': movie_info['title_clean'],
                'predicted_content_score': score,
                'genres_str': movie_info['genres_str']
            })
            
    recs_df = pd.DataFrame(recommendations)
    return recs_df


def get_hybrid_recommendations(user_id, pred_ratings_matrix_hybrid, user_item_matrix_hybrid, 
                               user_ids_map_hybrid, movie_ids_map_hybrid, 
                               ratings_df_hybrid, movies_df_hybrid, cosine_sim_content_hybrid, 
                               indices_map_hybrid, 
                               top_n=10, collab_weight=0.5, content_weight=0.5, 
                               min_rating_threshold_hybrid=4.0):
    
    # Get more recommendations initially to have a larger pool for merging
    num_initial_recs = top_n * 3 

    content_recs = get_content_recommendations_for_user(
        user_id, ratings_df_hybrid, movies_df_hybrid, cosine_sim_content_hybrid, 
        indices_map_hybrid, top_n=num_initial_recs, min_rating_threshold=min_rating_threshold_hybrid
    )
    collab_recs = get_collaborative_recommendations(
        user_id, pred_ratings_matrix_hybrid, user_item_matrix_hybrid, 
        user_ids_map_hybrid, movie_ids_map_hybrid, movies_df_hybrid, 
        ratings_df_hybrid, top_n=num_initial_recs
    )

    # Handle empty cases
    no_content = content_recs.empty or not ('predicted_content_score' in content_recs.columns and content_recs['predicted_content_score'].notna().any())
    no_collab = collab_recs.empty or not ('predicted_collaborative_score' in collab_recs.columns and collab_recs['predicted_collaborative_score'].notna().any())

    if no_content and no_collab:
        st.info(f"No recommendations could be generated for User {user_id} from either model.")
        return pd.DataFrame(columns=['movieId', 'title_clean', 'hybrid_score', 'genres_str'])
    
    # Normalize scores
    scaler = MinMaxScaler()

    if not no_content:
        if content_recs['predicted_content_score'].nunique() > 1:
            content_recs['normalized_content_score'] = scaler.fit_transform(content_recs[['predicted_content_score']])
        elif len(content_recs['predicted_content_score']) > 0 : # Single unique value
            content_recs['normalized_content_score'] = 0.5 if content_recs['predicted_content_score'].iloc[0] != 0 else 0.0
        else: # Empty after all
            content_recs['normalized_content_score'] = 0.0
            no_content = True # re-evaluate
    else: # Ensure column exists if df was initially empty but now has a row from merge
        if 'normalized_content_score' not in content_recs.columns : content_recs['normalized_content_score'] = 0.0
        
    if not no_collab:
        if collab_recs['predicted_collaborative_score'].nunique() > 1:
            collab_recs['normalized_collab_score'] = scaler.fit_transform(collab_recs[['predicted_collaborative_score']])
        elif len(collab_recs['predicted_collaborative_score']) > 0: # Single unique value
            collab_recs['normalized_collab_score'] = 0.5 if collab_recs['predicted_collaborative_score'].iloc[0] !=0 else 0.0
        else: # Empty after all
            collab_recs['normalized_collab_score'] = 0.0
            no_collab = True # re-evaluate
    else:
        if 'normalized_collab_score' not in collab_recs.columns : collab_recs['normalized_collab_score'] = 0.0


    # After normalization, handle cases where one model has no recs
    if no_content and not no_collab:
        st.info("Content-based model returned no recommendations. Using only collaborative.")
        return collab_recs.head(top_n).rename(columns={'predicted_collaborative_score': 'hybrid_score'})[['movieId', 'title_clean', 'hybrid_score', 'genres_str']]
    if no_collab and not no_content:
        st.info("Collaborative model returned no recommendations. Using only content-based.")
        return content_recs.head(top_n).rename(columns={'predicted_content_score': 'hybrid_score'})[['movieId', 'title_clean', 'hybrid_score', 'genres_str']]
    if no_collab and no_content: # Should have been caught earlier, but as a safeguard
         return pd.DataFrame(columns=['movieId', 'title_clean', 'hybrid_score', 'genres_str'])


    # Merge recommendations
    # Essential columns for content_recs: movieId, title_clean, genres_str, normalized_content_score
    # Essential columns for collab_recs: movieId, title_clean, genres_str, normalized_collab_score (title/genres for filling)
    
    content_recs_to_merge = content_recs[['movieId', 'title_clean', 'genres_str', 'normalized_content_score']].copy()
    collab_recs_to_merge = collab_recs[['movieId', 'title_clean', 'genres_str', 'normalized_collab_score']].copy()

    merged_recs = pd.merge(
        content_recs_to_merge,
        collab_recs_to_merge,
        on='movieId',
        how='outer',
        suffixes=('_content', '_collab')
    )
    
    # Fill NaNs for scores that arose from outer merge
    merged_recs['normalized_content_score'] = merged_recs['normalized_content_score'].fillna(0)
    merged_recs['normalized_collab_score'] = merged_recs['normalized_collab_score'].fillna(0)
    
    # Consolidate title and genres
    merged_recs['title_clean'] = merged_recs['title_clean_content'].fillna(merged_recs['title_clean_collab'])
    merged_recs['genres_str'] = merged_recs['genres_str_content'].fillna(merged_recs['genres_str_collab'])
    
    # Drop redundant columns and any row where title is still NaN (should not happen if movieIds are valid)
    merged_recs.drop(columns=['title_clean_content', 'title_clean_collab', 'genres_str_content', 'genres_str_collab'], inplace=True)
    merged_recs.dropna(subset=['title_clean'], inplace=True)


    merged_recs['hybrid_score'] = (collab_weight * merged_recs['normalized_collab_score'] + 
                                   content_weight * merged_recs['normalized_content_score'])
    
    hybrid_recs_final = merged_recs.sort_values(by='hybrid_score', ascending=False)
    hybrid_recs_final = hybrid_recs_final.drop_duplicates(subset=['movieId'], keep='first') # Keep highest score for a movie
    hybrid_recs_final = hybrid_recs_final.head(top_n)
    
    return hybrid_recs_final[['movieId', 'title_clean', 'hybrid_score', 'genres_str']].reset_index(drop=True)


# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Movie Recommender")
    st.title("🎬 Hybrid Movie Recommendation System")

    # Load data
    movies_df, ratings_df, cosine_sim_content, indices_map = load_data(DATA_DIR)

    if movies_df is None or ratings_df is None or cosine_sim_content is None or indices_map is None:
        st.error("Failed to load necessary data. Application cannot start.")
        return

    if movies_df.empty or ratings_df.empty:
        st.error("Movies or ratings data is empty. Application cannot start.")
        return

    # Train CF model (cached)
    # Pass a copy of ratings_df to avoid issues with Streamlit's caching modifying original df
    predicted_ratings_matrix, user_item_matrix, user_ids_map, movie_ids_map = train_collaborative_filtering_model(ratings_df.copy())
    
    # --- User Input Sidebar ---
    st.sidebar.header("👤 User Preferences")
    available_user_ids = sorted(ratings_df['userId'].unique())
    if not available_user_ids:
        st.sidebar.error("No users found in ratings data.")
        return
        
    selected_user_id = st.sidebar.selectbox("Select User ID:", available_user_ids, index=0 if available_user_ids else -1)

    top_n_recs = st.sidebar.slider("Number of Recommendations:", min_value=5, max_value=30, value=10, step=1)
    
    st.sidebar.subheader("⚙️ Hybrid Model Weights")
    collab_w = st.sidebar.slider("Collaborative Filtering Weight:", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f")
    content_w = 1.0 - collab_w 
    st.sidebar.markdown(f"*Content-Based Weight: `{content_w:.2f}` (auto-adjusted)*")
    
    min_rating_thresh = st.sidebar.slider("Min Rating for Content Profile:", min_value=1.0, max_value=5.0, value=3.5, step=0.5, format="%.1f")

    # --- Main Panel for Recommendations ---
    if selected_user_id:
        st.header(f"✨ Recommendations for User ID: {selected_user_id}")

        if st.sidebar.button("🔄 Get Recommendations", type="primary"):
            with st.spinner(f"Generating recommendations for User {selected_user_id}..."):
                hybrid_recommendations = get_hybrid_recommendations(
                    user_id=selected_user_id,
                    pred_ratings_matrix_hybrid=predicted_ratings_matrix,
                    user_item_matrix_hybrid=user_item_matrix,
                    user_ids_map_hybrid=user_ids_map,
                    movie_ids_map_hybrid=movie_ids_map,
                    ratings_df_hybrid=ratings_df,
                    movies_df_hybrid=movies_df,
                    cosine_sim_content_hybrid=cosine_sim_content,
                    indices_map_hybrid=indices_map,
                    top_n=top_n_recs,
                    collab_weight=collab_w,
                    content_weight=content_w,
                    min_rating_threshold_hybrid=min_rating_thresh
                )

            if not hybrid_recommendations.empty:
                st.subheader("🏆 Top Hybrid Movie Recommendations")
                st.dataframe(hybrid_recommendations[['title_clean', 'hybrid_score', 'genres_str']].style.format({'hybrid_score': "{:.4f}"}))
            else:
                st.info("No hybrid recommendations found for this user with the current settings. Try adjusting the weights or rating threshold.")

            # --- Optional: Display individual model recommendations ---
            with st.expander("🔍 Show Individual Model Recommendations"):
                st.markdown("---")
                st.subheader("🧩 Content-Based Only")
                with st.spinner(f"Generating content-based recommendations for User {selected_user_id}..."):
                    content_recs_only = get_content_recommendations_for_user(
                        selected_user_id, ratings_df, movies_df, cosine_sim_content, indices_map, 
                        top_n=top_n_recs, min_rating_threshold=min_rating_thresh
                    )
                if not content_recs_only.empty:
                    st.dataframe(content_recs_only[['title_clean', 'predicted_content_score', 'genres_str']].style.format({'predicted_content_score': "{:.4f}"}))
                else:
                    st.caption("No content-based recommendations.")

                st.markdown("---")
                st.subheader("🤝 Collaborative Filtering Only")
                with st.spinner(f"Generating collaborative filtering recommendations for User {selected_user_id}..."):
                    collab_recs_only = get_collaborative_recommendations(
                        selected_user_id, predicted_ratings_matrix, user_item_matrix, user_ids_map, movie_ids_map,
                        movies_df, ratings_df, top_n=top_n_recs
                    )
                if not collab_recs_only.empty:
                    st.dataframe(collab_recs_only[['title_clean', 'predicted_collaborative_score', 'genres_str']].style.format({'predicted_collaborative_score': "{:.4f}"}))
                else:
                    st.caption("No collaborative filtering recommendations.")
        else:
            st.info("Click 'Get Recommendations' in the sidebar to view movie suggestions.")
            
    else:
        st.info("Please select a User ID from the sidebar to get recommendations.")
            
    # --- Display some info about the data ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Dataset Info:**")
    if not movies_df.empty: st.sidebar.markdown(f"- Movies: {len(movies_df):,}")
    if not ratings_df.empty: 
        st.sidebar.markdown(f"- Ratings: {len(ratings_df):,}")
        st.sidebar.markdown(f"- Unique Users: {ratings_df['userId'].nunique():,}")

if __name__ == '__main__':
    main()
