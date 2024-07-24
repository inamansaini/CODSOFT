import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data of user ratings
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 4, 5, 3, 5]
}

# Sample data of movie details
movies_data = {
    'movie_id': [1, 2, 3, 4],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'genre': ['Action', 'Action', 'Drama', 'Drama']
}

# Convert the dictionaries into pandas DataFrames
ratings_df = pd.DataFrame(ratings_data)
movies_df = pd.DataFrame(movies_data)

# Create a user-movie matrix for collaborative filtering
user_movie_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate cosine similarity between movies for collaborative filtering
item_similarity_matrix = cosine_similarity(user_movie_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Create a TF-IDF matrix of genres for content-based filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genre'])

# Calculate cosine similarity between movies based on genres
content_similarity_matrix = cosine_similarity(tfidf_matrix)
content_similarity_df = pd.DataFrame(content_similarity_matrix, index=movies_df['movie_id'], columns=movies_df['movie_id'])

# Function to get movie recommendations using collaborative filtering
def get_collaborative_recommendations(user_id, num_recommendations=2):
    user_ratings = user_movie_matrix.loc[user_id]
    similar_scores = user_ratings.dot(item_similarity_df)
    similar_scores = similar_scores[user_ratings == 0]  # Exclude movies already rated by the user
    top_recommendations = similar_scores.sort_values(ascending=False).index
    return movies_df[movies_df['movie_id'].isin(top_recommendations[:num_recommendations])]

# Function to get movie recommendations using content-based filtering
def get_content_recommendations(user_id, num_recommendations=2):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    similar_scores = pd.Series(dtype=float)
    for _, row in user_ratings.iterrows():
        similar_scores = similar_scores.add(content_similarity_df[row['movie_id']], fill_value=0)
    similar_scores = similar_scores.drop(user_ratings['movie_id'])  # Exclude movies already rated by the user
    top_recommendations = similar_scores.sort_values(ascending=False).index
    return movies_df[movies_df['movie_id'].isin(top_recommendations[:num_recommendations])]

# Example: Get recommendations for user 1
user_id = 1
collaborative_recommendations = get_collaborative_recommendations(user_id)
content_recommendations = get_content_recommendations(user_id)

print("Collaborative Filtering Recommendations for User 1:")
print(collaborative_recommendations[['title', 'genre']])

print("\nContent-Based Recommendations for User 1:")
print(content_recommendations[['title', 'genre']])
