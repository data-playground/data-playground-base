# %%

import pandas as pd
from google.cloud import bigquery

# %%

client = bigquery.Client()

# %%

query = """
  SELECT CONCAT("mov",id) id, title, overview,  
  ARRAY_TO_STRING(ARRAY((SELECT name FROM UNNEST(keywords.keywords))), " | ") keywords,
  ARRAY_TO_STRING(ARRAY((SELECT name FROM UNNEST(genres))), "| ") genres,
  FROM `impactful-post-292301.movies.details_movies_filtered` m
  where vote_count > 100

UNION ALL

  SELECT CONCAT("tv",id) id, name title, overview,  
  ARRAY_TO_STRING(ARRAY((SELECT name FROM UNNEST(keywords.results))), " | ") keywords,
  ARRAY_TO_STRING(ARRAY((SELECT name FROM UNNEST(genres))), "| ") genres,
  FROM `impactful-post-292301.movies.details_tv_filtered` m
  where vote_count > 100
"""

df = client.query(query).to_dataframe()

# %%

df['soup'] = df.apply(lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['genres']) + ' ' + x['overview'], axis=1)

df.to_excel(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv_4_recs.xlsx', index=False)
# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the 'soup' column
tfidf_matrix = tfidf.fit_transform(df['soup'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# %%

import numpy as np

# Create a mapping from movie titles to their index
# indices = pd.Series(df.index, index=df['title']).drop_duplicates()
indices = pd.Series(df.index, index=df['id']).drop_duplicates()

def get_recommendations(watched_titles, cosine_sim=cosine_sim, n_recommendations=10):
    """
    Get movie recommendations based on a list of watched movies.

    Args:
        watched_titles (list): A list of movie titles the user has watched.
        cosine_sim (np.ndarray): The cosine similarity matrix.

    Returns:
        pandas.Series: A Series of recommended movie titles.
    """
    # Get the indices of the watched movies
    watched_indices = [indices[title] for title in watched_titles if title in indices]

    if not watched_indices:
        return "Sorry, none of the movies you watched are in our database."

    # Calculate the average similarity scores for the watched movies
    avg_sim_scores = cosine_sim[watched_indices].mean(axis=0)

    # Get the indices and scores of all movies, sorted by similarity
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies (excluding the watched ones)
    recommended_movie_indices = [(i[0], float(i[1])) for i in sim_scores if i[0] not in watched_indices][1:n_recommendations+1]

    return [(df.loc[i[0], 'id'], df.loc[i[0], 'title'], round(i[1],4)) for i in recommended_movie_indices]

# %%

# Example usage:
my_watched_movies = ["tv94951"]
recommendations = get_recommendations(my_watched_movies)
print(recommendations)


# %%






# %%

# Use ngram_range=(1, 2) to include both unigrams and bigrams
tfidf_bigram = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# The rest of the process is the same
tfidf_matrix_bigram = tfidf_bigram.fit_transform(df['soup'])
cosine_sim_bigram = cosine_similarity(tfidf_matrix_bigram, tfidf_matrix_bigram)

# %%

my_watched_movies = ["tv94951"]
recommendations = get_recommendations(my_watched_movies, cosine_sim=cosine_sim_bigram)
print(recommendations)


# %%


# Use ngram_range=(1, 2) to include both unigrams and bigrams
tfidf_trigram = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

# The rest of the process is the same
tfidf_matrix_trigram = tfidf_trigram.fit_transform(df['soup'])
cosine_sim_trigram = cosine_similarity(tfidf_matrix_trigram, tfidf_matrix_trigram)

# %%

my_watched_movies = ["tv94951"]
recommendations = get_recommendations(my_watched_movies, cosine_sim=cosine_sim_trigram, n_recommendations=30)
print(recommendations)


np.save(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\cosine_sim_trigram.npy', cosine_sim_trigram)


# %%






# %%

from sentence_transformers import SentenceTransformer

# %%

# --- 🧠 Vectorization and Similarity (IMPROVED) ---
# Load a pre-trained Sentence-BERT model
# 'all-MiniLM-L6-v2' is a great, fast starting point.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each movie's 'soup'. This may take a moment.
# The .encode() method converts our text into a list of numerical vectors.
embeddings = model.encode(df['soup'].tolist(), show_progress_bar=True)

# Compute the cosine similarity matrix on the new embeddings
cosine_sim_embeddings = cosine_similarity(embeddings, embeddings)


# --- Recommender Function (Same as before, just uses the new similarity matrix) ---
indices_embeddings = pd.Series(df.index, index=df['id']).drop_duplicates()

# %%


def get_recommendations_embeddings(watched_titles, cosine_sim=cosine_sim_embeddings, n_recommendations=10):
    watched_indices = [indices_embeddings[title] for title in watched_titles if title in indices_embeddings]
    if not watched_indices:
        return "Sorry, none of the movies you watched are in our database."
    
    avg_sim_scores = cosine_sim[watched_indices].mean(axis=0)
    sim_scores = sorted(list(enumerate(avg_sim_scores)), key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [(i[0], float(i[1])) for i in sim_scores if i[0] not in watched_indices][1:n_recommendations+1]

    return [(df.loc[i[0], 'id'], df.loc[i[0], 'title'], round(i[1],4)) for i in recommended_movie_indices]

# --- Getting Recommendations (Same as before) ---
my_watched_movies = ["tv94951"]
recommendations = get_recommendations_embeddings(my_watched_movies)
print(recommendations)

np.save(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\cosine_sim_embeddings.npy', cosine_sim_embeddings)

# %%










# %%


import json

# %%

query = """
SELECT 
  id, 
  title,
  original_language,
  poster_path,
  runtime,
  watch_providers
FROM `impactful-post-292301.movies.details_movies_filtered`
  where vote_count > 100
"""

movies_df = client.query(query).to_dataframe()

query = """
SELECT 
  id, 
  name,
  original_language,
  poster_path,
  number_of_seasons,
  number_of_episodes,
  watch_providers
FROM `impactful-post-292301.movies.details_tv_filtered`
  where vote_count > 100
"""

tv_df = client.query(query).to_dataframe()

# %%

records = [
    {
        # Add the movie-level details to each record
        'content_type': 'movie',
        'id': "mov" + str(movie.get('id')),
        'title': movie.get('title'),        
        'original_language': movie.get('original_language'),
        'poster_path': movie.get('poster_path'),
        'info': "runtime: " + str(movie.get('runtime')) + ' minutes',
        
        # Add the provider-level details
        'country': country,
        'watch_type': view_type,
        **provider
    }
    # Loop through each movie in the main list
    for movie in json.loads(movies_df.to_json(orient='records'))
    # Loop through the countries in watch_providers, using .get() for safety
    for country, providers_by_country in movie.get('watch_providers', {}).items()
    # Loop through the view types (flatrate, rent, etc.)
    for view_type, providers in providers_by_country.items()
    # Ensure the provider list is a list and is not empty
    if isinstance(providers, list) and providers
    # Loop through the final list of provider dictionaries
    for provider in providers
]

# Create the DataFrame from the list of records
full_movie_df = pd.DataFrame(records)

# %%

records = [
    {
        # Add the movie-level details to each record
        'content_type': 'tv',
        'id': "tv" + str(tv.get('id')),
        'title': tv.get('name'),        
        'original_language': tv.get('original_language'),
        'poster_path': tv.get('poster_path'),
        'info': "seasons: " + str(tv.get('number_of_seasons')) + '; episodes: ' + str(tv.get('number_of_episodes')),
        
        # Add the provider-level details
        'country': country,
        'watch_type': view_type,
        **provider
    }
    # Loop through each movie in the main list
    for tv in json.loads(tv_df.to_json(orient='records'))
    # Loop through the countries in watch_providers, using .get() for safety
    for country, providers_by_country in tv.get('watch_providers', {}).items()
    # Loop through the view types (flatrate, rent, etc.)
    for view_type, providers in providers_by_country.items()
    # Ensure the provider list is a list and is not empty
    if isinstance(providers, list) and providers
    # Loop through the final list of provider dictionaries
    for provider in providers
]

# Create the DataFrame from the list of records
full_tv_df = pd.DataFrame(records)

# %%

full_df = pd.concat([full_movie_df, full_tv_df], axis = 0).reset_index(drop=True)
full_df.to_excel(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv.xlsx', index=False)
full_df.to_json(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv.json', orient='records')
# %%

content_type = json.loads(full_df['content_type'].drop_duplicates().to_json(orient = 'records'))
original_language = json.loads(full_df['original_language'].drop_duplicates().to_json(orient = 'records'))
providers = json.loads(full_df[['provider_id', 'provider_name', 'logo_path']].drop_duplicates().to_json(orient = 'records'))
watch_type = json.loads(full_df['watch_type'].drop_duplicates().to_json(orient = 'records'))
country = json.loads(full_df['country'].drop_duplicates().to_json(orient = 'records'))

# %%

filters = {
    'id': [i[0] for i in recommendations],
    'provider_id': [
        8, # Netflix
        9, # Amazon Prime Video
        119, # Amazon Prime Video
        350, # Apple TV
        386, # Peacock Premium
        531, # Parmount Plus
        15, # Hulu

        613, # Amazon Prime Video Free with Ads
        538, # Plex
        7, # Fandango at Home
        332, # Fandango at Home Free
        73, # Tubi TV
        235, # YouTube Free
        192, # YouTube
        207, # The Roku Channel
        300, # Pluto TV

    ],
    'watch_type': ['flatrate', 'free', 'ads'],
    'country': ['US']
}

# %%

recs = full_df.query(" & ".join([f"{key}.isin({value})" for key, value in filters.items() if value != []]))

main_content_cols = ['id', 'title', 'original_language', 'poster_path', 'info', 'content_type']

# Define columns that describe the watch options (providers)
watch_option_cols = ['country', 'watch_type', 'display_priority', 'provider_name', 'provider_id', 'logo_path']

# Initialize a list to store the final JSON objects
json_output = []

for content_id, group in recs.groupby('id'):
    # Extract the main content details from the first row of the group
    # (since these fields are identical for all rows within the same content_id group)
    content_details = group[main_content_cols].iloc[0].to_dict()

    # Extract the watch options for all rows in the group
    # Convert each row's watch option columns into a dictionary
    content_details['watch_options'] = group[watch_option_cols].to_dict(orient='records')

    # Add the combined content details to the list
    json_output.append(content_details)

# Convert the list of dictionaries to a JSON string with indentation for readability
json_string = sorted(json_output, key=lambda x: [i[0] for i in recommendations].index(x['id']))

print(json_string)
# %%

recs['sort'] = pd.Categorical(df['id'], categories=[i[0] for i in recommendations], ordered=True)

# %%
