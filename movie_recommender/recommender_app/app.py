# %%

import streamlit as st
from streamlit_javascript import st_javascript
import pandas as pd
import json
import numpy as np
import re
import os

# %%

logger = st.logger.get_logger(__name__)

@st.cache_resource
def build_cosine_similarity(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Use ngram_range=(1, 2) to include both unigrams and bigrams
    tfidf_trigram = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    
    # The rest of the process is the same
    tfidf_matrix_trigram = tfidf_trigram.fit_transform(df['soup'])    
    cosine_sim_trigram = cosine_similarity(tfidf_matrix_trigram, tfidf_matrix_trigram)

    return cosine_sim_trigram

@st.cache_data
def load_and_process_data():
    """
    Loads DataFrames and pre-processes data, caching the results.
    This function will run only once when the app starts or when its inputs change.
    """
    with st.spinner("Loading and processing data... This might take a moment."):
        script_dir = os.path.dirname(__file__)
        
        # Load the main DataFrame (df)
        df_path = os.path.join(script_dir, 'movies_tv_4_recs.json')
        df = pd.read_json(df_path)
        df['id'] = df['id'].astype(str) # Ensure ID is string

        # Load the full DataFrame (full_df) for content lookup
        full_df_path = os.path.join(script_dir, 'movies_tv.json')
        full_df = pd.read_json(full_df_path)
        full_df['id'] = full_df['id'].astype(str) # Ensure ID is string

        # Create mappings for provider names to IDs
        provider_name_to_id_map = full_df.set_index('provider_name')['provider_id'].drop_duplicates().to_dict()
        # Add a check to ensure no None/NaN keys for reverse map if they exist
        provider_id_to_name_map = {v: k for k, v in provider_name_to_id_map.items() if v is not None}

        # Create mappings for content titles to IDs for the watched content lookup
        content_title_to_id_map = full_df.set_index('title')['id'].drop_duplicates().to_dict()
        content_id_to_title_map = {v: k for k, v in content_title_to_id_map.items()}

        indices = pd.Series(df.index, index=df['id']).drop_duplicates()

        # Get unique values for filters
        unique_content_types = full_df['content_type'].unique().tolist()
        unique_languages = sorted(full_df['original_language'].unique().tolist())
        unique_countries = sorted(full_df['country'].unique().tolist())
        unique_watch_types = sorted(full_df['watch_type'].unique().tolist())
        unique_providers = sorted(full_df['provider_id'].dropna().unique().tolist())
        unique_providers_names = sorted(full_df['provider_name'].dropna().unique().tolist())
        all_content_titles = sorted(full_df['title'].unique().tolist())

        return df, full_df, provider_name_to_id_map, provider_id_to_name_map, content_title_to_id_map, content_id_to_title_map, indices, unique_content_types, unique_languages, unique_countries, unique_watch_types, unique_providers, unique_providers_names, all_content_titles


def get_recommendations(watched_ids, cosine_sim, indices_series, df_main, n_recommendations=10):
    """
    Get movie recommendations based on a list of watched movies.

    Args:
        watched_titles (list): A list of movie titles the user has watched.
        cosine_sim (np.ndarray): The cosine similarity matrix.

    Returns:
        pandas.Series: A Series of recommended movie titles.
    """
    # Get the indices of the watched movies
    valid_watched_indices = [indices_series[wid] for wid in watched_ids if wid in indices_series]

    if not valid_watched_indices:
        return "Sorry, none of the movies you watched are in our database."

    # Calculate the average similarity scores for the watched movies
    avg_sim_scores = cosine_sim[valid_watched_indices].mean(axis=0)

    # Get the indices and scores of all movies, sorted by similarity
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar content (excluding the watched ones)
    recommended_content_indices = []
    current_recs_count = 0
    for i, score in sim_scores:
        if current_recs_count >= n_recommendations:
            break
        # Check if the content at index 'i' corresponds to a watched ID
        content_id_at_i = df_main.loc[i, 'id'] # Assuming 'id' is a column in df_main
        if content_id_at_i not in watched_ids: # Exclude watched content
            recommended_content_indices.append((i, score))
            current_recs_count += 1
            
    if not recommended_content_indices:
        return "No new recommendations found based on your watched list after excluding watched content."

    # Get the top 10 most similar movies (excluding the watched ones)
    # recommended_movie_indices = [(i[0], float(i[1])) for i in sim_scores if i[0] not in watched_indices][1:n_recommendations+1]

    return [(df.loc[i[0], 'id'], df.loc[i[0], 'title'], round(i[1],4)) for i in recommended_content_indices]

# %%

screen_width = st_javascript("screen.width")
screen_height = st_javascript("screen.height")
screen_availWidth = st_javascript("screen.availWidth")
screen_availHeight = st_javascript("screen.availHeight")
screen_colorDepth = st_javascript("screen.colorDepth")
screen_pixelDepth = st_javascript("screen.pixelDepth")
screen_isExtended = st_javascript("screen.isExtended")
screen_orientation = st_javascript("screen.orientation.type")

screen = {
    'width': screen_width,
    'height': screen_height,
    'availWidth': screen_availWidth,
    'availHeight': screen_availHeight,
    'colorDepth': screen_colorDepth,
    'pixelDepth': screen_pixelDepth,
    'isExtended': screen_isExtended,
    'orientation': screen_orientation,
}

st.markdown(
    """
    <style>
    div.stElementContainer:has(iframe),
    div.stElementContainer:has(style) {
        display: none !important;
    }

    .st-emotion-cache-13kn1tw{
        width: 25% !important;
        max-width: 250px !important
    }

    .st-emotion-cache-1jw38fe {
        flex: 1 !important;
    }

    @media (max-width: 640px) {
        .st-emotion-cache-13kn1tw {
            min-width: 0 !important;
        }

        .st-emotion-cache-13kn1tw div[data-testid='stImageContainer'],
        .st-emotion-cache-13kn1tw div[data-testid='stImageContainer'] img {
            width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # df = pd.read_excel(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv_4_recs.xlsx')
# df = pd.read_json(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv_4_recs.json')
# # full_df = pd.read_excel(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv.xlsx')
# full_df = pd.read_json(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\movies_tv.json')
# # cosine_sim_trigram = np.load(r'C:\Users\Llubr\Desktop\Github\data-playground-internal\cosine_sim_trigram.npy')

# indices = pd.Series(df.index, index=df['id']).drop_duplicates()
# cosine_sim_trigram = build_cosine_similarity(df)


# --- 2. Data Loading and Caching ---
df, full_df, provider_name_to_id_map, provider_id_to_name_map, content_title_to_id_map, content_id_to_title_map, indices, unique_content_types, unique_languages, unique_countries, unique_watch_types, unique_providers, unique_providers_names, all_content_titles = load_and_process_data()
cosine_sim_trigram = build_cosine_similarity(df) # This will also be cached

# %%

# my_watched_movies = ["tv94951"]
# recommendations = get_recommendations(my_watched_movies, cosine_sim=cosine_sim_trigram, n_recommendations=30)
# print(recommendations)

# %%

# Define columns for main content and watch options
main_content_cols = ['id', 'title', 'original_language', 'poster_path', 'info', 'content_type']
watch_option_cols = ['country', 'watch_type', 'display_priority', 'provider_name', 'provider_id', 'logo_path']

# --- Helper Function for JSON Transformation ---
def transform_dataframe_to_json(dataframe):
    """
    Transforms a pandas DataFrame into a JSON object, grouping by content ID
    and nesting watch options to avoid duplication of main content details.
    """
    json_output = []

    # Ensure relevant columns exist before attempting to access
    current_main_cols = [col for col in main_content_cols if col in dataframe.columns]
    current_watch_cols = [col for col in watch_option_cols if col in dataframe.columns]

    # Watch option columns to include in the innermost list of options
    # We exclude 'country' and 'watch_type' as they will become keys in the nested dictionaries
    current_inner_watch_cols = [
        col for col in watch_option_cols 
        if col in dataframe.columns and col not in ['country', 'watch_type']
    ]

    if dataframe.empty:
        return []
    
    for content_id, content_group in dataframe.groupby('id'):
        content_details = content_group[current_main_cols ].iloc[0].to_dict()

        content_details['watch_options'] = {}

        # Group the watch options for the current content_id by 'country'
        # Ensure 'country' and 'watch_type' are available for grouping
        if 'country' in content_group.columns and 'watch_type' in content_group.columns:
            # Group the content_group further by 'country'
            for country, country_group in content_group.groupby('country'):
                # Initialize the dictionary for watch types within this country
                country_watch_types = {}

                # Group the country_group further by 'watch_type'
                for watch_type, type_group in country_group.groupby('watch_type'):
                    # Convert the type_group (subset of original dataframe) to a list of dictionaries
                    # for the innermost options, using only the relevant columns
                    inner_options = type_group[current_inner_watch_cols].to_dict(orient='records')
                    country_watch_types[watch_type] = inner_options
                
                # Assign the dictionary of watch types to the current country
                content_details['watch_options'][country] = country_watch_types
        else:
            # Fallback if 'country' or 'watch_type' columns are missing.
            # In this case, all watch options are put under a 'default_country' and 'default_type'
            # This handles unexpected data structures gracefully.
            st.warning(f"Missing 'country' or 'watch_type' column for ID {content_id}. Options grouped under default.")
            default_watch_options = content_group[[col for col in current_watch_cols if col != 'country']].to_dict(orient='records')
            content_details['watch_options']['default_country'] = {
                'default_type': default_watch_options
            }

        json_output.append(content_details)

    return json_output

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="Content Recommender")

st.title("Content Recommendation Engine")
st.markdown("Use the filters on the sidebar to narrow down recommendations.")

# --- Sidebar for Filters ---
st.sidebar.header("Filter Recommendations")


# Filter widgets
content_type_selection = st.sidebar.radio(
    "Content Type",
    ['All'] + unique_content_types,
    index=0 # Default to 'All'
)

# --- Improved Original Language Filter ---
with st.sidebar.expander("Original Language(s)", expanded=False): # Collapsed by default
    # Define a list of common languages
    # You can customize this list based on your data or target audience
    common_languages_list = ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh', 'hi', 'pt', 'ar', 'ru']

    # Initialize session state for language selection if not present
    if 'language_selection' not in st.session_state:
        st.session_state.language_selection = unique_languages # Default to all selected

    # Toggle to show all languages or just common ones
    show_all_languages_toggle = st.checkbox("Show All Available Languages", value=True, key="show_all_lang_toggle")

    if show_all_languages_toggle:
        # If showing all, use the full list of unique languages
        options_to_display = unique_languages
    else:
        # If showing common, filter the unique languages to only include common ones
        options_to_display = sorted(list(set(unique_languages) & set(common_languages_list)))
        if not options_to_display: # Fallback if no common languages are in the dataset
            options_to_display = unique_languages

    # Update selection based on options_to_display
    # If a language was selected that is no longer displayed, keep it in session_state,
    # but the multiselect will only show options from options_to_display.
    # To avoid this, we can filter session_state.language_selection based on options_to_display
    initial_selection_for_multiselect = [
        lang for lang in st.session_state.language_selection if lang in options_to_display
    ]

    language_selection = st.multiselect(
        "Select Language(s) (type to search)",
        options_to_display,
        default=initial_selection_for_multiselect,
        key="lang_multiselect" # Use a key for internal Streamlit state management
    )
    # Ensure st.session_state.language_selection reflects the actual selection made
    st.session_state.language_selection = language_selection


country_selection = st.sidebar.multiselect(
    "Country(ies)",
    unique_countries,
    default=unique_countries # Default to all selected
)

watch_type_selection = st.sidebar.multiselect(
    "Watch Type(s)",
    unique_watch_types,
    default=['ads', 'flatrate', 'free'] # Default to all selected
)

# Improved Provider Filter (display name, filter by ID)
selected_provider_names = st.sidebar.multiselect(
    "Provider(s) (type to search)",
    unique_providers_names,
    default=[
        "Netflix", # 8
        "Amazon Prime Video", # 9 & 119
        "Apple TV", # 350
        "Peacock Premium", # 386
        "Paramount Plus", # 531
        "Hulu", # 15

        # 613, # Amazon Prime Video Free with Ads
        "Plex", # 538, # Plex
        # 7, # Fandango at Home
        "Fandango at Home Free", # 332, # Fandango at Home Free
        "Tubi TV", # 73, # Tubi TV
        "YouTube Free", # 235, # YouTube Free
        "YouTube", # 192, # YouTube
        "The Roku Channel", # 207, # The Roku Channel
        "Pluto TV", # 300, # Pluto TV

    ] # Default to all selected
)
selected_provider_ids = [provider_name_to_id_map[name] for name in selected_provider_names if name in provider_name_to_id_map]


# provider_selection = st.sidebar.multiselect(
#     "Provider(s)",
#     unique_providers,
#     default=[
#         "Netflix", # 8
#         "Amazon Prime Video", # 9 & 119
#         "Apple TV", # 350
#         "Peacock Premium", # 386
#         "Paramount Plus", # 531
#         "Hulu", # 15

#         # 613, # Amazon Prime Video Free with Ads
#         # 538, # Plex
#         # 7, # Fandango at Home
#         # 332, # Fandango at Home Free
#         # 73, # Tubi TV
#         # 235, # YouTube Free
#         # 192, # YouTube
#         # 207, # The Roku Channel
#         # 300, # Pluto TV

#     ] # Default to all selected
# )

# --- Main Content Area ---
# st.header("Your Watched Content")
# watched_input = st.text_area(
#     "Enter IDs of content you've already watched (comma-separated):",
#     "mov284470, mov740996" # Example watched content
# )

if 'current_watched_ids' not in st.session_state:
    st.session_state.current_watched_ids = []

if 'watched_selectbox_key' not in st.session_state: # <--- ADD THIS CHECK
    st.session_state.watched_selectbox_key = 0

selected_watched_title = st.selectbox(
    "Search and add content you've watched:",
    options=[''] + all_content_titles,
    index=0,
    key=f"watched_title_select_{st.session_state.watched_selectbox_key}" # Dynamic key
)

if selected_watched_title and selected_watched_title != '':
    selected_id_for_title = content_title_to_id_map.get(selected_watched_title)
    if selected_id_for_title and selected_id_for_title not in st.session_state.current_watched_ids:
        st.session_state.current_watched_ids.append(selected_id_for_title)
        # Increment the key counter *before* calling rerun
        # This tells Streamlit that on the next run, this selectbox is "new"
        st.session_state.watched_selectbox_key += 1
        st.rerun()

if st.session_state.current_watched_ids:
    st.subheader("Currently Watched:")
    for i, wid in enumerate(st.session_state.current_watched_ids):
        title = content_id_to_title_map.get(wid, f"Unknown Content (ID: {wid})")
        col_item, col_btn = st.columns([0.8, 0.2])
        col_item.write(f"- {title} (ID: {wid})")
        if col_btn.button("Remove", key=f"remove_watched_{i}"):
            st.session_state.current_watched_ids.pop(i)
            st.rerun()
else:
    st.info("No watched content added yet.")


# --- Apply Button ---
st.markdown("---")
apply_filters_button = st.button("Apply Filters and Get Recommendations", type="primary")

if apply_filters_button:

    # # Process watched input
    # watched_ids = []
    # if watched_input:
    #     watched_ids = [item.strip() for item in watched_input.split(',') if item.strip()]

    # recommendations = get_recommendations(st.session_state.current_watched_ids, cosine_sim=cosine_sim_trigram, n_recommendations=30)

    recommendations_list_of_tuples = get_recommendations(
        st.session_state.current_watched_ids,
        cosine_sim=cosine_sim_trigram,
        indices_series=indices, # Pass indices
        df_main=df, # Pass the main df used for cosine sim for lookups
        n_recommendations=30
    )

    if isinstance(recommendations_list_of_tuples, str): # Handle error/info messages from get_recommendations
        st.info(recommendations_list_of_tuples)
    else:

        recommended_ids = [item[0] for item in recommendations_list_of_tuples]

        # --- Apply Filters ---
        filtered_df = full_df.query(f"id.isin({recommended_ids})").copy()

        # Apply content type filter
        if content_type_selection != 'All':
            filtered_df = filtered_df[filtered_df['content_type'] == content_type_selection]

        # Apply language filter
        if language_selection: # Only filter if languages are selected
            filtered_df = filtered_df[filtered_df['original_language'].isin(language_selection)]

        # Apply country filter
        if country_selection: # Only filter if countries are selected
            filtered_df = filtered_df[filtered_df['country'].isin(country_selection)]

        # Apply watch type filter
        if watch_type_selection: # Only filter if watch types are selected
            filtered_df = filtered_df[filtered_df['watch_type'].isin(watch_type_selection)]

        # # Apply provider filter
        # if provider_selection: # Only filter if providers are selected
        #     filtered_df = filtered_df[filtered_df['provider_name'].isin(provider_selection)]

        # # Exclude watched content
        # if watched_ids:
        #     filtered_df = filtered_df[~filtered_df['id'].isin(watched_ids)]

        if selected_provider_ids:
            filtered_df = filtered_df[filtered_df['provider_id'].isin(selected_provider_ids)]

        if st.session_state.current_watched_ids:
            filtered_df = filtered_df[~filtered_df['id'].isin(st.session_state.current_watched_ids)]


        # --- Display Results ---
        st.header("Recommended Content")

        if filtered_df.empty:
            st.info("No content found matching your criteria. Try adjusting your filters or watched content.")
        else:
            # Transform the filtered DataFrame into the desired JSON structure
            recommended_json = sorted(transform_dataframe_to_json(filtered_df), key=lambda x: recommended_ids.index(x['id']))

            # Display the JSON structure directly (for debugging/inspection)
            # st.subheader("JSON Output:")
            # st.json(recommended_json)

            st.subheader("Visual Display of Recommendations:")
            # Display recommendations in a more user-friendly card-like format
            for item in recommended_json:
                col1, col2 = st.columns([1, 3]) # Column for poster, column for details

                with col1:
                    # Placeholder for poster_path as we don't have actual image URLs
                    # In a real app, you'd construct a full URL like f"https://image.tmdb.org/t/p/w500{item['poster_path']}"
                    poster_url = f"https://placehold.co/100x150/000000/FFFFFF?text=No+Image" # Generic placeholder
                    if item['poster_path']:
                        # If you had a base URL for your images, you'd use it here.
                        # For this example, we'll just show the path or a generic placeholder.
                        # If the path is relative, it won't display an image directly in Streamlit without a base URL.
                        st.image("https://media.themoviedb.org/t/p/w600_and_h900_bestv2/" + item['poster_path'], caption=item['title'], width=250)
                    else:
                        st.image(poster_url, caption="No Poster", width=250)


                with col2:
                    logo_url = f"https://placehold.co/50x50/000000/FFFFFF?text=Logo" # Generic placeholder

                    st.markdown(f"### [{item['title']}](https://www.themoviedb.org/{item['content_type']}/{re.sub('[a-zA-Z]+', '', item['id'])}) ({item['content_type'].capitalize()})")
                    st.write(f"**Language:** {item['original_language'].upper()}")
                    st.write(f"**Info:** {item['info']}")

                    overview = df.query(f"id == '{item['id']}'")['overview'].values[0]
                    st.write(f"**Overview:** {overview}")
                    

                    st.markdown("#### Available On:")
                    for country, watch_types in item['watch_options'].items():
                        st.markdown(f"##### {country}")

                        if screen['width'] > 1250:
                            num_cols_type = 3
                        elif 1100 < screen['width'] <= 1250:
                            num_cols_type = 2
                        elif 640 < screen['width'] <= 1100:                        
                            num_cols_type = 1
                        elif screen['width'] <= 640:
                            num_cols_type = 1

                        cols_type = st.columns(num_cols_type) 

                        # Keep track of the current column index
                        col_idx_type = 0 

                        for watch_type, options in watch_types.items():
                            # Map watch_type to st.badge variants
                            # Choose the variant that best represents the watch type's 'status'
                            badge_variant = 'neutral' # Default variant
                            if watch_type == 'flatrate':
                                badge_variant = 'success' # Often implies included in subscription
                            elif watch_type == 'rent' or watch_type == 'buy':
                                badge_variant = 'primary' # Action-oriented, often paid
                            elif watch_type == 'ads' or watch_type == 'free':
                                badge_variant = 'info' # Informational, often free with ads
                            
                            with cols_type[col_idx_type]:
                                # You can add more mappings here
                                st.badge(label=watch_type, color="primary")

                                num_cols_options = 4
                                cols_options = st.columns(num_cols_options) 

                                # Keep track of the current column index
                                col_idx_options = 0 
                                
                                # Placeholder for logo_path
                                for option in options:
                                    with cols_options[col_idx_options]:

                                        if option['logo_path']:
                                            # Similar to poster, this would need a base URL for actual display
                                            logo_url = "https://image.tmdb.org/t/p/original/" + option['logo_path']

                                        st.image(logo_url, use_container_width=True) # Keep the placeholder for now
                                        st.markdown(f"<p style='font-size:10px; text-align:center; margin-top:-10px'>{option.get('provider_name', 'N/A')}</p>", unsafe_allow_html=True)

                                    # Move to the next column, wrapping around if needed
                                    col_idx_options = (col_idx_options + 1) % num_cols_options

                            col_idx_type = (col_idx_type + 1) % num_cols_type

                    st.markdown("---") # Separator for readability


# %%
