# %%

import requests
from tqdm import tqdm
from datetime import date
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery
import pandas as pd
import copy

# %%

client = bigquery.Client()

def func_load_data(data, project_id, table, schema = [], write_disposition = 'append'):
    '''
        Load data into BigQuery tables
        Process is preset to append data, taking dataframe into consideration
    '''

    # Setup BigQuery load configurations
    job_config = bigquery.LoadJobConfig()

    # Set write disposition to append or overwrite data to BigQuery table
    if write_disposition == 'append':
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_APPEND
    elif write_disposition == 'overwrite':
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

    # The source format defaults to CSV, so the line below is optional.
    # job_config.source_format = bigquery.SourceFormat.CSV

    if schema == []:
        job_config.autodetect = True
    else:
        job_config.schema = schema

    # Set project, database and table IDs for the load 
    project = client.project
    dataset_id = bigquery.DatasetReference(project, project_id)
    table_id = dataset_id.table(table)

    try:
        # Run the load job
        job = client.load_table_from_json(data, table_id, job_config=job_config)  # Make an API request.
        job.result()  # Wait for the job to complete.

        print(f'Loaded table {table}')
    except:
        print(job.errors)

# %%

def discover(req_type="movie", page=1, start_date="2025-01-01", end_date="2025-01-01"):
    # Discover Movies

    if req_type == "movie":
        url = f"https://api.themoviedb.org/3/discover/{req_type}"

        params = {
            "include_adult": False,
            "include_video": False,
            "language": "en-US",
            "page": page,
            "sort_by": "popularity.desc",
            "watch_region": "US",
            "primary_release_date.gte": start_date,
            "primary_release_date.lte": end_date
        }
    elif req_type == "tv":
        url = f"https://api.themoviedb.org/3/discover/{req_type}"
        
        params = {
            "include_adult": False,
            "include_video": False,
            "language": "en-US",
            "page": page,
            "sort_by": "popularity.desc",
            "watch_region": "US",
            "first_air_date.gte": start_date,
            "first_air_date.lte": end_date
        }
    else:
        raise Exception("req_type is required to be either 'movie' or 'tv'")

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers, params=params)

    return response.json()

# %%

sd = date(2000, 1, 1)
ed = date.today()

date_ranges = []

while sd <= ed:
    sd1 = sd.strftime("%Y-%m-%d")
    sd += relativedelta(months=1)
    date_ranges.append((sd1, (sd-relativedelta(days=1)).strftime("%Y-%m-%d")))

# %%

discover_movies = []

for start_date, end_date in date_ranges[:-1]:
    print(start_date, end_date)

    r_json = discover(req_type="movie", page=1, start_date=start_date, end_date=end_date)
    total_pages = r_json['total_pages']

    discover_movies.extend(r_json['results'])

    for page in tqdm(range(2, total_pages+1)):
        r_json = discover(req_type="movie", page=page, start_date=start_date, end_date=end_date)
        discover_movies.extend(r_json['results'])

func_load_data(discover_movies, 'movies', 'discover_movies', schema = [], write_disposition = 'append')

# %%

discover_tv = []

for start_date, end_date in date_ranges[:-1]:
    print(start_date, end_date)

    r_json = discover(req_type="tv", page=1, start_date=start_date, end_date=end_date)
    total_pages = r_json['total_pages']

    discover_tv.extend(r_json['results'])

    for page in tqdm(range(2, total_pages+1)):
        r_json = discover(req_type="tv", page=page, start_date=start_date, end_date=end_date)
        discover_tv.extend(r_json['results'])

func_load_data(discover_tv, 'movies', 'discover_tv', schema = [], write_disposition = 'append')

# %%


# %%






# %%

def genres(req_type="movie"):
    # Movie Genre IDs
    # To connect to movie details
    if req_type in ("movie", "tv"):
        url = f"https://api.themoviedb.org/3/genre/{req_type}/list?language=en"
    else:
        raise Exception("req_type is required to be either 'movie' or 'tv'")

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers)

    return response.json()

# %%

genres_movie = genres("movie")
genres_tv = genres("tv")

genres_all = genres_movie['genres']
genres_all.extend(genres_tv['genres'])

genres_all = sorted([dict(t) for t in {tuple(d.items()) for d in genres_all}], key=lambda x: x['id'])

# %%

func_load_data(genres_all, 'movies', 'genres', schema = [], write_disposition = 'append')

# %%






# %%

def movie_details(id, req_type="movie"):
# Movie Details
# Only get Watch Providers in the US
# Use Similar and Recommendation as matter of comparison
    if req_type in ("movie", "tv"):
        url = f"https://api.themoviedb.org/3/{req_type}/{id}?append_to_response=keywords%2Cwatch%2Fproviders%2Csimilar%2Crecommendations%2Cexternal_ids&language=en-US"
    else:
        raise Exception("req_type is required to be either 'movie' or 'tv'")

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers)

    # print(response.text)

    r_json = response.json()
    # r_json["watch/providers"] = r_json["watch/providers"]["results"]["US"]
    r_json["similar"] = [i["id"] for i in r_json["similar"]["results"]]
    r_json["recommendations"] = [i["id"] for i in r_json["recommendations"]["results"]]

    return r_json

def fix_key_name(lst, old_key, new_key):
    for item_dict in lst:
        if old_key in item_dict:  # Check if the old key exists in the current dictionary
            item_dict[new_key] = item_dict.pop(old_key)
    
    return lst

def merge_dictionaries(dict_list):
    """
    Recursively merges a list of dictionaries to create the most complete version.

    This function combines multiple dictionaries, ensuring that all keys from all
    dictionaries are present in the final result. When a key exists in multiple
    dictionaries:
    - If the values are dictionaries, it merges them recursively.
    - If one value is a dictionary and others are not (e.g., None), it prioritizes
      the dictionary.
    - If all values are non-dictionaries, it keeps the first encountered value.

    Args:
        dict_list (list): A list of dictionaries to merge.

    Returns:
        dict: A single, comprehensively merged dictionary.
    """
    # If the list is empty or contains no dictionaries, return an empty dict.
    if not any(isinstance(d, dict) for d in dict_list):
        return {}

    # Initialize the result dictionary.
    result = {}
    
    # Collect all unique keys from all dictionaries in the list.
    all_keys = set(key for d in dict_list if isinstance(d, dict) for key in d)

    for key in sorted(all_keys):
        # Gather all values for the current key from all dictionaries.
        values = [d.get(key) for d in dict_list if isinstance(d, dict) and key in d]

        # Separate the values into two lists: one for dictionaries and one for others.
        dict_values = [v for v in values if isinstance(v, dict)]
        
        if dict_values:
            # If there are dictionary values for this key, merge them recursively.
            result[key] = merge_dictionaries(dict_values)
        else:
            # Otherwise, use the first value found for this key.
            # In the provided example, all non-dict values are None, so this works.
            result[key] = values[0] if values else None
            
    return result

def fix_watch_providers(lst, template_dict):
    for d in lst:
        try:
            d["watch_providers"] = {**template_dict, **d["watch_providers"]["results"]}
        except:
            continue

    return lst

def filter_us_br(lst):
    lst['watch_providers'] = {key: lst['watch_providers'][key] for key in ['BR', 'US']}
    return lst

# %%

query = f'''
    SELECT id 
    FROM `impactful-post-292301.movies.discover_movies` 
    where vote_count >= 20
'''

job = client.query(query)

if not job.errors:
    movie_ids = [row.id for row in job.result()]
else:
    if len([i for i in job.errors if i['reason'] == 'notFound']) > 0:
        movie_ids = []
        print('Table does not exist')

movie_detail = []

for id in tqdm(movie_ids):
    movie_detail.append(movie_details(id, req_type="movie"))

movie_detail_fixed = copy.deepcopy(movie_detail)

# with open('movies.json', 'r') as file:
#     # Load the JSON data into a Python dictionary
#     movie_detail_fixed = json.load(file)

movie_detail_fixed = fix_key_name(movie_detail_fixed, "watch/providers", "watch_providers")

template_dict = merge_dictionaries([{key: {vk: None if vk == 'link' else [{'logo_path': None, 'provider_id': None, 'provider_name': None, 'display_priority': None}] for vk in value.keys()} if value is not [] else None for key, value in mov["watch_providers"]['results'].items()} for mov in movie_detail_fixed])

movie_detail_fixed = fix_watch_providers(movie_detail_fixed, template_dict)

# func_load_data(movie_detail_fixed, 'movies', 'details_movies', schema = [], write_disposition = 'append')

movie_detail_filtered = [filter_us_br(lst) for lst in movie_detail_fixed]

func_load_data(movie_detail_filtered, 'movies', 'details_movies_filtered', schema = [], write_disposition = 'append')

# %%

query = f'''
    SELECT id 
    FROM `impactful-post-292301.movies.discover_tv` 
    where vote_count >= 20
'''

job = client.query(query)

if not job.errors:
    tv_ids = [row.id for row in job.result()]
else:
    if len([i for i in job.errors if i['reason'] == 'notFound']) > 0:
        tv_ids = []
        print('Table does not exist')

tv_detail = []

for id in tqdm(tv_ids):
    tv_detail.append(movie_details(id, req_type="tv"))

tv_detail_fixed = copy.deepcopy(tv_detail)

tv_detail_fixed = fix_key_name(tv_detail_fixed, "watch/providers", "watch_providers")

template_dict = merge_dictionaries([{key: {vk: None if vk == 'link' else [{'logo_path': None, 'provider_id': None, 'provider_name': None, 'display_priority': None}] for vk in value.keys()} if value is not [] else None for key, value in mov["watch_providers"]['results'].items()} for mov in tv_detail_fixed])

tv_detail_fixed = fix_watch_providers(tv_detail_fixed, template_dict)

# func_load_data(tv_detail_fixed, 'movies', 'details_movies', schema = [], write_disposition = 'append')

tv_detail_filtered = [filter_us_br(lst) for lst in tv_detail_fixed]

func_load_data(tv_detail_filtered, 'movies', 'details_tv_filtered', schema = [], write_disposition = 'append')




# %%




# %%

def movie_credits(id):
    # Movie Credits
    # To get actors and directors for recommendations
    url = f"https://api.themoviedb.org/3/movie/{id}/credits?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers)

    return response.json()

# %%

movie_credit = []

for id in tqdm(movie_ids):
    movie_credit.append(movie_credits(id))

func_load_data(movie_credit, 'movies', 'credits_movies', schema = [], write_disposition = 'append')


# %%







# %%

def tv_credits(id):
# Aggregated TV Credits
# Better than just credits, since credits only show last season
    url = f"https://api.themoviedb.org/3/tv/{id}/aggregate_credits?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers)

    return response.json()

# %%

tv_credit = []

for id in tqdm(tv_ids):
    tv_credit.append(tv_credits(id))

func_load_data(tv_credit, 'movies', 'credits_tv', schema = [], write_disposition = 'append')


# %%






# %%

def trending(req_type = "movie", timeframe = "day"):
# Trending Movies
# Can change from day to week
    if req_type in ["movie", "tv", "people", "all"]:
        pass
    else:
        raise Exception("Please enter req_type as one of the following: 'movie', 'tv', 'people', 'all'")

    if req_type in ["day", "week"]:
        pass
    else:
        raise Exception("Please enter timeframe as one of the following: 'day', 'week'")

    url = "https://api.themoviedb.org/3/trending/{req_type}/{timeframe}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjVjNWU2NmIwMzlhZjY0ZTBiZTJjMzgwMzI5YWM0OSIsIm5iZiI6MTc1MjA4MjkwNy44MTcsInN1YiI6IjY4NmVhOWRiZjNkODA1MDU2MDUxZWU2OCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.D8SNbaXaMan_ZiThANN9FJcmb2UWIfDNLiEYuVaovjY"
    }

    response = requests.get(url, headers=headers)

    print(response.text)
