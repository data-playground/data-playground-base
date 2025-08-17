# %%

#########################################################
# FIFA World Cup data
#########################################################

import requests
from datetime import datetime, date
from tqdm import tqdm
from google.cloud import bigquery
import json

# %%

class FIFA:
    def __init__(self, competition_id = None, from_date_init = None, to_date_init = None):
        
        # Create BigQuery client
        self.client = bigquery.Client()

        # List of matches
        self.games_lst = []

        # List to hold the details for each match
        self.games_detail = []

        # List to hold all events for each match
        self.games_events = []

        # Basic headers to be used on the requests. Changes will be done dependending on the request
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
            "Connection": "keep-alive",
            "Host": "api.fifa.com",
            "Origin": "https://www.fifa.com",
            "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "Windows",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        }

        # Extracting competitions available in the FIFA database
        self.competitions = self.get_competitions()

        print("""
The processes to be done next are dependent on providing a competition ID. The __init__ process gathered all the competitions available in the FIFA database. Approach carefully as there are more than 200 competitions available.
              
Below, I made available a couple of interesting competitions that will be used in this process or that the data from those will be used in future projects.

    * Competition ID -> Competition Name     
    * 17 -> FIFA World Cup
    * 10005 -> FIFA Club World Cup
    * 2000001032 -> Champions League
    * 2000000000 -> Barclays Premier League
    * 2000000078 -> Campeonato Brasileiro Série A
    * 2000001041 -> Europa League
    * 2000000018 -> Ligue 1
    * 2000000019 -> Bundesliga
    * 2000000026 -> Serie A
    * 2000001035 -> Copa Libertadores
    * 2000000037 -> La Liga      
    * 2000000103 -> MLS
    * 504 -> UEFA UEFA Nations League
    * 512 -> Olympic Football Tournament
    * 8tddm56zbasf57jkkay4kbf11 -> UEFA Men's European Championship
        """)

        if not competition_id:
            self.competition_id = input("Enter the competition ID [default is 17 (for FIFA World Cup)]") or 17
        else:
            self.competition_id = competition_id

        if not from_date_init:
            from_date_init = input("Enter the start date to extract matches in the following format YYYY-mm-dd [default is 2022-01-01]") or '2022-01-01'

        self.from_date = from_date_init + "T00:00:00Z"

        if not to_date_init:
            to_date_init = input("Enter the end date to extract matches in the following format YYYY-mm-dd [default is 2022-12-31]") or '2022-12-31'

        self.to_date = to_date_init + "T23:59:59Z"

    def get_games(self, competition_id, from_date, to_date, continuation_token = None, continuation_hash = None):
        """
            Get all the matches for a defined competition, between two dates.
            Continuation tokens and hashes are present if there are more than 500 entries (matches) in the database for the defined settings. 
                If there are more entries available after the ones displayed, a value will be available for both token and hash
                If there are less than 500 entries or the end of the entries list was reached, both token and hash will be NULL
        """
        
        headers = self.headers.copy()
        headers["X-Mdp-Continuation-Token"] = continuation_token

        params = {
            "from": from_date, # minimum accepted value is "1930-01-01T00:00:00Z"
            "to": to_date,
            "language": "en",
            "count": 500, # 500 is the maximum number of records retrieved
            "idCompetition": competition_id,
            "continuationhash": continuation_hash
        }

        url = "https://api.fifa.com/api/v3/calendar/matches"

        r = requests.get(url, params = params, headers = headers)

        r_json = r.json()

        return r_json['Results'], r_json['ContinuationToken'], r_json['ContinuationHash']


    def get_competitions(self):
        """
            Pulls all the competitions available in the FIFA database.
            While this step is not a requirement to get the expected output of this process, it provides guidance over which competitions can have data extracted.
        """

        url = f"https://cxm-api.fifa.com/fifaplusweb/api/sections/matches/competitionslist/3jlHZVPUI0eeeBFTX9qSZ1?locale=en"

        headers = self.headers.copy()
        headers['Host'] = "cxm-api.fifa.com"

        r = requests.get(url, headers = headers)
        output_dict = r.json()['competitions']
        output_dict = [{k: v for k, v in d.items() if k in ('competitionId', 'name')} for d in output_dict]

        return output_dict

    def get_match_details(self, game, req_type):
        """
            Extract details for an individual match.
            Two types of data can be extracted:
                * details: gathers general information about the match, like players, officials, location, stadium...
                * playbyplay: extracts events that appear on a play-bu-play description of the match (such as goals, cards, penalties...). Older games do not contain as many events, as only goals were tracked (and play-by-play was not a thing)
        """

        if req_type == 'details':
            url = f"https://api.fifa.com/api/v3/live/football/{game['IdCompetition']}/{game['IdSeason']}/{game['IdStage']}/{game['IdMatch']}?language=en"
            r = requests.get(url, headers = self.headers)
            output_dict = r.json()
        elif req_type == 'playbyplay':
            url = f"https://api.fifa.com/api/v3/timelines/{game['IdCompetition']}/{game['IdSeason']}/{game['IdStage']}/{game['IdMatch']}?language=en"
            r = requests.get(url, headers = self.headers)
            output_dict = r.json()['Event']
            output_dict = [{**d, 'IdMatch': game['IdMatch']} for d in output_dict]
        else:
            raise ValueError("The only req_type values allowed are 'details' and 'playbyplay'")

        return output_dict
    
    def run_proc(self):
        """
            Gather each game as well as details (such as the players, score, goalscorers, stadium, officials...) and events (important event in the game that would appear on a play-by-play description - older games will have less events included)
            In this example, games for all the "FIFA World Cup" seasons will be displayed
        """

        # Get first set of games (500 max), alongside the hash and token that provide continuation (or next set of values)
        games_res, continuation_token, continuation_hash = self.get_games(competition_id = self.competition_id, from_date = self.from_date, to_date = self.to_date)
        self.games_lst.extend(games_res)

        # Continue to get game data while a continuation hash exist, which means a next page of results still exists
        while continuation_hash:
            games_res, continuation_token, continuation_hash = self.get_games(competition_id = self.competition_id, from_date = self.from_date, to_date = self.to_date, continuation_token = continuation_token, continuation_hash = continuation_hash)
            self.games_lst.extend(games_res)

        # Get details for every game in the list of games
        for game in tqdm(self.games_lst):
            self.games_detail.append(self.get_match_details(game, 'details'))
            self.games_events.extend(self.get_match_details(game, 'playbyplay'))
            
    def func_load_data(self, data, project_id, table, schema = [], write_disposition = 'append'):
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
        project = self.client.project
        dataset_id = bigquery.DatasetReference(project, project_id)
        table_id = dataset_id.table(table)

        # Run the load job
        job = self.client.load_table_from_json(data, table_id, job_config=job_config)  # Make an API request.
        job.result()  # Wait for the job to complete.

        print(f'Loaded table {table}')

    def get_starter_dict(self, attr_name):
        """
            Create a dictionary containing all the possible keys from the list of dictionaries selected.
            This might be a necessary step if the BigQuery table needs to be created, to ensure the schema is correct and complete
        """
        
        starter_dict = [{}]
        for event in getattr(self, attr_name):
            for key, value in event.items():
                if value:
                    starter_dict[0][key] = value
        
        return starter_dict
    
    def fix_dictionary(self, attr_name):
        """
            Fills the dictionary with all the keys that appear in the list of dictionaries
            This step is impotant to ensure every dictionary has the same keys
        """
        
        template_dict = self.get_starter_dict(attr_name)[0]
        for dictionary in getattr(self, attr_name):
            # Iterate through the keys in the template dictionary
            for key in template_dict:
                # If the key is not in the current dictionary, add it with the default value (None)
                if key not in dictionary:
                    dictionary[key] = None  # or simply dictionary[key] = None if all default values are None

        return dictionary

# %%

# Instantiate FIFA class
fifawc = FIFA(competition_id=17, from_date_init="1930-01-01", to_date_init="2022-12-31")

# %%

# Get all information on the FIFA intance (specially interesting to get to know all the competitions available)
fifawc.__dict__

# %%

# Confirm list of games is empty
fifawc.games_lst

# %%

# Run process to gather all games and their details
fifawc.run_proc()

# %%

# Confirm list of games is now built
fifawc.games_lst

# %%

# Load games data to BigQuery
fifawc.func_load_data(
    fifawc.games_lst, 
    'fifawc',
    'matches',
)

# Load game detail data to BigQuery
fifawc.func_load_data(
    fifawc.games_detail, 
    'fifawc',
    'match_details',
)


# %%

#######################################################
# If the creation of the table is necessary, some extra steps might be necessary to ensure the schema is correct
#######################################################

# games_events_starter = fifawc.get_starter_dict('games_events')

# fifawc.func_load_data(
#     games_events_starter, 
#     'fifawc',
#     'match_events',
# )

# # Construct the TRUNCATE TABLE statement
# query = f"TRUNCATE TABLE `{fifawc.client.project}.fifawc.match_events`"

# # Execute the query
# query_job = fifawc.client.query(query)

# # Wait for the job to complete
# query_job.result()

# %%

# Define table the game events will be loaded
table_ref = fifawc.client.dataset('fifawc').table('match_events')
table = fifawc.client.get_table(table_ref)

# Get the current schema for the existing table
schema = table.schema

# fifawc.fix_dictionary('games_events')

# Load game evtnts data to BigQuery
fifawc.func_load_data(
    json.loads(json.dumps(fifawc.games_events)), 
    'fifawc',
    'match_events',
    # schema = schema
)