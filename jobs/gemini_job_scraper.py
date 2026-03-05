# %%

import base64
import datetime
import json

import requests
from google import genai
from google.cloud import bigquery, secretmanager
from google.genai import types

# %%

def init():
    """Similar to __init__ in a class, this function defines the data that will be used in other functions.
    The reason this process is not a class is to make is easily usable in Airflow
    """

    # Intantiate the dictionary
    start_dict = {}

    # Get API key for Gemini
    start_dict['gemini_key'] = get_key("Gemini-API")
    # Get API key for GitHub
    start_dict['github_key'] = get_key("Github-Key")

    # Define GitHub data to be used when updating repositories
    start_dict['SUBMODULE_OWNER'] = 'data-playground'
    start_dict['SUBMODULE_REPO'] = 'data-playground-data'
    start_dict['PARENT_OWNER'] = 'data-playground'
    start_dict['PARENT_REPO'] = 'data-playground.github.io'
    start_dict['BRANCH'] = 'main'
    start_dict['SUBMODULE_PATH'] = '_data/data_playground_data' 
    
    # Headers to be used to get RAW contents from a file
    start_dict['HEADERS_RAW']={
        "Accept": "application/vnd.github.v3.raw", 
        "Authorization": f"Bearer {start_dict['github_key']}", 
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Headers to be used to get metadata from a file
    start_dict['HEADERS_META']={
        "Accept": "application/vnd.github+json", 
        "Authorization": f"Bearer {start_dict['github_key']}", 
        "X-GitHub-Api-Version": "2022-11-28"
    }
    return start_dict

def get_key(SECRET_NAME):
    """
        Get API Key from Google Secret Manager
    """
    # Initialize the Secret Manager client
    SMclient = secretmanager.SecretManagerServiceClient()

    # Set the project ID 
    project_id = "impactful-post-292301"

    # Build the request to access the secret version
    request = {"name": f"projects/{project_id}/secrets/{SECRET_NAME}/versions/latest"}

    # Access the secret version
    response = SMclient.access_secret_version(request)

    # Get the secret value
    SECRET_VALUE = response.payload.data.decode("UTF-8")

    return SECRET_VALUE

def func_load_data(data, table, schema = None, write_disposition = 'append'):
    '''
        Load data into BigQuery tables
        Process is preset to append data, taking dataframe into consideration
    '''

    # Create BigQuery client
    client = bigquery.Client()

    # Setup BigQuery load configurations
    job_config = bigquery.LoadJobConfig()

    # Set write disposition to append or overwrite data to BigQuery table
    if write_disposition == 'append':
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_APPEND
    elif write_disposition == 'overwrite':
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

    # The source format defaults to CSV, so the line below is optional.
    # job_config.source_format = bigquery.SourceFormat.CSV

    if schema:
        job_config.schema = schema

    # Set project, database and table IDs for the load 
    project = client.project
    dataset_id = bigquery.DatasetReference(project, 'jobs')
    table_id = dataset_id.table(table)

    # Run the load job
    job = client.load_table_from_json(data, table_id, job_config=job_config)  # Make an API request.
    job.result()  # Wait for the job to complete.

    print(f'Loaded table {table}')

def genai_response(client, target_url=None):
    """
        Run Gemini AI Job Scraper
    """

    # 2. Upload your files
    resume_summary = client.files.upload(file=r"C:\Users\Llubr\Downloads\Pedro Mano Resume.pdf")
    resume_full = client.files.upload(file=r"C:\Users\Llubr\Downloads\Full Resume.pdf")

    # 4. Send the prompt with the files
    if target_url:
        prompt = """
        ### ROLE
        You are an expert Career Agent and Recruitment Specialist. Your goal is to find high-fit job opportunities for the candidate based on the provided resumes.

        ### CONTEXT
        You have been provided with two files: "Pedro Mano's Resume.pdf" (Summary) and "Full Resume.pdf" (Technical Deep-Dive). 
        Key Strengths to prioritize: 
        - GCP Ecosystem (BigQuery, Vertex AI, Gemini 1.5 Pro).
        - Python Automation (Selenium, Django, Airflow).
        - Business Impact ($1M cost savings, A/B testing optimization).
        - Background: MBA in Business Analytics + Industrial Engineering.

        ### EMPLOYMENT FILTER (MANDATORY)
        - ONLY include positions explicitly labeled as "Full-Time".
        - IMMEDIATELY DISCARD any roles labeled as "Contract", "C2C", "Temporary", "Part-time", or "Internship".
        - If a job description does not clearly state the employment type, look for clues like "Benefits-eligible" or "401k matching" to verify it is a permanent full-time role.
        
        ### TASK
        1. ANALYZE THIS SEED URL: """ + target_url + """
        - If you cannot access the contents of the URL, return an error message.
        - First, evaluate the candidate's fit for this specific role.
        2. FIND SIMILAR ROLES: 
        - Use 'google_search' to find 5 additional, open job listings that closely mirror the seniority and technical requirements of the Seed URL.
        3. For each job (including the Seed), verify it matches the technical stack (Python/GCP) and seniority.
        4. Conduct a "Qualification Analysis" for every result against the provided resumes.
        5. Extract compensation/salary range if listed.
            ### QUALIFICATION CRITERIA
            Evaluate the candidate's fit based on:
            - Technical Match: Python/GCP/Airflow/Vertex AI.
            - Impact Match: Proven $1M annual cost savings via automation.
            - Specialized Fit: Experience with 'Agent Development Kits' and A/B Testing.

        ### OUTPUT FORMAT
        Provide the results strictly in the following JSON-style list format. DO NOT INCLUDE ANY EXPLANATORY TEXT OUTSIDE THE JSON. Ensure all fields are filled for each job listing.:

        [{
            "job_title": "Position Title",
            "company": "Company Name",
            "location": "The city/state or 'Remote'",
            "remote": "true if the job is fully remote or hybrid; false if strictly on-site"
            "link": "Direct link to job posting",
            "explanation": "Briefly explain why this job fits the candidate's specific achievements (e.g., matching the $1M savings or Vertex AI experience).",
            "fit_score": "Score out of 100",
            "qualification_analysis": "A detailed explanation of why the candidate is qualified (e.g., 'Matches their need for BigQuery cost auditing').",
            "skill_gaps": "List any specific tools from the JD not clearly highlighted in the resume (e.g., 'dbt', 'Snowflake')."
            "compensation": "Compensation range if listed (e.g., '$160,000 - $210,000'); NULL if not listed."
        }]

        ### SEARCH SCOPE
        Search broadly across the USA, prioritizing high-growth tech hubs and remote-first companies.

        ### CONSTRAINTS
        - Ensure all links are active and lead to the specific job page.
        - Do not include expired roles.
        - Focus on companies in Tech, Retail, or Finance where data optimization is a priority.

        ### URL CONSTRAINTS
        - DO NOT provide links to search result pages (e.g., URLs containing '/jobs/search', '/jobs/results', or '?q=').
        - PRIORITIZE "Deep Links" that lead directly to a specific Job Description (JD).
        - PREFER links from the company's own 'careers' portal (e.g., Greenhouse, Lever, Workday, or the company domain).
        - If a direct JD link cannot be found for a listing, skip that listing and find another.
        """
    else:
        prompt = """
        ### ROLE
        You are an expert Career Agent and Recruitment Specialist. Your goal is to find high-fit job opportunities for the candidate based on the provided resumes.

        ### CONTEXT
        You have been provided with two files: "Pedro Mano's Resume.pdf" (Summary) and "Full Resume.pdf" (Technical Deep-Dive). 
        Key Strengths to prioritize: 
        - GCP Ecosystem (BigQuery, Vertex AI, Gemini 1.5 Pro).
        - Python Automation (Selenium, Django, Airflow).
        - Business Impact ($1M cost savings, A/B testing optimization).
        - Background: MBA in Business Analytics + Industrial Engineering.

        ### EMPLOYMENT FILTER (MANDATORY)
        - ONLY include positions explicitly labeled as "Full-Time".
        - IMMEDIATELY DISCARD any roles labeled as "Contract", "C2C", "Temporary", "Part-time", or "Internship".
        - If a job description does not clearly state the employment type, look for clues like "Benefits-eligible" or "401k matching" to verify it is a permanent full-time role.
        
        ### TASK
        1. Use the 'google_search' tool to find 10 current, open job listings across the United States.
        2. Search for roles: 'Senior Analytics Engineer', 'AI Solutions Architect', 'Senior BI Engineer', 'Data Engineer' or 'Solutions Engineer (GCP)'.
        3. For each job, verify it matches the technical stack (Python/GCP) and the seniority level.
        4. For each job, conduct a "Qualification Analysis" against the provided resumes.
        5. For each job, extract the compensation/salary range if listed. Be precise (e.g., "$160,000 - $210,000"). If not listed, leave NULL (empty value).
        6. Finally, confirm the job post is live by visiting the link and checking for a 200 HTTP status code. If the post is expired or the link is broken, discard it and find a replacement.

        ### QUALIFICATION CRITERIA
        Evaluate the candidate's fit based on:
        - Technical Match: Python/GCP/Airflow/Vertex AI.
        - Impact Match: Proven $1M annual cost savings via automation.
        - Specialized Fit: Experience with 'Agent Development Kits' and A/B Testing.

        ### OUTPUT FORMAT
        Provide the results strictly in the following JSON-style list format. DO NOT INCLUDE ANY EXPLANATORY TEXT OUTSIDE THE JSON. Ensure all fields are filled for each job listing.:

        [{
            "job_title": "Position Title",
            "company": "Company Name",
            "location": "The city/state or 'Remote'",
            "remote": "true if the job is fully remote or hybrid; false if strictly on-site. Should be a BOOLEAN value."
            "link": "Direct link to job posting",
            "explanation": "Briefly explain why this job fits the candidate's specific achievements (e.g., matching the $1M savings or Vertex AI experience).",
            "fit_score": "Score out of 100, provide only the number, keeping it as an integer",
            "qualification_analysis": "A detailed explanation of why the candidate is qualified (e.g., 'Matches their need for BigQuery cost auditing').",
            "skill_gaps": "List any specific tools from the JD not clearly highlighted in the resume (e.g., 'dbt', 'Snowflake')."
            "compensation": "Compensation range if listed (e.g., '$160,000 - $210,000'); NULL if not listed."
        }]

        ### SEARCH SCOPE
        Search broadly across the USA, prioritizing high-growth tech hubs and remote-first companies.

        ### CONSTRAINTS
        - Ensure all links are active and lead to the specific job page.
        - Do not include expired roles.
        - Focus on companies in Tech, Retail, or Finance where data optimization is a priority.

        ### URL CONSTRAINTS
        - DO NOT provide links to search result pages (e.g., URLs containing '/jobs/search', '/jobs/results', or '?q=').
        - PRIORITIZE "Deep Links" that lead directly to a specific Job Description (JD).
        - PREFER links from the company's own 'careers' portal (e.g., Greenhouse, Lever, Workday, or the company domain).
        - If a direct JD link cannot be found for a listing, skip that listing and find another.
        
        """

    # job_schema = {
    #     "type": "ARRAY",
    #     "items": {
    #         "type": "OBJECT",
    #         "properties": {
    #             "job_title": {"type": "STRING"},
    #             "company": {"type": "STRING"},
    #             "link": {"type": "STRING"},
    #             "location": {"type": "STRING"},
    #             "remote": {"type": "BOOLEAN"},
    #             "fit_score": {"type": "INTEGER"},
    #             "qualification_analysis": {"type": "STRING"},
    #             "skill_gaps": {"type": "STRING"}
    #         },
    #         "required": ["job_title", "company", "link", "location", "remote", "fit_score"]
    #     }
    # }

    # search_tool = types.Tool(
    #     google_search=types.GoogleSearch()
    # )
    # 4. Run the request using the Flash model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[resume_summary, resume_full, prompt],
        config=types.GenerateContentConfig(
            tools=[types.Tool(url_context=types.UrlContext()), types.Tool(google_search=types.GoogleSearch())],
            # response_mime_type="application/json",
            # response_schema=job_schema,
            temperature=1.0 # Optimized for search grounding
        )
    )

    # Close the sync client to release resources.
    # client.close()

    # Attempt to parse the response text as JSON
    try:
        result = json.loads(response.text.replace('```json', '').replace('```', ''))
    except Exception as e:
        print("Error found in JSON transformation: ", e)
        print(response.candidates[0].grounding_metadata.web_search_queries)
        result = response

    return result

def load_to_github(file_name, jobs_data, start_dict):

    r_raw = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=start_dict['HEADERS_RAW'])

    # Load historical data
    jobs_data_full = json.loads(r_raw.content)

    r_meta = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=start_dict['HEADERS_META'])

    # Filter out already processed posts
    jobs_data = [i for i in jobs_data if i['link'] in list(
        set([i['link'] for i in jobs_data]) - set([i['link'] for i in jobs_data_full])
    )]

    # Process new posts
    if len(jobs_data) > 0:
        
        # Append new data to historical data
        jobs_data_full.extend(jobs_data)
        
        # Save the updated historical data back to the file
        data = json.dumps({
            "message": "Updating data and metadata",
            "content": base64.b64encode(json.dumps(jobs_data_full).encode('cp1252')).decode('ascii'),
            "sha": r_meta.json()['sha']
        })
        r1 = requests.put(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}?ref=main", headers=start_dict['HEADERS_META'], data=data)


def update_submodule_direct(start_dict):
    """Update the GitHub submodule into the main repository"""
    
    # Get the latest commit SHA of the branch
    ref_res = requests.get(f"https://api.github.com/repos/{start_dict['PARENT_OWNER']}/{start_dict['PARENT_REPO']}/git/ref/heads/{start_dict['BRANCH']}", headers=start_dict['HEADERS_META'])
    ref_res.raise_for_status()
    last_commit_sha = ref_res.json()['object']['sha']
    
    # Get the Tree SHA from that commit
    commit_res = requests.get(f"https://api.github.com/repos/{start_dict['PARENT_OWNER']}/{start_dict['PARENT_REPO']}/git/commits/{last_commit_sha}", headers=start_dict['HEADERS_META'])
    parent_tree_sha = commit_res.json()['tree']['sha']

    # Get the real SHA from the submodule repo
    sub_branch = start_dict.get('SUBMODULE_BRANCH', start_dict['BRANCH'])
    sub_ref_res = requests.get(
        f"https://api.github.com/repos/{start_dict['SUBMODULE_OWNER']}/{start_dict['SUBMODULE_REPO']}/git/ref/heads/{sub_branch}", 
        headers=start_dict['HEADERS_META']
    )
    sub_ref_res.raise_for_status()
    real_submodule_sha = sub_ref_res.json()['object']['sha']

    # Create a new Tree pointing to the real submodule commit
    tree_payload = {
        "base_tree": parent_tree_sha,
        "tree": [{
            "path": start_dict['SUBMODULE_PATH'],
            "mode": "160000",
            "type": "commit",
            "sha": real_submodule_sha
        }]
    }
    tree_post = requests.post(f"https://api.github.com/repos/{start_dict['PARENT_OWNER']}/{start_dict['PARENT_REPO']}/git/trees", headers=start_dict['HEADERS_META'], json=tree_payload)
    new_tree_sha = tree_post.json()['sha']

    # Create a new Commit
    commit_payload = {
        "message": f"Update submodule {start_dict['SUBMODULE_PATH']} to {real_submodule_sha[:7]}",
        "tree": new_tree_sha,
        "parents": [last_commit_sha]
    }
    commit_post = requests.post(f"https://api.github.com/repos/{start_dict['PARENT_OWNER']}/{start_dict['PARENT_REPO']}/git/commits", headers=start_dict['HEADERS_META'], json=commit_payload)
    new_commit_sha = commit_post.json()['sha']
    
    # Update the Branch Reference to point to the new Commit
    ref_update_payload = {"sha": new_commit_sha, "force": False}
    patch_res = requests.patch(f"https://api.github.com/repos/{start_dict['PARENT_OWNER']}/{start_dict['PARENT_REPO']}/git/refs/heads/{start_dict['BRANCH']}", headers=start_dict['HEADERS_META'], json=ref_update_payload)
    
    if patch_res.status_code == 200:
        print(f"Successfully updated submodule! New parent commit: {new_commit_sha[:7]}")
    else:
        print(f"Failed to update ref: {patch_res.text}")

# %%

# GEMINI_KEY = get_key("Gemini-API")

start_dict = init()

today = datetime.date.today().isoformat()

# Initialize GenAI Client with API Key from Secret Manager
client = genai.Client(api_key=start_dict['gemini_key'])

try:
    jobs = genai_response(client)

    for job in jobs:
        job['dt'] = today

        if isinstance(job['skill_gaps'], list):
            job['skill_gaps'] = ', '.join(job['skill_gaps'])

    func_load_data(jobs, 'gemini_jobs')

    load_to_github("JOBS.json", jobs, start_dict)

    update_submodule_direct(start_dict)
except Exception as e:
    print(e)
finally:
    client.close()

# %%
