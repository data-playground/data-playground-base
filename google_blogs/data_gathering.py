# %%

import base64
import hashlib
import json
import re
import secrets
import unicodedata
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai
from google.cloud import secretmanager
from selenium import webdriver
from tqdm import tqdm

# %%

class BLOG_SITES:

    def __init__(self):
        self.gemini_key = self.get_key("Gemini-API")
        self.github_key = self.get_key("Github-Key")

        self.SUBMODULE_OWNER = 'data-playground'
        self.SUBMODULE_REPO = 'data-playground-data'
        self.PARENT_OWNER = 'data-playground'
        self.PARENT_REPO = 'data-playground.github.io'
        self.BRANCH = 'main'
        self.SUBMODULE_PATH = '_data/data_playground_data' 

        self.GOOGLE_WORKSPACE_BLOG = "https://workspace.google.com/blog/"
        self.GOOGLE_APPS_UPDATES = "https://feeds.feedburner.com/GoogleAppsUpdates"
        self.GOOGLE_CLOUD_BLOG = "https://cloudblog.withgoogle.com/rss"
        self.GOOGLE_RESEARCH_BLOG = "https://research.google/blog/rss/"
        self.GOOGLE_BLOG = "https://blog.google/rss/"
        self.GOOGLE_TECHNOLOGY_BLOG = "https://blog.google/technology/rss/"
        self.GOOGLE_DEEPMIND_BLOG = "https://blog.google/technology/google-deepmind/rss/"
        self.GOOGLE_DEVELOPERS_BLOG = "https://developers.googleblog.com/rss/"
        self.GOOGLE_DEVS_SITEMAP = "https://developers.google.com/sitemap.xml"

        self.HEADERS_RAW={
            "Accept": "application/vnd.github.v3.raw", 
            "Authorization": f"Bearer {self.github_key}", 
            "X-GitHub-Api-Version": "2022-11-28"
        }

        self.HEADERS_META={
            "Accept": "application/vnd.github+json", 
            "Authorization": f"Bearer {self.github_key}", 
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def get_key(self, SECRET_NAME):
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

    def fetch_and_parse_feed(self, url, site_name, enrich = False):
        """Fetches and parses the RSS feed."""

        print(f"Fetching feed from: {url}")
        feed = feedparser.parse(url)

        # file_name = f"C:\\Users\\Llubr\\Desktop\\Github\\data-playground-data\\{site_name}.json"
        file_name = f"{site_name}.json"

        post_list = []
        
        # Iterate over each entry (blog post)
        for entry in feed.entries:
            # Get the publish date (if available) and format it
            published_date = ""
            if hasattr(entry, 'published_parsed'):
                # Convert the time structure to a datetime object, then format
                published_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M:%S")

            # Create a dictionary for the post
            post_info = {
                "website": site_name,
                "link": entry.link,
                "title": json.loads(json.dumps(entry.title)),
                "thumbnail": entry.media_content[0]['url'] if 'media_content' in entry.keys() else entry.media_thumbnail[0]['url'] if "media_thumbnail" in entry.keys() else None,
                "author": json.loads(json.dumps(entry.author_detail.name if 'author_detail' in entry.keys() else None)),
                "track": None,
                "description": json.loads(json.dumps(entry.summary_detail.value if 'summary_detail' in entry.keys() else entry.summary[:200] if 'summary' in entry.keys() else None)),
                "published_date": published_date
            }
            post_list.append(post_info)

        r_raw = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=self.HEADERS_RAW)

        # Check if historical data file exists
        # if os.path.exists(file_name):
        if r_raw.status_code == 200:
            # Load existing historical data
            # with open(file_name, "r", encoding="utf-8") as f:
                # post_data_full = json.load(f)
            post_data_full = json.loads(r_raw.content)

            r_meta = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=self.HEADERS_META)

            # Filter out posts that are already in historical data
            post_list = [i for i in post_list if i['link'] in list(
                set([i['link'] for i in post_list]) - set([i['link'] for i in post_data_full])
            )]

            # Check if there are new posts to add
            if len(post_list) > 0:
                # Enrich new posts with AI-generated summaries and tags
                if enrich:
                    print("Enriching data")
                    gen_ai_data = self.get_ai_summary_tags([i['link'] for i in post_list if not 'summary' in i.keys()][:100], content_type = 'video' if site_name.upper().startswith('YOUTUBE') else 'article')
                    post_list = self.merge_list_left_join(post_list, gen_ai_data)
                
                # Append new posts to historical data
                post_data_full.extend(post_list)
                post_data_full = sorted(post_data_full, key = lambda x: x['published_date'], reverse=True)

                data = json.dumps({
                    "message": "Updating data and metadata",
                    "content": base64.b64encode(json.dumps(post_data_full).encode('cp1252')).decode('ascii'),
                    "sha": r_meta.json()['sha']
                })
                r1 = requests.put(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}?ref=main", headers=self.HEADERS_META, data=data)
        else:
            # If no historical data, use the current post list
            post_data_full = sorted(post_list, key = lambda x: x['published_date'], reverse=True)

            data = json.dumps({
                "message": "Updating data and metadata",
                "content": base64.b64encode(json.dumps(post_data_full).encode('cp1252')).decode('ascii')
            })
            r1 = requests.put(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}?ref=main", headers=self.HEADERS_META, data=data)

        # Save the updated historical data back to the file
        # with open(file_name, "w", encoding="utf-8") as f:    
            # json.dump(post_data_full, f, ensure_ascii=False, indent= 4)    

        return post_list

    def workspace_data_dict(self, card):
        """Extracts data from a workspace blog card."""

        # Find the parent <a> tag of the card
        card_a = card.find_parent('a')

        # Extract author and read time
        author_read = card_a.select_one('p').text if card_a.select_one('p') else card_a.parent.parent.select('p')[2].text

        return {
            'website': "GOOGLE_WORKSPACE_BLOG",
            'link': card_a.get('href'),
            'title': card_a.select_one('h3').text if card_a.select_one('h3') else card_a.parent.parent.select_one('h4').text, #card_a.get('data-g-action'),
            'thumbnail': card_a.select_one('img').get('src'),
            'author': re.match('By (.*) • (.*) read', author_read).group(1),
            'read_time': re.match('By (.*) • (.*) read', author_read).group(2),
            'track': card_a.select_one('*[track-type="tag"]').get('track-name') if card_a.select_one('*[track-type="tag"]') else card_a.parent.parent.select_one('*[track-type="tag"]').get('track-name')
        }

    def clean_html(self, soup):
        """Cleans unwanted attributes from HTML content."""

        # List of attributes to delete
        attrs_to_delete = ['class', 'data-node-index', 'data-ogpc', 'data-p', 'jsdata', 'jsmodel', 'jsrenderer', 'view', 'jsname', 'jsaction', 'jscontroller', 'track-name', 'track-type']

        # Iterate through all tags and remove unwanted attributes
        for tag in soup.find_all(True):
            for attr in list(tag.attrs):
                if attr in attrs_to_delete:
                    del tag.attrs[attr]

        return str(soup)

    def post_detail(self, post):
        """Fetches details of a blog post"""

        # Get the page content
        r = requests.get(post['link'])
        soup = BeautifulSoup(r.content)

        # Extract post description and published date
        post['description'] = soup.select_one('meta[name="description"]').get('content')
        post['published_date'] = soup.select_one('meta[name="track-metadata-page_first_published"]').get('content')
        # post['html'] = clean_html(soup.select_one('c-wiz'))

        return post

    def get_user_confirmation(self, prompt_message="Do you want to proceed? (yes/no): "):
        """
        Prompts the user for confirmation and returns True for 'yes' or False for 'no'.
        Handles invalid input by repeatedly asking until a valid response is given.
        """

        # Loop until valid input is received
        while True:
            user_input = input(prompt_message).lower().strip()
            if user_input in ['yes', 'y']:
                return True
            elif user_input in ['no', 'n']:
                return False
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def deduplicate_list_of_dicts(self, list_of_dicts, key):
        """Deduplicates a list of dictionaries based on a specified key."""

        # Create a set to track seen keys
        seen_keys = set()

        # Create a new list to hold unique dictionaries
        unique_dicts = []

        # Iterate through the list of dictionaries
        for d in list_of_dicts:
            if d[key] not in seen_keys:
                unique_dicts.append(d)
                seen_keys.add(d[key])

        return unique_dicts

    def workspace_data_get(self):
        """Fetches and processes data from the Google Workspace blog."""

        file_name = "GOOGLE_WORKSPACE_BLOG.json"

        # Send a GET request to the blog URL
        r = requests.get(self.GOOGLE_WORKSPACE_BLOG)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(r.content)

        # Select all blog post cards
        cards = soup.select('.PBkdHd')

        # Extract data from each card
        worskpace_data = [self.workspace_data_dict(card) for card in cards]

        r_raw = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=self.HEADERS_RAW)

        # Load historical data
        # with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_WORKSPACE_BLOG.json", "r", encoding="utf-8") as f:
        #     worskpace_data_hist_full = json.load(f)
        worskpace_data_hist_full = json.loads(r_raw.content)

        r_meta = requests.get(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}", headers=self.HEADERS_META)

        # Filter out already processed posts
        worskpace_data = [i for i in worskpace_data if i['link'] in list(
            set([i['link'] for i in worskpace_data]) - set([i['link'] for i in worskpace_data_hist_full])
        )]

        # Process new posts
        if len(worskpace_data) > 0:
            # Get the detailed information for new posts
            for i, post_dict in tqdm(enumerate(worskpace_data), total=len(worskpace_data)):
                self.post_detail(post_dict)

            # Enrich new posts with AI-generated summaries and tags
            gen_ai_data = self.get_ai_summary_tags([i['link'] for i in worskpace_data if not 'summary' in i.keys()][:100], content_type = 'article')
            
            # Merge AI-generated data with workspace data
            worskpace_data = self.merge_list_left_join(worskpace_data, gen_ai_data)

            # Append new data to historical data
            worskpace_data_hist_full.extend(worskpace_data)

            # Deduplicate historical data based on 'link' and sort by published date
            worskpace_data_hist_full = sorted(worskpace_data_hist_full, key = lambda x: x['published_date'], reverse=True)
            
            # Save the updated historical data back to the file
            # with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_WORKSPACE_BLOG.json", "w", encoding="utf-8") as f:    
            #     json.dump(worskpace_data_hist_full, f, ensure_ascii=False, indent= 4)

            data = json.dumps({
                "message": "Updating data and metadata",
                "content": base64.b64encode(json.dumps(worskpace_data_hist_full).encode('cp1252')).decode('ascii'),
                "sha": r_meta.json()['sha']
            })
            r1 = requests.put(f"https://api.github.com/repos/data-playground/data-playground-data/contents/{file_name}?ref=main", headers=self.HEADERS_META, data=data)
        return worskpace_data

    def workspace_hist(self):
        """Fetches historical data from the Google Workspace blog using Selenium."""

        # Initialize the Chrome WebDriver
        driver = webdriver.Chrome()

        # Navigate to the Google Workspace blog
        driver.get(self.GOOGLE_WORKSPACE_BLOG)

        # Prompt user for confirmation to proceed
        if self.get_user_confirmation():
            # Parse the page source using BeautifulSoup
            soup_hist = BeautifulSoup(driver.page_source)
            
            # Select all blog post cards
            cards_hist = soup_hist.select('.PBkdHd')

            # Extract data from each card and deduplicate
            worskpace_data_hist = [dict(t) for t in set(tuple(d.items()) for d in [workspace_data_dict(card) for card in cards_hist])]

            # Close the WebDriver
            driver.quit()

            # Get detailed information for each historical post
            for i, post_dict in tqdm(enumerate(worskpace_data_hist), total=len(worskpace_data_hist)):
                self.post_detail(post_dict)

            return worskpace_data_hist

    def google_devs_map(self, link):
        """Fetches and parses the Google Developers sitemap."""

        # Send a GET request to the sitemap URL
        r = requests.get(link)

        # Parse the XML content
        root = ET.fromstring(r.content)

        # Extract link and last modified date from each child element
        children = [self.get_link_date_from_map(child) for child in root]

        return children

    def get_link_date_from_map(self, child):
        """Extracts link and last modified date from a sitemap child element."""

        # Define the XML namespaces
        namespaces = {
            'ns0': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            # You don't need the 'html' namespace unless you plan to search for those tags.
        }

        # Extract loc and lastmod elements
        loc_node = child.find('ns0:loc', namespaces)
        lastmod_node = child.find('ns0:lastmod', namespaces)

        # Extract text content if nodes are found
        link_url = loc_node.text if loc_node is not None else None
        mod_date = lastmod_node.text if lastmod_node is not None else None

        return {
            'link': link_url,
            'lastmod': mod_date
        }

    def get_google_devs_full_map(self):
        """Fetches the full Google Developers sitemap."""

        # Get the initial sitemap map
        init_map = self.google_devs_map(self.GOOGLE_DEVS_SITEMAP)

        # Initialize an empty list to hold the full sitemap
        full_map = []

        # Iterate through each page in the initial sitemap and fetch its links
        for page in init_map:
            print(page['link'])
            full_map.extend(self.google_devs_map(page['link']))

        return full_map

    def merge_list_left_join(self, list1, list2, join_key = 'link'):
        """Merges two lists of dictionaries based on a common key using a left join approach."""

        # Initialize an empty list to hold the merged results
        merged_list = []

        # Create the lookup from list2
        lookup = {item[join_key]: item for item in list2}

        # Iterate through list1 and use an if/else
        for item1 in list1:
            item2_match = lookup.get(item1[join_key])
            
            if item2_match:
                # Match found: merge them
                merged_list.append({**item1, **item2_match})
            else:
                # No match: just append the original item from list1
                merged_list.append(item1)

        return merged_list

    def get_ai_summary_tags(self, content_list, content_type = 'article'):
        """Generates AI summaries and tags for a list of content URLs using Gemini API."""

        # Initialize the Gemini client
        client = genai.Client(api_key=self.gemini_key)

        # Prepare the prompt
        prompt = f'''
    You are an expert news digest curator for a daily push notification and email service. Your goal is to research the content at each provided URL, generate a concise, highly information-dense summary, and a set of relevant tags. The entire output must be returned in a strict JSON array format.

    ---
    **INSTRUCTIONS**
    1.  **ACCESS CONTENT:** For each object in the 'INPUT ITEMS' list, access the content available at the provided 'link'.
        * **For Blog Posts:** Extract the full text content.
        * **For YouTube Videos:** Analyze the video's title, description, and transcript/key visual elements.
    2.  **GENERATE SUMMARY:** Create a summary that is **2 to 3 sentences long**. The summary must be **information-dense** (including specific people, products, companies, and key statistics/takeaways) to serve as a high-quality chunk for a Retrieval-Augmented Generation (RAG) system.
    3.  **GENERATE METADATA:**
        * **tools (3-8 items):** Identify and list only **Google-branded products and services** mentioned in the content. This includes, but is not limited to: Google Sheets, Docs, Slides, Gemini, NotebookLM, Drive, Meet, Chat, Classroom, and Google Cloud services. **Do not include non-Google products or general hardware here.**
        * **tags (5-8 items):** List all **main topics, industries, broad keywords, and core concepts** discussed. This list may include non-Google products, hardware, and company names as general keywords (e.g., 'Dell', 'Home Office', 'Networking').
    4.  **STRICT OUTPUT:** The final output MUST be a JSON array. Do not include any text, headers, preambles, or markdown outside of the JSON array itself.

    ---
    **INPUT ARTICLES**

    {content_list}

    ---
    **EXPECTED JSON OUTPUT FORMAT**

    [
        {{
            "link": "...",
            "summary": "...",
            "tools": ["Google Docs", "Gemini", "Drive"], // ONLY Google Products
            "tags": ["Collaboration", "AI", "Productivity"] // Broader Topics/Keywords    }}
    ]
        '''

        # Generate content based on the content type
        if content_type == 'video':
            # Prepare contents for video type
            contents = [
                {
                    "fileData": {
                        "fileUri": link,
                        "mimeType": "video/*" 
                    }
                } for link in content_list
            ] + [
                {
                    "text": prompt
                }
            ]

            # Generate content using the Gemini API
            response_1 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents, 
            )
        else:
            # For article type, use the prompt directly
            response_1 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt, 
            )

        # Close the sync client to release resources.
        client.close()

        # Attempt to parse the response text as JSON
        try:
            result = json.loads(response_1.text.replace('```json', '').replace('```', ''))
        except Exception as e:
            print("Error found in JSON transformation: ", e)
            result = response_1

        return result

    def extract_app_update_data(self, r):
        """Extracts app update data from the Google Apps Updates blog page."""

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(r.content)

        # Select all blog post cards
        cards = soup.select('article')

        # Extract relevant data from each card
        content = [{
            "website": "GOOGLE_APPS_DATA",
            "link": card.select_one('a').get('href'),
            "title": card.select_one('a').get('title'),
            "thumbnail": card.select_one('img').get('src'),
            "author": None,
            "track": None,
            "description": unicodedata.normalize("NFKD", card.select_one(".blog-summary__body").text.strip()),
            "published_date": datetime.strptime(card.select_one(".blog-summary__date").text.strip(), "%A, %B %d, %Y").strftime("%Y-%m-%d %H:%M:%S")
        } for card in cards]

        # Determine the new date for pagination
        new_date = (datetime.strptime(content[-1]['published_date'], "%Y-%m-%d %H:%M:%S") + timedelta(days=1)).strftime('%Y-%m-%d')

        return content, new_date

    def app_update_hist(self):
        """Fetches historical app update data from the Google Apps Updates blog."""
        
        # Initialize an empty list to hold all content
        all_content = []

        # Set the starting date to today
        new_date = date.today().strftime('%Y-%m-%d')

        # Loop to fetch data until reaching the specified date
        while new_date >= '2020-01-01': # The articles actually date back to 2007-02-27
            print(new_date)

            # Fetch the blog page for the current date
            r = requests.get(f"https://workspaceupdates.googleblog.com/search?updated-max={new_date}T00:00:00-05:00&max-results=20&start=20&by-date=true")

            # Extract content and the new date for pagination
            content, new_date = self.extract_app_update_data(r)

            # Append the extracted content to the all_content list
            all_content.extend(content)

        # Deduplicate the collected content based on 'link'
        all_content = self.deduplicate_list_of_dicts(all_content)

        # Load existing historical data
        with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_APPS_DATA.json", "r", encoding="utf-8") as f:
            google_apps_full = json.load(f)

        # Filter out already processed posts
        all_content = [i for i in all_content if i['link'] in list(
            set([i['link'] for i in all_content]) - set([i['link'] for i in google_apps_full])
        )]

        if len(all_content) > 0:
            # Append new content to historical data
            google_apps_full.extend(all_content)

            # Sort historical data by published date
            google_apps_full = sorted(google_apps_full, key = lambda x: x['published_date'], reverse=True)
            
            # Save the updated historical data back to the file
            with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_APPS_DATA.json", "w", encoding="utf-8") as f:    
                json.dump(google_apps_full, f, ensure_ascii=False, indent= 4)

        return google_apps_full

    def enrich_catchup(self, site_name, qty):
        """Enriches existing data with AI-generated summaries and tags."""

        # Construct the file path for the JSON data
        file_name = f"C:\\Users\\Llubr\\Desktop\\Github\\data-playground-data\\{site_name}.json"

        # Load existing data
        with open(file_name, "r", encoding="utf-8") as f:
            post_data_full = json.load(f)

        # Enrich data with AI-generated summaries and tags
        gen_ai_data = self.get_ai_summary_tags([i['link'] for i in post_data_full if not 'summary' in i.keys()][:qty], content_type = 'video' if site_name.upper().startswith('YOUTUBE') else 'article')
        
        # Merge AI-generated data with existing data
        post_data_full = self.merge_list_left_join(post_data_full, gen_ai_data)

        # Save the enriched data back to the file
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(sorted(post_data_full, key = lambda x: x['published_date'], reverse=True), f, ensure_ascii=False, indent = 4)

    def generate_random_sha256(self):
        """Generates a random SHA256 hash."""
        # 1. Generate 32 cryptographically secure random bytes (256 bits)
        random_data = secrets.token_bytes(16)
        
        # 2. Create a new SHA256 hash object
        hash_object = hashlib.sha256()
        
        # 3. Update the hash object with the random bytes
        hash_object.update(random_data)
        
        # 4. Get the hexadecimal representation (64 characters long)
        random_sha256_hash = hash_object.hexdigest()
        
        return random_sha256_hash

    def update_submodule_direct(self):
        # 1. Get the latest commit SHA of the branch
        ref_res = requests.get(f"https://api.github.com/repos/{self.PARENT_OWNER}/{self.PARENT_REPO}/git/ref/heads/{self.BRANCH}", headers=self.HEADERS_META).json()
        last_commit_sha = ref_res['object']['sha']

        # 2. Create a new Tree that updates the submodule reference
        # Mode '160000' is the magic code for a Git Submodule
        tree_payload = {
            "base_tree": last_commit_sha,
            "tree": [{
                "path": self.SUBMODULE_PATH,
                "mode": "160000",
                "type": "commit",
                "sha": hashlib.sha1(self.generate_random_sha256().encode('utf-8')).hexdigest()
            }]
        }
        tree_res = requests.post(f"https://api.github.com/repos/{self.PARENT_OWNER}/{self.PARENT_REPO}/git/trees", headers=self.HEADERS_META, json=tree_payload).json()
        new_tree_sha = tree_res['sha']

        # 3. Create a new Commit pointing to the new Tree
        commit_payload = {
            "message": f"Update submodule {self.SUBMODULE_PATH}",
            "tree": new_tree_sha,
            "parents": [last_commit_sha]
        }
        commit_res = requests.post(f"https://api.github.com/repos/{self.PARENT_OWNER}/{self.PARENT_REPO}/git/commits", headers=self.HEADERS_META, json=commit_payload).json()
        new_commit_sha = commit_res['sha']

        # 4. Update the Branch Reference to point to the new Commit
        ref_update_payload = {"sha": new_commit_sha, "force": False}
        patch_res = requests.patch(f"https://api.github.com/repos/{self.PARENT_OWNER}/{self.PARENT_REPO}/git/refs/heads/{self.BRANCH}", headers=self.HEADERS_META, json=ref_update_payload)

        if patch_res.status_code == 200:
            print(f"Successfully pushed: {new_commit_sha[:7]}")
        else:
            print("Error:", patch_res.json())

# %%

################################################
################ Usage Examples ################
################################################

#### Instantiate the class 
# blog_sites = BLOG_SITES()

#### Get data for the Workspace blog
# GOOGLE_WORKSPACE_DATA = blog_sites.workspace_data_get()

#### Get data for various Google blogs (runs with enrich=True run the Gemini API process to get AI generated data)
# GOOGLE_APPS_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_APPS_UPDATES, "GOOGLE_APPS_UPDATES", enrich=True)
# GOOGLE_CLOUD_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_CLOUD_BLOG, "GOOGLE_CLOUD_BLOG")
# GOOGLE_RESEARCH_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_RESEARCH_BLOG, "GOOGLE_RESEARCH_BLOG")
# GOOGLE_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_BLOG, "GOOGLE_BLOG")
# GOOGLE_TECHNOLOGY_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_TECHNOLOGY_BLOG, "GOOGLE_TECHNOLOGY_BLOG", enrich=True)
# GOOGLE_DEEPMIND_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_DEEPMIND_BLOG, "GOOGLE_DEEPMIND_BLOG")
# GOOGLE_DEVELOPERS_DATA = blog_sites.fetch_and_parse_feed(blog_sites.GOOGLE_DEVELOPERS_BLOG, "GOOGLE_DEVELOPERS_BLOG")

#### Get full site map for google devs documentation
# # GOOGLE_DEVS_FULLMAP = blog_sites.get_google_devs_full_map()

#### Get data for Tableau related blogs
# flerlagetwins = blog_sites.fetch_and_parse_feed("https://www.flerlagetwins.com/feeds/posts/default", "TABLEAU_flerlagetwins")
# vizwiz = blog_sites.fetch_and_parse_feed("https://www.vizwiz.com/feeds/posts/default", "TABLEAU_vizwiz")
# storytellingwithdata = blog_sites.fetch_and_parse_feed("https://www.storytellingwithdata.com/blog?format=rss", "TABLEAU_storytellingwithdata")
# playfairdata = blog_sites.fetch_and_parse_feed("https://playfairdata.com/feed/", "TABLEAU_playfairdata")
# theinformationlab = blog_sites.fetch_and_parse_feed("https://www.theinformationlab.com/", "TABLEAU_theinformationlab")

#### Get data for Google YouTube channels
# YOUTUBE_GOOGLE_WORKSPACE = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCBmwzQnSoj9b6HzNmFrg_yw", "YOUTUBE_GOOGLE_WORKSPACE")
# YOUTUBE_GOOGLE_DEVELOPERS = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UC_x5XG1OV2P6uZZ5FSM9Ttw", "YOUTUBE_GOOGLE_DEVELOPERS")
# YOUTUBE_GOOGLE_CREATORS = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCXZNlYNefXV3iMcyzszt_9Q", "YOUTUBE_GOOGLE_CREATORS")
# YOUTUBE_GOOGLE_DEEPMIND = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCP7jMXSY2xbc3KCAE0MHQ-A", "YOUTUBE_GOOGLE_DEEPMIND")
# YOUTUBE_GOOGLE = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCK8sQmJBp8GCxrOtXWBpyEA", "YOUTUBE_GOOGLE")
# YOUTUBE_GOOGLE_CLOUD = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCTMRxtyHoE3LPcrl-kT4AQQ", "YOUTUBE_GOOGLE_CLOUD")
# YOUTUBE_GOOGLE_QUANTUMAI = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCO5cgpkcYjnsdxZdEDR3Jog", "YOUTUBE_GOOGLE_QUANTUMAI")
# YOUTUBE_GOOGLE_TALKS = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCbmNph6atAoGfqLoCL_duAg", "YOUTUBE_GOOGLE_TALKS")

#### Get data for Tableau related YouTube channels
# YOUTUBE_TABLEAU_FLERLAGE = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCDyr5VgVvkmfhHpeMUB8ZDA", "YOUTUBE_TABLEAU_FLERLAGE")
# YOUTUBE_TABLEAU = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCWGrtxO6JrPSDUcgp3Qm_Gw", "YOUTUBE_TABLEAU")
# YOUTUBE_TABLEAU_TIM = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UC7HYxRWmaNlJux-X7rNLZyw", "YOUTUBE_TABLEAU_TIM")
# YOUTUBE_TABLEAU_VIZWIZ = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCTlX7UpqASrldmx5_CpG3CA", "YOUTUBE_TABLEAU_VIZWIZ")
# YOUTUBE_TABLEAU_SQLBELLE = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCW2E1sGBVde5WMMxEh5CW4w", "YOUTUBE_TABLEAU_SQLBELLE")
# YOUTUBE_TABLEAU_DATAFAM = blog_sites.fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCuDUG9ZHa-IlTm6y-2Lko_Q", "YOUTUBE_TABLEAU_DATAFAM")

#### Update submodule to get latest updates to data-playground
# blog_sites.update_submodule_direct()
# %%
