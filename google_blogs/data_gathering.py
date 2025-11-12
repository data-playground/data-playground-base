# %%

import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from tqdm import tqdm 
import json
import os
import xml.etree.ElementTree as ET
from google import genai

# %%

def fetch_and_parse_feed(url, site_name):
    """Fetches and parses the RSS feed."""
    print(f"Fetching feed from: {url}")
    feed = feedparser.parse(url)

    file_name = f"C:\\Users\\Llubr\\Desktop\\Github\\data-playground-data\\{site_name}.json"

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
            "title": entry.title,
            "thumbnail": entry.media_content[0]['url'] if 'media_content' in entry.keys() else entry.media_thumbnail[0]['url'] if "media_thumbnail" in entry.keys() else None,
            "author": entry.author_detail.name if 'author_detail' in entry.keys() else None,
            "track": None,
            "description": entry.summary_detail.value if 'summary_detail' in entry.keys() else entry.summary[:200] if 'summary' in entry.keys() else None,
            "published_date": published_date
        }
        post_list.append(post_info)

    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            post_data_full = json.load(f)

        post_list = [i for i in post_list if i['link'] in list(
            set([i['link'] for i in post_list]) - set([i['link'] for i in post_data_full])
        )]

        post_data_full.extend(post_list)
        # post_data_full = deduplicate_list_of_dicts(post_data_full, 'link')
        post_data_full = sorted(post_data_full, key = lambda x: x['published_date'], reverse=True)
    else:
        post_data_full = sorted(post_list, key = lambda x: x['published_date'], reverse=True)

    with open(file_name, "w") as f:    
        json.dump(post_data_full, f, indent= 4)    

    return post_list

# %%

def workspace_data_dict(card):
    card_a = card.find_parent('a')
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

def clean_html(soup):
    attrs_to_delete = ['class', 'data-node-index', 'data-ogpc', 'data-p', 'jsdata', 'jsmodel', 'jsrenderer', 'view', 'jsname', 'jsaction', 'jscontroller', 'track-name', 'track-type']

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr in attrs_to_delete:
                del tag.attrs[attr]

    return str(soup)


def post_detail(post):
    r = requests.get(post['link'])
    soup = BeautifulSoup(r.content)

    post['description'] = soup.select_one('meta[name="description"]').get('content')
    post['published_date'] = soup.select_one('meta[name="track-metadata-page_first_published"]').get('content')
    # post['html'] = clean_html(soup.select_one('c-wiz'))

    return post

def get_user_confirmation(prompt_message="Do you want to proceed? (yes/no): "):
    """
    Prompts the user for confirmation and returns True for 'yes' or False for 'no'.
    Handles invalid input by repeatedly asking until a valid response is given.
    """
    while True:
        user_input = input(prompt_message).lower().strip()
        if user_input in ['yes', 'y']:
            return True
        elif user_input in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def deduplicate_list_of_dicts(list_of_dicts, key):
    seen_keys = set()
    unique_dicts = []
    for d in list_of_dicts:
        if d[key] not in seen_keys:
            unique_dicts.append(d)
            seen_keys.add(d[key])
    return unique_dicts
# %%

def workspace_data_get(GOOGLE_WORKSPACE_BLOG):
    r = requests.get(GOOGLE_WORKSPACE_BLOG)

    soup = BeautifulSoup(r.content)

    cards = soup.select('.PBkdHd')

    worskpace_data = [workspace_data_dict(card) for card in cards]

    # with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\worskpace_data.json", "r") as f:
    with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_WORKSPACE_BLOG.json", "r") as f:
        worskpace_data_hist_full = json.load(f)

    worskpace_data = [i for i in worskpace_data if i['link'] in list(
        set([i['link'] for i in worskpace_data]) - set([i['link'] for i in worskpace_data_hist_full])
    )]

    if len(worskpace_data) > 0:
        for i, post_dict in tqdm(enumerate(worskpace_data), total=len(worskpace_data)):
            post_detail(post_dict)

        worskpace_data_hist_full.extend(worskpace_data)
        # worskpace_data_hist_full = deduplicate_list_of_dicts(worskpace_data_hist_full, 'link')
        worskpace_data_hist_full = sorted(worskpace_data_hist_full, key = lambda x: x['published_date'], reverse=True)
        
        with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_WORKSPACE_BLOG.json", "w") as f:    
            json.dump(worskpace_data_hist_full, f, indent= 4)

    return worskpace_data

# %%

def workspace_hist(GOOGLE_WORKSPACE_BLOG):
    driver = webdriver.Chrome()

    driver.get(GOOGLE_WORKSPACE_BLOG)

    if get_user_confirmation():

        soup_hist = BeautifulSoup(driver.page_source)

        cards_hist = soup_hist.select('.PBkdHd')

        worskpace_data_hist = [dict(t) for t in set(tuple(d.items()) for d in [workspace_data_dict(card) for card in cards_hist])]

        for i, post_dict in tqdm(enumerate(worskpace_data_hist), total=len(worskpace_data_hist)):
            post_detail(post_dict)

        return worskpace_data_hist

    # with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\GOOGLE_WORKSPACE_BLOG.json", "w") as f:
    #     json.dump(sorted(worskpace_data_hist, key = lambda x: x['published_date'], reverse=True), f)

# %%

def google_devs_map(link):
    r = requests.get(link)

    root = ET.fromstring(r.content)

    children = [get_link_date_from_map(child) for child in root]

    return children


def get_link_date_from_map(child):
    namespaces = {
        'ns0': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        # You don't need the 'html' namespace unless you plan to search for those tags.
    }

    loc_node = child.find('ns0:loc', namespaces)
    lastmod_node = child.find('ns0:lastmod', namespaces)

    link_url = loc_node.text if loc_node is not None else None
    mod_date = lastmod_node.text if lastmod_node is not None else None

    return {
        'link': link_url,
        'lastmod': mod_date
    }

def get_google_devs_full_map(GOOGLE_DEVS_SITEMAP):
    init_map = google_devs_map(GOOGLE_DEVS_SITEMAP)

    full_map = []

    for page in init_map:
        print(page['link'])
        full_map.extend(google_devs_map(page['link']))

    return full_map

def merge_list_left_join(list1, list2, join_key = 'link'):
    merged_list = []

    # 1. Create the lookup from list2 (same as before)
    lookup = {item[join_key]: item for item in list2}

    # 2. Iterate through list1 and use an if/else
    for item1 in list1:
        item2_match = lookup.get(item1[join_key])
        
        if item2_match:
            # Match found: merge them
            merged_list.append({**item1, **item2_match})
        else:
            # No match: just append the original item from list1
            merged_list.append(item1)

    return merged_list

def get_ai_summary_tags(gemini_api_key, content_list, content_type = 'article'):
    client = genai.Client(api_key=gemini_api_key)

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

    if content_type == 'video':
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

        response_1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents, 
        )
    else:

        response_1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt, 
        )

    # Close the sync client to release resources.
    client.close()

    try:
        result = json.loads(response_1.text.replace('```json', '')[:-3])
    except Exception as e:
        print("Error found in JSON transformation: ", e)
        result = response_1

    return result

# %%

GOOGLE_WORKSPACE_BLOG = "https://workspace.google.com/blog/"
GOOGLE_APPS_UPDATES = "https://feeds.feedburner.com/GoogleAppsUpdates"
GOOGLE_CLOUD_BLOG = "https://cloudblog.withgoogle.com/rss"
GOOGLE_RESEARCH_BLOG = "https://research.google/blog/rss/"
GOOGLE_BLOG = "https://blog.google/rss/"
GOOGLE_TECHNOLOGY_BLOG = "https://blog.google/technology/rss/"
GOOGLE_DEEPMIND_BLOG = "https://blog.google/technology/google-deepmind/rss/"
GOOGLE_DEVELOPERS_BLOG = "https://developers.googleblog.com/rss/"

GOOGLE_DEVS_SITEMAP = "https://developers.google.com/sitemap.xml"
# %%


GOOGLE_WORKSPACE_DATA = workspace_data_get(GOOGLE_WORKSPACE_BLOG)

# gen_ai_data = get_ai_summary_tags([i['link'] for i in worskpace_data_hist_full if not 'summary' in i.keys()][:100], content_type = 'article')
# worskpace_data_hist_full = merge_list_left_join(worskpace_data_hist_full, gen_ai_data)

# with open(r"C:\Users\Llubr\Desktop\Github\data-playground-data\GOOGLE_WORKSPACE_BLOG.json", "w") as f:
#     json.dump(sorted(worskpace_data_hist_full, key = lambda x: x['published_date'], reverse=True), f)
# %%


GOOGLE_APPS_DATA = fetch_and_parse_feed(GOOGLE_APPS_UPDATES, "GOOGLE_APPS_UPDATES")
GOOGLE_CLOUD_DATA = fetch_and_parse_feed(GOOGLE_CLOUD_BLOG, "GOOGLE_CLOUD_BLOG")
GOOGLE_RESEARCH_DATA = fetch_and_parse_feed(GOOGLE_RESEARCH_BLOG, "GOOGLE_RESEARCH_BLOG")
GOOGLE_DATA = fetch_and_parse_feed(GOOGLE_BLOG, "GOOGLE_BLOG")
GOOGLE_TECHNOLOGY_DATA = fetch_and_parse_feed(GOOGLE_TECHNOLOGY_BLOG, "GOOGLE_TECHNOLOGY_BLOG")
GOOGLE_DEEPMIND_DATA = fetch_and_parse_feed(GOOGLE_DEEPMIND_BLOG, "GOOGLE_DEEPMIND_BLOG")
GOOGLE_DEVELOPERS_DATA = fetch_and_parse_feed(GOOGLE_DEVELOPERS_BLOG, "GOOGLE_DEVELOPERS_BLOG")

GOOGLE_DEVS_FULLMAP = get_google_devs_full_map(GOOGLE_DEVS_SITEMAP)
# %%

flerlagetwins = fetch_and_parse_feed("https://www.flerlagetwins.com/feeds/posts/default", "TABLEAU_flerlagetwins")
vizwiz = fetch_and_parse_feed("https://www.vizwiz.com/feeds/posts/default", "TABLEAU_vizwiz")
storytellingwithdata = fetch_and_parse_feed("https://www.storytellingwithdata.com/blog?format=rss", "TABLEAU_storytellingwithdata")
playfairdata = fetch_and_parse_feed("https://playfairdata.com/feed/", "TABLEAU_playfairdata")
theinformationlab = fetch_and_parse_feed("https://www.theinformationlab.com/", "TABLEAU_theinformationlab")
# %%

YOUTUBE_GOOGLE_WORKSPACE = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCBmwzQnSoj9b6HzNmFrg_yw", "YOUTUBE_GOOGLE_WORKSPACE")
YOUTUBE_GOOGLE_DEVELOPERS = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UC_x5XG1OV2P6uZZ5FSM9Ttw", "YOUTUBE_GOOGLE_DEVELOPERS")
YOUTUBE_GOOGLE_CREATORS = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCXZNlYNefXV3iMcyzszt_9Q", "YOUTUBE_GOOGLE_CREATORS")
YOUTUBE_GOOGLE_DEEPMIND = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCP7jMXSY2xbc3KCAE0MHQ-A", "YOUTUBE_GOOGLE_DEEPMIND")
YOUTUBE_GOOGLE = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCK8sQmJBp8GCxrOtXWBpyEA", "YOUTUBE_GOOGLE")
YOUTUBE_GOOGLE_CLOUD = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCTMRxtyHoE3LPcrl-kT4AQQ", "YOUTUBE_GOOGLE_CLOUD")
YOUTUBE_GOOGLE_QUANTUMAI = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCO5cgpkcYjnsdxZdEDR3Jog", "YOUTUBE_GOOGLE_QUANTUMAI")
YOUTUBE_GOOGLE_TALKS = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCbmNph6atAoGfqLoCL_duAg", "YOUTUBE_GOOGLE_TALKS")

# %%

YOUTUBE_TABLEAU_FLERLAGE = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCDyr5VgVvkmfhHpeMUB8ZDA", "YOUTUBE_TABLEAU_FLERLAGE")
YOUTUBE_TABLEAU = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCWGrtxO6JrPSDUcgp3Qm_Gw", "YOUTUBE_TABLEAU")
YOUTUBE_TABLEAU_TIM = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UC7HYxRWmaNlJux-X7rNLZyw", "YOUTUBE_TABLEAU_TIM")
YOUTUBE_TABLEAU_VIZWIZ = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCTlX7UpqASrldmx5_CpG3CA", "YOUTUBE_TABLEAU_VIZWIZ")
YOUTUBE_TABLEAU_SQLBELLE = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCW2E1sGBVde5WMMxEh5CW4w", "YOUTUBE_TABLEAU_SQLBELLE")
YOUTUBE_TABLEAU_DATAFAM = fetch_and_parse_feed("https://www.youtube.com/feeds/videos.xml?channel_id=UCuDUG9ZHa-IlTm6y-2Lko_Q", "YOUTUBE_TABLEAU_DATAFAM")


# %%
