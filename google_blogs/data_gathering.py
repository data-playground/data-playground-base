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

# %%

def fetch_and_parse_feed(url, site_name):
    """Fetches and parses the RSS feed."""
    print(f"Fetching feed from: {url}")
    feed = feedparser.parse(url)

    file_name = f"C:\\Users\\Llubr\\Desktop\\Github\\data-playground-base\\google_blogs\\{site_name}.json"

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

        post_data_full.extend(post_list)
        post_data_full = sorted(post_data_full, key = lambda x: x['published_date'], reverse=True)
    else:
        post_data_full = sorted(post_list, key = lambda x: x['published_date'], reverse=True)

    with open(file_name, "w") as f:    
        json.dump(post_data_full, f, indent= 4)    

    return post_data_full

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

    for i, post_dict in tqdm(enumerate(worskpace_data), total=len(worskpace_data)):
        post_detail(post_dict)

    with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\worskpace_data.json", "r") as f:
        worskpace_data_hist_full = json.load(f)

    worskpace_data_hist_full.extend(worskpace_data)
    worskpace_data_hist_full = sorted(worskpace_data_hist_full, key = lambda x: x['published_date'], reverse=True)
    
    with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\worskpace_data.json", "w") as f:    
        json.dump(worskpace_data_hist_full, f, indent= 4)

    return worskpace_data_hist_full

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

    # with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\worskpace_data.json", "w") as f:
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

GOOGLE_APPS_DATA = fetch_and_parse_feed(GOOGLE_APPS_UPDATES, "GOOGLE_APPS_UPDATES")
GOOGLE_CLOUD_DATA = fetch_and_parse_feed(GOOGLE_CLOUD_BLOG, "GOOGLE_CLOUD_BLOG")
GOOGLE_RESEARCH_DATA = fetch_and_parse_feed(GOOGLE_RESEARCH_BLOG, "GOOGLE_RESEARCH_BLOG")
GOOGLE_DATA = fetch_and_parse_feed(GOOGLE_BLOG, "GOOGLE_BLOG")
GOOGLE_TECHNOLOGY_DATA = fetch_and_parse_feed(GOOGLE_TECHNOLOGY_BLOG, "GOOGLE_TECHNOLOGY_BLOG")
GOOGLE_DEEPMIND_DATA = fetch_and_parse_feed(GOOGLE_DEEPMIND_BLOG, "GOOGLE_DEEPMIND_BLOG")
GOOGLE_DEVELOPERS_DATA = fetch_and_parse_feed(GOOGLE_DEVELOPERS_BLOG, "GOOGLE_DEVELOPERS_BLOG")

GOOGLE_DEVS_FULLMAP = get_google_devs_full_map(GOOGLE_DEVS_SITEMAP)
# %%

