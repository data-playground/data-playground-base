# %%

import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from tqdm import tqdm 
import json

# %%

def fetch_and_parse_feed(url):
    """Fetches and parses the RSS feed."""
    print(f"Fetching feed from: {url}")
    feed = feedparser.parse(url)

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
            "title": entry.title,
            "link": entry.link,
            "published": published_date,
            "summary": entry.summary[:200] + "..." # Get the first 200 characters of the summary
        }
        post_list.append(post_info)

    return post_list

# %%

def workspace_data(card):
    card_a = card.find_parent('a')
    author_read = card_a.select_one('p').text if card_a.select_one('p') else card_a.parent.parent.select('p')[2].text

    return {
        'link': card_a.get('href'),
        'title': card_a.select_one('h3').text if card_a.select_one('h3') else card_a.parent.parent.select_one('h4').text, #card_a.get('data-g-action'),
        'thumbnail': card_a.select_one('img').get('src'),
        'author': re.match('By (.*) • (.*) read', author_read).group(1),
        'read_time': re.match('By (.*) • (.*) read', author_read).group(1),
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
    post['html'] = clean_html(soup.select_one('c-wiz'))

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
# %%

def workspace_data(GOOGLE_WORKSPACE_BLOG):
    r = requests.get(GOOGLE_WORKSPACE_BLOG)

    soup = BeautifulSoup(r.content)

    cards = soup.select('.PBkdHd')

    worskpace_data = [workspace_data(card) for card in cards]

    for i, post_dict in tqdm(enumerate(worskpace_data), total=len(worskpace_data)):
        post_detail(post_dict)

# %%

def workspace_hist(GOOGLE_WORKSPACE_BLOG):
    driver = webdriver.Chrome()

    driver.get(GOOGLE_WORKSPACE_BLOG)

    get_user_confirmation()

    soup_hist = BeautifulSoup(driver.page_source)

    cards_hist = soup_hist.select('.PBkdHd')

    worskpace_data_hist = [dict(t) for t in set(tuple(d.items()) for d in [workspace_data(card) for card in cards_hist])]

    for i, post_dict in tqdm(enumerate(worskpace_data_hist), total=len(worskpace_data_hist)):
        post_detail(post_dict)

    with open(r"C:\Users\Llubr\Desktop\Github\data-playground-base\google_blogs\worskpace_data.json", "w") as f:
        json.dump(worskpace_data_hist, f)

# %%

GOOGLE_WORKSPACE_BLOG = "https://workspace.google.com/blog/"
GOOGLE_APPS_UPDATES = "https://feeds.feedburner.com/GoogleAppsUpdates"
GOOGLE_CLOUD_BLOG = "https://cloudblog.withgoogle.com/rss"
GOOGLE_RESEARCH_BLOG = "https://research.google/blog/rss/"
GOOGLE_BLOG = "https://blog.google/rss/"
GOOGLE_TECHNOLOGY_BLOG = "https://blog.google/technology/rss/"
GOOGLE_DEEPMIND_BLOG = "https://blog.google/technology/google-deepmind/rss/"