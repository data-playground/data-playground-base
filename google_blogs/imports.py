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