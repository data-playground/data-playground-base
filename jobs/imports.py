import base64
import datetime
import json
import time
import urllib.parse
from collections import defaultdict
from itertools import zip_longest

import mysql.connector
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai
from google.cloud import bigquery, secretmanager
from google.genai import types
from tqdm import tqdm