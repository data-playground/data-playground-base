import itertools
import json
import operator
import time
from datetime import date, datetime, timedelta

import lxml.html as LH
import pandas as pd
import requests
from google.cloud import bigquery