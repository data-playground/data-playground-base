# %%

import datetime
import json
import time
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai
from google.cloud import bigquery, secretmanager
from google.genai import types
from tqdm import tqdm

# %%

job_searches = [
    "Senior Analytics Engineer", "AI Solutions Architect", "Senior BI Engineer", "Senior Data Analyst", "Senior Data Scientist", "Senior Data Engineer", "Senior Machine Learning Engineer", "Senior AI Engineer", "Senior Analytics Manager",
    "Data Engineer (GCP), Solutions Engineer (Vertex AI), Analytics Architect",
    "Revenue Operations Engineer", "Product Data Scientist",
    "Senior Data Engineer AND (GCP OR BigQuery) AND Full-time", "(AI Architect OR Analytics Engineer) AND (Vertex OR GenAI) -Contract", "(Analytics Engineer OR Data Engineer) AND (Python AND SQL) AND Senior"
]

jobs = []


for search in job_searches:
    print(search)
    url = f"https://www.linkedin.com/jobs/search?keywords={urllib.parse.quote(search)}&location=New%20York%20City%20Metropolitan%20Area&geoId=90000070&f_SB2=5&f_TPR=&f_JT=F&position=1&pageNum=0"

    headers = {
        'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        # 'Cookie': 'bcookie="v=2&9363bb27-3613-47b3-88fc-eb41e753596f"; bscookie="v=1&202303242358160482442c-14ff-40ad-8d75-1e51d3f23f85AQGNOZkUlFsbHsNxwh0tbfyjZBJdFUsQ"; liap=true; JSESSIONID="ajax:1896546110745477250"; li_sugr=e37a32f2-0ada-49bd-8214-219c53b84ca1; li_theme=light; li_theme_set=app; _guid=ca4865f3-8663-459f-b723-a6c3c8c0af5d; _gcl_au=1.1.623995099.1749397231; dfpfpt=35d78705a3ab480987e299253926f0d7; li_at=AQEDARbC__wCHDkgAAABl1FXAm0AAAGY3dRmaE4AOs64RpCxby3g0qWG3D0K6SORbniNvY32CB9eGJaJgSB6H2WoQlw0MgqVsMfBke36AqxoKFpQSF_LXdojPaI4Cr-SYR2giQeuns8tT_BHsZFPNVwz; lang=v=2&lang=en-us; __cf_bm=iVyibkBFWRDQc.rtPmUFSydyvpb3HlzUvytl0HmFm9k-1755813871-1.0.1.1-BPbYcJu9NxF8MwTPd0whGWf68rgRE5uNKW0mAf6uolcmW8PUNA_rILeS9.aU907h1apTc4NdqBMZu8U9M9lxou2rk31UEKR.6D89qF4raFE; timezone=America/New_York; UserMatchHistory=AQJwH_KQrqDrUgAAAZjOqZ4Oq5AOI5XygGQfddJvA_-naraH5nRvD98SaGP2tW3YdXIvTzNU090z9gTMyTsCx_2_3ikkQiLvDDLytu1YpcRCQ9DQ2aUioIE9UeeXj829kRD0nFG00TePV_0R46zkWqLvyB8gIDdid-CkNPowaJPRDGNZN-hdFN9SM15g_L3JfvmJK5_GpbifvBf7Bvp1KWZ-_Qy-BuaROFOtC0JuU8WVgvtajCc6priPClVAKUwNXw5Hpvz-Y416H2QNgcBVSJakLX4BjFnMesS0REfCPSrDwcXcWfn4nEKaYe5aWk_zJR5MOneK4sWuuS2zUCumYQhMfTl985t7wnC-6Go6hd6V76ljaA; AnalyticsSyncHistory=AQI0gLZjcBZjfAAAAZjOqZ4OCcnzy...
    }

    r = requests.get(url, headers=headers)
    r

    soup = BeautifulSoup(r.content, 'html.parser')

    job_cards = soup.select('div.base-card')

    for job_card in job_cards:

        jobs.append(
            {
                "job_id": job_card.get("data-entity-urn").split(":")[-1],
                "job_title": job_card.find("h3", class_="base-search-card__title").text.strip(),
                "company_name": job_card.find("h4", class_="base-search-card__subtitle").text.strip(),
                "location": job_card.find("span", class_="job-search-card__location").text.strip(),
                "post_date": job_card.find("time", class_="job-search-card__listdate").get("datetime", "").strip() if job_card.find("time", class_="job-search-card__listdate") else None,
                "job_link": job_card.find("a", class_="base-card__full-link").get("href").strip(),
                "job_search": search,
                "search_date": datetime.date.today().strftime("%Y-%m-%d")
            }
        )
# %%

def deduplicate_jobs(raw_list):
    seen_ids = set()
    unique_jobs = []
    for job in raw_list:
        if job['job_id'] not in seen_ids:
            unique_jobs.append(job)
            seen_ids.add(job['job_id'])
    return unique_jobs

jobs = deduplicate_jobs(jobs)

for job in tqdm(jobs):
    r = requests.get(job['job_link'], headers=headers)

    soup = BeautifulSoup(r.content, 'html.parser')

    try:
        job["description"] = soup.find("div", class_="show-more-less-html__markup").get_text(separator="\n", strip=True)
    except:
        job["description"] = None

    try:
        job["salary"] = soup.find("div", class_="salary compensation__salary").text.strip() if soup.find("div", class_="salary compensation__salary") else soup.find("span", class_="salary-snippet").text.strip() if soup.find("span", class_="salary-snippet") else None
    except:
        job["salary"] = None

# %%

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


from collections import defaultdict
from itertools import zip_longest


def get_job_search_chunks(jobs, chunk_size=10, max_chunks=15):
    cat_dict = json.loads(pd.DataFrame(jobs).groupby('job_search')['job_id'].count().sort_values(ascending=False).to_json())
    
    chunks = []
    reservoir = []
    small_cat = False

    for key, val in cat_dict.items():
        if len(chunks) >= max_chunks:
            break
            
        # Get all jobs belonging to this specific search
        current_cat_jobs = [i for i in jobs if i['job_search'] == key]
        
        if val >= 10:
            # Take the top 10 for a pure chunk
            chunks.append(current_cat_jobs[:10])
            # Add the rest (11 onwards) to the reservoir for later use
            reservoir.extend(current_cat_jobs[10:])
            
        else:
            if not small_cat:
                small_cat = True
                # --- NICHE CATEGORY FILLING ---
                # 1. Group the reservoir to interleave it
                groups = defaultdict(list)
                for item in reservoir:
                    groups[item['job_search']].append(item)

                # 2. Interleave so we pull 1 from each 'big' category
                interleaved = zip_longest(*groups.values())
                reservoir = [item for group in interleaved for item in group if item is not None]

            # 3. Build the niche chunk
            chunk = current_cat_jobs # Take all (since it's < 10)
            items_needed = chunk_size - len(chunk)
            
            # Fill from the reservoir
            chunk.extend(reservoir[:items_needed])
            chunks.append(chunk)
            
            # 4. Update reservoir: Remove the items we just used
            reservoir = reservoir[items_needed:]

    # --- FINAL MOP UP ---
    # If we have space left, turn the remaining reservoir into full chunks
    while reservoir and len(chunks) < max_chunks:

        chunks.append(reservoir[:chunk_size])
        reservoir = reservoir[chunk_size:]

    return chunks
     
# %%

client = genai.Client(api_key=get_key("Gemini-API"))


def genai_response(client, prompt_str):
    """
        Run Gemini AI Job Scraper
    """

    # 2. Upload your files
    # resume_summary = client.files.upload(file=r"C:\Users\Llubr\Downloads\Pedro Mano Resume.pdf")
    # resume_full = client.files.upload(file=r"C:\Users\Llubr\Downloads\Full Resume.pdf")

    # 4. Send the prompt with the files
    system_instrucions = """
        ### ROLE
        You are an expert Career Agent and Recruitment Specialist. Your task is to analyze a batch of job descriptions and determine how well they match the provided Resume.

        ### CANDIDATE RESUME DATA
        ```markdown
            # Python

            ## Automation

            * **Amazon Vendor Central invoices download:** 

            Created a process to download invoices from Amazon Vendor Central that allowed the Program Management team to consolidate data for all promotional chargebacks Canon was liable to pay Amazon, saving them more than 6 hours a week of repetitive work that did not add value. The process downloads PDFs for every chargeback invoice and supporting Excel files for selected chargeback types. For the Excel files, the automation also cleaned and transformed the files in specific manners, also adding sheets with Pivot Tables on each file, making the analysis of the PM team much quicker and simpler.  
            The technical part of the process was performed in Python and involved opening a browser utilizing \`selenium\`, so the required login and two-factor authentication required to access Amazon Vendor Central could be performed. After that, a headless interaction, utilizing \`requests\`, with Vendor Central backend APIs, powered by the logged session’s cookies, performed the searches for required invoices and files download. Lastly, \`pandas\` cleaned the data and saved into Excel files and while \`openpyxl\` created pivot tables for each file.

            * **Ingram invoices download**

            Developed an automation process that downloaded files from Ingram’s portal (a distributor which some companies rely on to buy services and products). The files contained sales done through the portal and saved Program Management team more than 10 hours a month, between downloading and combining the files, joining to other data points and updating files for analysis and reporting  
            The technical steps of the process were done in Python and relied on \`requests\` to interact with Ingram’s APIs, downloading files into a shared drive location. Utilizing \`pandas\` to combine and treat the data, and \`openpyxl\` to style the sheets, the result Excel file was provided to Program Management

            * **Bank Statement Audit**

            Built an executable file (using Python) that allowed Finance teams to enter bank statements (as PDF) to be analyzed and transformed into a database. The process would compare values in different statements to ensure the transactions between banks were correct. The output of the process (emails and simple Excel database) were used as proof in Audit processes.  
            Backed by \`tkinter\`, I build a simple executable that presented a selection of Canon subsidiaries. Each subsidiary had its own predefined process to read PDFs (utilizing \`PDFminer\`), compare statement values, collate the results (\`pandas\` building simple comparison tables)  and ship a report through email (using \`win32com\` to send the email through the user defined within the computer).

            * **Email Auto-sorting**

            Created a process to analyze email content and correctly categorize email for ease of further analysis. This process compares data present in subject and body of emails in a shared mailbox to a list of customers to categorize financial emails. This saved teams focused on certain groups of customers from scrolling through emails that were not relevant to them  
            Supported by the use of \`win32com\` to read email from a specific folder within the user’s Outlook, the process compared information presented in each email body and subject to a predefined lookup table, then moving to the designed folder.

            ## Data Engineering

            * **Articles and Videos Extraction**

            Built a script to extract metadata for articles and videos that could be used in Content Recommenders throughout Canon USA’s website. The process gathered data such as title, author, links, thumbnails, summary and any tagging provided within the asset to guide the recommendations  
            The technical side of the process consisted of authorizing access to Adobe Experience Manager’s QueryBuilder API, which allows to query specific areas of the tool to gather a JSON-like response for the assets metadata. Then the data is loaded to BigQuery, so other processes can further clean for their own purposes. All of this is powered by Python and its \`requests\` and \`google-cloud\` packages

            * **AEM images**

            The environment where corporate, product and marketing images for Canon USA was not being well maintained, with little-to-no rich metadata to allow the users of the asset management tool to quickly find which images are available for specific situations. I built a process to extract every image from Adobe’s Digital Asset Manager. Then established the presence of duplicated images and marked duplicates for removal. Enriched images by prompting AI to provide more and better tags. And finally, built a web-service that, while the main area is fixed, served as an asset search tool to find images, powered by AI’s tags and descriptions  
            This process is built on top of Python, utilizing \`requests\` to interact with AEM’s QueryBuilder, \`PIL\` and hashing packages to define duplicated images, Google BigQuery to host the data for all the assets, Gemini acting as the tool to enrich the data about each asset and \`flask\` powering the web-service, allowing users to interact with all the above.

            * **Sony Alpha’s articles**

            As one of Canon’s main competitors, it was important to understand the Sony Alpha’s behavior related to content creation. I then created a tracker that gather publicly available information on their articles, providing background on content strategy for a large competitor  
            Technically, this process was very straight-forward, relying on Python’s packages such as \`requests\` to gather information from Sony’s API, \`google-bigquery\` to push data to BigQuery, where it was hosted and various packages to treat and enrich data contained in the articles. A Tableau dashboard to quickly highlight behavior, trends and typical topics related to the articles was also built

            * **B\&H product placement extraction**

            Scripted a daily process that gathers product placement on B\&H’s website. This allows business and product teams to compare how high Canon products are being placed in certain category pages to competitors’ products, as well as pricing teams to ensure B\&H was not undercutting prices, violating contract agreements.  
            The technical part of the process was built on top of \`requests\`, used to interact with the listing pages, \`beautifulsoup\` to extract data from specific areas on the pages, \`google-bigquery\` to host the data and a set of packages that allow Python to run the data extraction for many pages in parallel.

            * **B2B Analytics**

            Built a pipeline to extract customer psychographic and behavioral data from B2B email database. The B2B email tool used was Pardot (a Salesforce tool), but needed to be connected internal databases (sales and products) as well as external databases that enriched the B2B profiles  
            To build such a pipeline I utilized the REST API capabilities the tool make it available, using Python to make calls to the endpoints, saving the daily extract of the data to BigQuery. This set of tables powered attribution and recommendation models as well as dashboards to displayed the results of marketing actions.

            * **BI Internalization Project**

            Reverse Engineered an environment of pipelines that were built by a third-party provider. This environment was used to power a series of dashboards for business and marketing teams. Taking this environment internally saved about $1M annually in contracts with this third party, while also allowing us to better understand the needs of the business teams and a more personalized approach to analytics.  
            Powered by Apache Airflow (run in a Google Composer environment), this setup connected a large amount of data and tables in BigQuery, making the resulting data available in more than 20 Tableau dashboards ranging from email analytics, user psych/demographic analysis, sales and more.

            * **Web-Behavioral Attribution Engine**

            Built a pipeline that calculated 4 separate attribution models for Canon USA’s ecommerce. The idea behind building those four models was to analyze customer behavior prior to purchase through different formats. Basically the models were: GA4-similar attribution, GA UA-like attribution, last interaction (before purchase) and first session channel attribution.  
            Utilizing BigQuery (and Airflow for scheduling purposes), these attribution models are used in many different processes, including a number of Tableau dashboards to display how users reached Canon USA’s website and how different channels lead to sales (and other conversions)

            * **Product Management**

            PIM explanation

            ## Data Processing

            * **B2B Customer Data Enrichment**

            Created a pipeline between internal customer databases and an external tool (Dun & Bradstreet) used to enrich the customer data. The output of this pipeline provided information on businesses’ industry, which allowed marketing to improve the ongoing services to those businesses.  
            Initially using website interactions (through Selenium) then moving into an API approach, this process was built to load selected data points from BigQuery tables, enrich those records and load the “enriched” data back to BigQuery to power customer analysis.

            * **UTM Parameters Injection**

            Built a tool that facilitated the B2B team to add UTM parameters to their emails. B2B emails used to lack the parameters to track user journeys, which often made it complex to attribute revenue to that channel. To manually add the UTM parameters to the correct links within the email message proved convoluted and this process made such process simple and accurate.  
            Using JavaScript, this process gathers UTM parameters provided by internal user as well the email HTML template and, using knowledge or where to UTM parameters are needed, injects them in the email, making the process much quicker and more reliable then if done manually.

            * **Python Scripts Made Easy through Web App**

            Developed a website using a Python framework (Django) that allowed for a two-fold benefit. For non-technical teams, they could quickly run Python processes by themselves with a single click. For the data team, they could share the processes with business teams while keeping full control of the environment where those processes are run  
            Python allowed this web application to happen, creating a quick DRY (Don’t Repeat Yourself) environment for User, Groups and Scripts, providing an easy way to manage what is being being shared and who has access to each script. It also provided communication to the users by sending completion email through third-party tools SMTP.

            * **YouTube data extraction**

            Built a daily process to gather YouTube videos from Canon USA’s channel to power content recommender and generative AI processes in Canon’s personalization strategy. Later on, this process also started gathering competitor and creators data, powering market analysis as well as external success tracking.  
            This process was built to use Python to interact with YouTube API endpoints, extracting daily public data on videos, whether Canon owned them or not. This powered analytics (both for videos about Canon or competitors), recommenders and a dynamic video player page in Canon USA’s website

            ## Machine Learning

            * **3rd party vendors Voice of Customer**

            Developed a sentiment and categorization analysis for customer reviews for Canon products on large retailer’s websites. This process allowed product teams to better understand the voice of customer as well as sales teams to discuss with retailers what would entail the provider (Canon in this case) to reply to such reviews versus what would be the retailer's responsibility (such as delivery issues, payment problems…)  
            Through a range of Python packages (\`requests\` for data gathering, \`pandas\` for data wrangling and \`nltk\`, \`scipy\`and \`sklearn\` for text analysis), I created a Tableau dashboard that presented an overview of Voice of Customer for Canon products in retailer’s websites, allowing business and marketing teams to improve internal and external messaging and products based on how customers viewed them

            * **Content (Articles and Videos) Recommender**

            Built a content recommender combining content-to-content models (where the recommendations are done based on how close one content text is to other content texts) with user-to-content (in which user behavior is taken into consideration, showing which content other users have seen) to better serve content to users visiting Canon USA’s website.  
            Python-powered, this process uses text and user behavior analysis to rank which content pieces are better suited to each user. Packages such as \`sklearn\`, \`nltk\`, \`scipy\` and \`pandas\` are used in the process to build such models.

            * **Long-tail Site Searches**

            Analyzed in-site searches that provided more context than common keyword-based searches. Those are usually longer in nature and provide a better understanding of what the customer is looking for. They are used as sources by product, content and business teams to enhance content being developed for consumption in the website as well as on other platforms, such as marketing for social media and vendor websites.  
            This process checks those “uncommon” search queries by analyzing its text and potential intent. Packages such as \`sklearn\`, \`nltk\`, \`scipy\` and \`pandas\` are used in the process to build such models.

            ## AI

            * **AI Image Tagging**

            One of the main problems when working at a large scale company with a ton of assets (images, videos, documents…) that are distributed to end-users is how hard it is to find those assets some time after it is initially uploaded. Search capabilities on asset management tools make it easier, but only as much as the effort to add tags or metadata to each of those assets. At Canon, we encountered a situation where, after a couple tool migrations, assets were duplicated (some times more than 10 times) and had bare metadata (basically no tags and often generic titles), making impossible for a search tool to work well using this data.  
            I developed a pipeline that extracted every “active” image asset (those that have a public URL, so available to be served publicly), compared images to find duplicates (using hashing techniques and AI tools) then fed the images to enrich the metadata for them, adding data such as: contains product, fit for marketing and more generic tags about the content of the image (seasons, geography, holidays…)  
            The process was fully built in Python using \`requests\` to extract the data from AEM (using its querybuilder API), \`hashlib\` and \`ImageHash\` for hashing techniques and Gemini endpoints for both duplicate detection and AI enrichment.   
            The next step (in progress) is to build an UI (likely powered by Python, probably Flask) for users to interact with this clean and rich asset environment, instead of requiring a full rebuild of the current asset management tool

            * **Content and Email AI analyst**

            While Canon was going through its newest AI transformation, getting Google tools available to everyone in the company, I took upon myself to reach out to some teams that were common “customers” to understand how I could make their lives easier by utilizing the AI tools that were becoming available. The most heard response was that a specialized analyst would be the best addition. So, by combining existing pipelines and models, making the results available to our internal AI tools, and engineering agents that were specialized on each team’s data, I created analysts that knew the team, the data, the company as a whole and some of the *modus operandi*, to help guide the teams in their operations.  
            Gemini capabilities to create agents and GEMs were the main driver for the innovative part of this process, while more complex capabilities are being studied to use tools such as Google’s ADK (Agent Development Kit) to build highly custom agent environments for each team, instead of separate, siloed areas.

            * **Non-technical teams AI adoption support**

            With the rollout of an Enterprise version of Gemini and NotebookLM, combined with the migration to Google Workspace, I worked as a liaison between the technical side of understand the nuances of the tools and AI techniques with the non-technical teams, providing them a deeper understand of how to better utilize the tools, with prompt engineering, agent and GEM creations for their utilization.

            * **Content (Article and Social Media post) Starter**

            In 2023, in the height of the AI discovery, I built a Content starter Gen AI POC using Llama 2 8B to show how generative AI could support human content creators by providing a framework, to which those creators could add details and make unique. Later, made the model official by fine tuning through Vertex AI, using PaLM. The idea was brought back recently, with the great progress LLM have made, and a new version of the model, using Gemini 2.0 has been shared.

            * **Email Agent (before easy agent endpoints)**

            Developed a pipeline and a web application to allow marketing channel teams to create communications (email, search, social posts…) by using Gemini 2.0 multimodal model. This empowers channels with the capability to churn multiple versions of the communications, heavily personalizing the messaging based on the content and audience. It also increased productivity when creating messaging based on image and video, allowing teams to adjust the content rather than create from nothing.

            ## Apache Airflow

            * **Reporting Pipeline Rebuild**

            As the largest Airflow process for my team, I rebuilt a pipeline used to build tables that power dashboards for a variety of teams. This pipeline used to be managed by a third-party, a contract that cost the company more than one million dollars ($1,000,000+). By internalizing this pipeline, I effectively saved the company this value, while optimizing the workflow and its individual tasks, making the process simpler, quicker, cheaper and better results. Another improvement on taking control of this process was the turnover speed, where, by knowing the data and how it is used by multiple teams, I (followed by my full team) can provide the best solution in a much quicker timeline.

            # Business Intelligence 

            ## Tableau

            * **Tableau Server Migration**

            Performed the migration of the two largest Tableau Servers within Canon, managing the Cloud ecosystem in which they were placed as well as supporting each individually by being Site Administrator for both.  
            I performed content, user, group and permission migration for both servers, allowing their users to enjoy a stress-free migration, alongside all the perks of moving from Server to Cloud (no server machine management and always latest version)

            * **Tableau Server/Cloud Management**

            Managed a Tableau Server instance, ensuring the latest version was installed, checking if upgrades caused any issues with existing dashboards, maintaining user access correctly set and providing updates to creators on what new capabilities were included with new versions of the Server.  
            Later, managed two sites of Tableau Cloud, managing content and users on both sites, allowing users to be shared between sites and sharing best practice knowledge between all power users.

            * **Tableau Dashboards**

            Designed dozens of dashboards for teams like: marketing, finance, product management, audit, logistics, HR and even for the crisis committee during the pandemic. Some examples below:

            * Subscription Management: Three dashboards for Canon subscription plans (Pixma Print Plan and Auto-Replenishment Service for printers and Canon Profession Services for cameras), providing a full-picture of the subscription service in that dashboard, from enrollment and churning, to detailed information on the user and devices connected to the account.  
            * Return to Office analysis: A data representation of the all Canon USA’s offices, with each workspace (desk, cubicles and offices) mapped in a X-Y plane. The dashboard would provide the distribution of employees through the offices, allowing management to reorganize employee seating based on the government requirement of 6ft distance between employees.  
            * Channel Dashboard: An all-inclusive dashboard that provides all marketing channels the visibility of how they are influencing customer behavior on the website. This dashboard provides a high-level breakout on how a channel compares to others on selected KPIs, further breakout on specific channels to understand campaign level behavior, products sold by users coming through selected channels and more. The main differentiator when comparing this dashboard to out-of-the-box analytics tools is that it provides multiple internally discussed and built attribution models

            * **Tableau Extract Refresh**

            Created a tool that allowed team members to easily start an extract refresh in Tableau Server. This became especially important due to how Tableau Server was set up for the team. The server was set in a vendor’s environment, which required multiple levels of authentication, even once under Canon’s network. This tool allowed specific users to request an extract refresh to start through Tableau Rest API.

            * **Team Documentation Center**

            Built an all-inclusive documentation, learning and event-sharing tool in Tableau, allowing team members, specially non-technical personnel, to easily access all internal documentation, learning opportunities from multiple providers as well as an ongoing calendar with events that the team could benefit from attending. This dashboard data came from a scraping process I also designed that would constantly extract documents, articles, course and, calendars from selected websites

            ## SAS Business Objects

            * **Registration Dashboard**

            Monthly report to compare the number of products registered (which allows products to be serviced when sold by third-party vendors) to products sold by Canon to vendors. This provides deeper understanding of LifeTime Value (LTV) for customers that purchase their product through vendors while also giving a high-level understanding on how much customers from third-party vendors interact with Canon systems, by registering their products.

            * **Vendor Chargeback**

            Gathers quantity and sales numbers for products sold by third-party vendors while they are in promotion. These numbers are used to provide proof to vendors of sales made during the promotional period and how much Canon has to pay back to the vendors as part of the chargeback agreement.  
            Later, this report was duplicated and slightly changed to give the Audit Department a full picture of the chargeback environment, providing a report that could be easily filtered by vendor, product and/or product lines, to ensure contract agreements were being held true at the time of chargeback payments.

            # SQL 

            ## Microsoft SQL Server

            * Managed team’s SQL Server, ensuring access and disk size was constantly reviewed and adjusted based on necessity  
            * Created processes that connected Excel to MSSQL, allowing non-technical personnel to perform complex queries only by filling a form. In one of the processes it saved the team more than 3 hours weekly on Invoice selection for risk assessment, changing from manual selection to rule-based selection

            ## BigQuery

            * Utilized BigQuery as the main source of data, creating many pipelines that would insert and/or extract data from BigQuery for diverse purposes.  
            * Optimized the usage of BigQuery by adjust many pipelines to include best practices as partitioning and clustering  
            * Built pipelines for Google Analytics, order tables, CRM tools, social media tools and more, which powered many reports and dashboards.

            # Google Cloud Tools

            ## Cloud Run Functions

            * Created full-fledge APIs and added endpoint to existing internal APIs  
            * Built processes to extract, clean and enrich data using Cloud Run Functions capabilities

            ## Vertex AI

            * Scheduled processes utilizing Vertex AI’s notebooks to gather data, especially in situations a more powerful machine was needed.  
            * Built AI agents directly in the UI and utilizing Agent Development Kit (ADK), making available for internal users through Vertex AI

            # HTML/CSS/Javascript

            ## Adobe Target

            * **MyCanon Insider Experience**

            Supported the development of a personalized home page for logged in users for Canon USA. Once users login they encounter a page with content specially selected for them, from recommended products, articles and videos, to best deals for them, all the way to latest software updates on their owned products, this page provided a one-stop location for everything Canon.

            * **Content Recommender Placement Test**

            Built experience to display a content recommender at Canon USA’s home page. An A/B/n test was built to show that higher placement and better design would lead to more engagement. Higher placement showed to be statistically significantly better, while design differences allowed more conversations to happen.

            * **Video Player Page**

            Developed a dynamic YouTube video player page, that combined with a Google Cloud Function, facilitated the access to videos within Canon USA’s website. This allowed marketing teams to drive more sessions to the website rather than YouTube, while maintaining YouTube statistics. The idea to lead users to the website allows for further personalization as more data is gathered.

            * **Holiday Gift Guides**

            Created a dynamic product guide that displayed selected “in stock” products for specific categories with a personalized description for each category. This guide also had an A/B test for design testing.

            * **Shareable Wishlist**

            Developed a page to be used as a shareable “favorites” page, allowing users to see products currently in their favorites selection and share this list through many channels (emails, facebook, twitter or copy the link). 

            ## Google Tag Manager

            * GA4 setup  
            * Page load time  
            * Interaction with Firestore
        ```

        ### KEY STRENGTHS TO WEIGHT HEAVILY:
        - GCP Ecosystem (BigQuery, Vertex AI, Gemini 1.5 Pro).
        - Python Automation (Selenium, Django, Airflow).
        - Business Impact ($1M cost savings, A/B testing optimization).
        - Background: MBA in Business Analytics + Industrial Engineering.

        ### TASK
        1. ANALYZE THE PROVIDED JOB DESCRIPTION: 
           - Conduct a deep dive into the specific technical requirements and business responsibilities listed at this source.
        2. VERIFICATION:
           - Verify the technical stack (Python/GCP/SQL) and the seniority level.
        2. FIT SCORING:
           - Assign a fit score out of 100 based on how well the job matches the candidate's specific achievements and skills, based on:
              - Technical Skill overlap (60%)
              - Seniority/Level alignment (20%)
              - Industry/Domain experience (20%)
        3. QUALIFICATION ANALYSIS:
           - Map the candidate's specific achievements (e.g., the $1M savings or Vertex AI experience) directly to the JD's "Requirements" or "Preferred Qualifications" sections.
        4. GAP IDENTIFICATION:
           - Explicitly list any tools, frameworks, or certifications mentioned in the JD that are missing or weak in the provided resumes.
           
        ### QUALIFICATION CRITERIA
        Evaluate the candidate's fit based on:
        - Technical Match: Python/SQL/GCP/Airflow/Vertex AI.
        - Impact Match: Proven $1M annual cost savings via automation.
        - Specialized Fit: Experience with 'Agent Development Kits' and A/B Testing.

        ### CONSTRAINTS:
        - Return ONLY a JSON array of objects.
        - Do not include any conversational text or markdown outside the JSON.
        - If a job is completely irrelevant, give it a low score (<30).
    """

    prompt = prompt_str

    job_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "remote": {"type": "BOOLEAN"},
                "explanation": {"type": "STRING"},
                "fit_score": {"type": "INTEGER"},
                "qualification_analysis": {"type": "STRING"},
                "skill_gaps": {"type": "STRING"}
            },
            "required": ["remote", "explanation", "fit_score", "qualification_analysis", "skill_gaps"]
        }
    }

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instrucions,
            response_mime_type="application/json",
            response_schema=job_schema,
            temperature=0.2
        )
    )

    # Close the sync client to release resources.
    # client.close()

    # Attempt to parse the response text as JSON
    job_data = json.loads(response.text)

    return job_data

# %%

chunks = get_job_search_chunks(jobs)

all_results = []
for i, chunk in enumerate(chunks):
        print(f"Processing Batch {i+1}...")
        
        # Format the chunk into a string
        batch_content = "\n\n\n".join([
            f"ID: {j['job_id']}\nTitle: {j['job_title']}\nDesc: {j['description']}\n\n\n---" 
            for j in chunk
        ])

        try:
            result = genai_response(client, batch_content)
            all_results.extend(result)
        except Exception as e:
            if "429" in str(e):
                print("Quota hit! Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                raise e

# %%

for job in tqdm(jobs):
    completed = False
    while not completed:
        try:
            # 1. Run your analysis function
            result = genai_response(client, job['description'])
            job['remote'] = result[0]['remote']
            job['explanation'] = result[0]['explanation']
            job['fit_score'] = result[0]['fit_score']
            job['qualification_analysis'] = result[0]['qualification_analysis']
            job['skill_gaps'] = result[0]['skill_gaps']
            
            # 2. THE PACE: On Free Tier (10 RPM), wait 6 seconds between jobs
            # This keeps you safe even if the API processing is instant.
            print(f"Processed {job['job_id']}. Waiting 7 seconds...")
            time.sleep(7) 
            completed = True
        except Exception as e:
            if "429" in str(e):
                print("Quota hit! Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                raise e


# %%

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



# %%


# %%

from selenium import webdriver

# %%

url = "https://www.sonara.ai/dashboard"

driver = webdriver.Chrome()

# %%

driver.get(url)

driver.find_element('xpath', '//input[@name="email"]').send_keys("pedrohmanodemoura@gmail.com")
driver.find_element('xpath', '//input[@name="password"]').send_keys("j8gS4tL4@jz@Sw6")
driver.find_element('id', 'login-btn').click()

# %%

js_string = """
url = "https://www.sonara.ai/ea/api/v1/expert-apply/profiles/2954c26c-9895-b742-fc80-5592e09a47a2/jobs/suggested?page=1&size=100000&suggestedJobsType=Both&IncludeNonExpertJobs=false&datePosted=Any%20time"

headers = {
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'en-US,en;q=0.9,pt;q=0.8',
    'Cookies': document.cookie,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
}

return fetch(url, {
    method: 'GET',
    headers: headers
})
    .then(response => response.json())
    .then(data => data);
"""
# %%

results = driver.execute_script(js_string)
# %%

### Built in NYC

# %%

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
    # "Cookie": "_ga=GA1.1.1848700245.1749417744; BIX_AUTH=chunks-2; BIX_AUTHC1=CfDJ8N49pBgB_XBPsL35S65RW-dGOlQuSIn-x8sCXNwyD-Y16XwrJSvVEXZhfBPOK-CJoX-bVOmXAlsH5sHO_JxKcvTogAT9hD8QhSGa6BWgPtG280k7rSyMijGF6m-7z7d1COvbibtEJgmRO9y_cLwFCJbHJtm6TiHLyIJ9H1nsNH4kNbXC-uYQF4iNT2khN5iM9F_cJJxXdv3R1_yHWWzv1dxI1BSYRGdBihphNYpOwiahKPZds96Aaym0YrtvBQG0H7pnJrnEg_w4u-vao6UdLfLV-7aCzRMUU5cS-xbPtr0B8D5aKh1yj73SENReQP7Q37WAgysA8Jg8CXy9yT7BcJtF5-VSEv4jfXlvRwK2aQq4h63n-1vka-HhIabM-W4zChD23iG_zgyGPWWVV9rvofOeOrH7-kCKxS7iHKQC-2IsdH_hIMgdDIov0xvJP47VibYcigRDQIJxpPKnt78UlhNh29PnpcuJ1u2qCmmpIjZr-ISd_KpdJ9Wg0aG312uhO7r2eGLiinC1Mgy4uWv-pdpQX3pUopkpCM0JSnfXBjGdISTAYTZjvYHVQ17vyfZpL4Lu7GTDAd0iMZGEN7e9O-BOmYBw2cZa128G3Koqs_52LdcMebKQHiqElrOGWYlNaFWUphOlJyqYFaj9aNs4d_fbkbSu-787Ri7Qc4KxN4WL0P_61U744eZtp0W-RhNYSOdaWGTVC9kIXAN-u0eMHNg38DOy6GZQ4o9TBFLjkazYFUBMyjUYK3G489MDvQwiSomqc-UOGOkQNVC0Ptmn1dYN4CgIblo5LM90Z0fXhml2qYb68QTyGlbxM5mH-gMiswkfi8hGBsUbp8gJQo3OyYR7JGs9EHbH9QZEwIVfIq7XiWSBR4aDqsxLW5K-DnQ6ubuOsmOJadeVA9SUXFsuCQ_RSYNyF0dIE1xe2h20f5VqjhVDs0f_hUssYaQARp2tY-JmPHTVyrbaP-4GZctOE53KF7qXuoBlKaRv7B9CJfVrh8KcGTcslXCXhiUiEIgMEE8lL193m4GL0R3D7VsKyBT4XWmMEKjOlPYwZIHKVPEFTZsoaBvpg7wQ7cStQD-z7x0SH4dar3FPoxhbMQr1Ap6fIojqsG3SPfmEO_ZQ_XU9bjPkZbERNWq4nU3CwwfAsdC0U2ZFYkfZQHMRux6sduSLtdZdDfRLwG1GaGml_9STzNmhWn4_7NToifQ88jKL_ALMkTKQLmxBKHg8hOP0X01cYVQf2TWbLIRt1z-8q1cMnO-8V1fMy1aTtzB0cd1UTyYmoeDiCGSX4T14g_SXGczmvJhQqNBaRJ7-R61hPpEIEf92uTtUeDE1WJAu5P4-wkCTQOGt-muBdCmlSezc6LaG0Pytxn5V4JYpuTowAoH2-pxoTAGyxfV-543sMfTpGUUWNenHesYkIQZ-O2PN9TRgPG_I7qets10NB4nhoCF--B4GB3lvzN8XgidJoJUt76lbrgHCLPWhyOa5YkI5uUTicnzIlCUjS5UiojiET6WzhCOsG1hQjmlKCvdKAOCCLwdkcBs_6WP86RG2p8FD0nLBC3bMHArWx28jnk1c3FxoxgLgIc9jYc-Ld1pTxKbXGlcg3_L5clFE9Zm7d0SCVJ3gONCYAP0fR6M6FIz1wAtY-muQftW5b3h8V-eYD2AW1R6cvinqDmN5Yb-htF_OEk8RKB2LKe7M5dJd2zZhwHv6EFcQ8aMaxOaxATPCL4InDtc8uBD_yF3Ctq9UqRWq6CbtFc3ch-1ate4UTKS-dXJInCtzTQCBkBmcWyr2CwXV9siYIbTbvwqA79M76MQuyJ8kZWr6OsSkKaxOKw4t1ZCMYOAfK1WUe-pWq2Y3r41kV8I9-OJoDh-8VVdFkQxUWlvF1hQT9cX1XR4x7zI2ToZsC5L5hgb2XhK9z3E23gr9q0s7KJTftyIVZjog4oufan7Sle1Tj7NG3wXVVZcRD50TzqKgz2gawtQN7nQ2lVkw3Ht1CIP6cxjDRWz2f7qWpH3HVwK_WmfMrVtA-nlg4_uL5AAP7DhrFu5slnGcVGQXV-ODA-ZZbYXEmVp0RpULCwp7XgyzMyV1jJ2tWsdV3H5YXDRj3vsOxoak3OxID-RxikkrpMy5ctZzvfzEKC-wMGdzO3a1pzQOBoiKrfZpeUBpSanhUEbOmY97GRpG-mp1ynXQnvw1nAFYoN_LLoLHPWyOTk4O5IniliM_lTkRLXJWvLUa0Oz2lpXfibULTv05Mdck9_6mp4unt9N9Ujpx46jnaPQoErdLzG1ep_NBXbrYk_DsmiNZtYfiK6hhA3iAy2W8bzn0-MUeRNq7c-ala923WgtIYaNHxuoBegB7J5aAwpasW_wWt333Z5HGCLJfKKXBKAFIbukWTxaBrk6WEvM0NY4gaygRDuR8leVlw8akOvPbQiCTyHM6ShhMKZK5qRfPxTABwr23qYH3fY19hMAbEHRAr2dzXIj33AtvIKCgGWgckjbb0Y-NXXbz3eBDjGrzW8sP1-fIvaAIUNMj91ObwxlG2ZG5lfTusUWVaw_65nUHrxGUVv6kkKKM0Wh2DkgJBlQtOPsn069gtimk5dIja0Lt-N7GVVzIo5NBI_t5tDJKPQjT6YbChkEusJ0F-52Eysrk9AsUtqLfmTma3xVEOkAarv0wX2LC8BdLqj1DYEoAWfkwjh2j-x9JXjsFKPGMZwX-UH33s92n_et3hH4FEynR_Q8u50LMH5Sz3iEY5xjtP_HBJnm6BgsQJTGRaE1PBOwUMef-LHktYHZNQ9BQcljOWcuu3qFQf6TjY8reXsPt0CZk5DLr5bHnSsvVa9GFwKznOj74gWBcWaF8PsCf0cXcB5DG-VVqOA7Yl6rkfWRZWWe3p6vikHchsvO8AKZby9etiR4WklEAGiR-g1XY0A_dRLhxKtw5f9msrmgEWd0aQsS1jSF0A3luQKhZE06UeBvZzctBaf3HIJvqIoM2GDzGI0A1fTetc9l70v8v_Zpq5eLoGBicXXW7iwXVkxI3W9Xq1uDi5BolWIPxSVr-zhKAm19YOOTjOQEB4maM65jbOgktNgi5FxlDN6ANN4b-hxFNCqrtWiq4ecTbQ6lQ6cdqTMqRSn4P1S_Dn6pipSInKq0_H7EJqB_7xG66BO_pJuTWhNL1fN3JK3F6_Xhgpw81U0H7WrQD9iEpaT1J4oquBKIUB4012iw3PRwQFg-Gf7m4IlrUdd1JnrqDqgvJ6opFPnSsZeVZF0plYxagUpCh-JdxbTrBaD7IHdkFw1V1lyB2ZFxOcDzrLs4yKPAYPxxI0krfF5yD0JPpV9oD6Ioy1C-4d_ULWi4JDPXl-kHwYba-P4p43wf8g7k7n5-sz9nDF175EQ7p4P6xMLGU66VPkenR_hwOy8-rFwXqJ9p4GcQeZwCMcZD25rfU02s51zfbMYTEBHJlDRmDGUePMYws6nZchxAdtFC27iYobyoaUlB9_Jx6NXjoqwvr_7e-ZgKEHlKAL03zql2nrt4n2Yf36vw0ipd4ykwf8UYXCyFTYLYxjmCuOq3mm4e9ppqUhLp5SXQMeni1gW5OqKQR_Y7UdgpXHpj5pxwpLdQ_qWFnnSOFTdvGSVE2-ZHKb4mJo0vXxcvcYaW34MKoaKEzGXWPhUKdga7R-guKThZ1_860d4B4v_X3DF_HhxB-oGXriDpp2OfHd61napMFKa6EYWw-4hXESL5UijXskYbFvN9JvfHexase3uldaT5dLikMD6DAqbGkYJzBhXH8wJer_5D9T84T5eyhRzs4O7CDms5_VX3nYv-CzZ5y_XxFWxlI9NE0RypItj4QRzgRij7s8ngwGzYSbzdqe_ns6HTc3V-n7VJg1rTgZO9zz33q8CcPyF5gY0OZfPVJUJCYXnrEPWdb0V_BwLxIPKB9LeeueleEk06hk_IR_AxDZ-C9KCN7rjOORas_42nYau6tLNoIJrVgu3M; BIX_AUTHC2=coYrA6ekdQKXXmd1RoryiHo9ERu5Zdxd1833aexZWgHWPysFQpP06U63WdZF7Yl3s36W2y8Sc7NUPr3VXgaDdD9ec16D6yHgBfb3RAeTdFOcZ0hVo30x-lEY6ZefUeXbzeF5XI4_7nYFscarJ6Pmorl3iNL0TNym2KtjAO1KmK3cLN7nYTh7lnQL6GuH9vtXBL7lI0vFBpbCXasMyaITP7eokX0vct8uo7AwCaxZqYPronBHH0RDAgONh6afjjSlaujDCp8pR3r1jFzYot596krAgzqidcuFQbnniPeg4kxzndPApZOMGxp_5uUq2KCDXB_qUboAwffv8QkF1AVU5rOBLUqKxOvkWp9NrkITe5CpEbet_EYV_Ta4QOt8l0nP2UlwJRDUuCOwKIATECdBPblieHpQAinxp7lppZFld7S35_knW9JB_rchExBfumU3ZHnfVKxfmRZlftyGA58Bs0QRMjuFHGQpnndgOSuNmpPN6rnYgEMTOcUcTqXvLIL76gV5egt6GWqVQ6EKKTQPIwXJmUrM5KKrL4vDk5p68rkVidyuZZjg1td-fv185gw2UziJAE7kKxgQJMvQkHDbZY1AFQwBmTzDdEV60zwpgV9fLb9EGypJGY4vp76DRbaMCF4_qn_w186ernQE3OVGLItynxIy3SSeUXgJTjo-yCIQ5qnH0nyPFy-kBwi6EdN-3Nk1NZ-ejkC6-UA; clientId=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjIwMTc4MGU4LTQ3NTctNDFiZC1iOTNiLWQ0YjQwZDY1Yzg0YSJ9.P4u59dbPhSnvpjwLCdXCK8IhvamjgDlrzP48Ojfhfmc; _ga_PD9P2L7TK7=GS2.1.s1756656227$o2$g1$t1756656411$j59$l0$h0; OptanonConsent=isGpcEnabled=0&datestamp=Sun+Aug+31+2025+12%3A06%3A51+GMT-0400+(Eastern+Daylight+Time)&version=202503.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0004%3A1%2CC0002%3A1&AwaitingReconsent=false",
    "Referer": "https://www.builtinnyc.com/",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}
# %%

def get_jobs(page):
    url = f"https://www.builtinnyc.com/jobs/data-analytics/senior/expert-leader?skills=Python,Excel,SQL,AI,GCP,JavaScript,Tableau,MS%20Office&city=New%20York%20City&state=New%20York&country=USA&allLocations=true&page={page}"

    r = requests.get(url, headers=headers)

    soup = BeautifulSoup(r.content, 'html.parser')

    jobs = soup.find_all("div", attrs={"data-id": "job-card"})

    job_details = [get_details(job) for job in jobs]

    num_pages = soup.select('.pagination > li')[-2].text

    return job_details, int(num_pages)

# %%

def get_details(job):
    job_details = {
        "company_title": "",
        "job_title": "",
        "job_link": "",
        "reposted": None,
        "remote_option": None,
        "location": None,
        "easy_apply": False,
        "salary": None,
        "level": None,
        "categories": None,
        "summary": None,
        "skills": None
    }

    job_details["company_title"] = job.find("a", attrs={"data-id": "company-title"}).text.strip()
    job_details["job_title"] = job.find("a", attrs={"data-id": "job-card-title"}).text.strip()
    job_details["job_link"] = "https://www.builtinnyc.com" + job.find("a", attrs={"data-id": "job-card-title"}).get("href").strip()

    attributes = job.select(".bounded-attribute-section > div > div > div")

    for attr in attributes:
        if attr.find(class_="fa-clock"):
            job_details["reposted"] = attr.find(class_="fa-clock").parent.text.strip()
        elif attr.find(class_="fa-house-building"):
            job_details["remote_option"] = attr.text.strip()
        elif attr.find(class_="fa-location-dot"):
            if attr.find("span", attrs = {'data-bs-toggle': "tooltip"}):
                job_details["location"] = " • ".join([i.text for i in BeautifulSoup(attr.find("span", attrs = {'data-bs-toggle': "tooltip"}).get('data-bs-title'), 'html.parser').find_all('div')])
            else:
                job_details["location"] = attr.text.strip()
        elif attr.find(class_="fa-bolt"):
            job_details["easy_apply"] = True
        elif attr.find(class_="fa-sack-dollar"):
            job_details["salary"] = attr.text.strip()
        elif attr.find(class_="fa-trophy"):
            job_details["level"] = attr.text.strip()

    job_details['categories'] = job.find(class_ = "mb-md fs-xs fw-bold").text.strip()
    job_details['summary'] = job.find(class_ = "fs-sm fw-regular mb-md text-gray-04").text.strip()
    job_details['skills'] = " • ".join([i.text for i in job.select(".d-md-inline.ps-md-sm > .fs-xs.text-gray-04.mx-sm")])

    return job_details

def get_job_description(s, url):
    r = s.get(url, headers=headers)

    soup = BeautifulSoup(r.content, 'html.parser')

    description_body = soup.find("div", class_ = "html-parsed-content")

    return "\n\n".join([i.get_text() for i in description_body.children])

# %%

job_details = []
num_pages = 1000
page = 1

while page <= num_pages:
    print(f"Page: {page}")
    
    page_results, num_pages = get_jobs(page)

    job_details.extend(page_results)

    page += 1

# %%

# for job in tqdm(job_details):
#     s = requests.Session()

#     try:
#         job['description'] = get_job_description(s, job['job_link'])
#     except Exception as e:
#         print(f"Error: {e}")
#         job_details['description'] = None
# %%
