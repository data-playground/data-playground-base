# airflow/agents/job_resume_context.py
"""
Resume context for the Job Scout fit-scoring agent.

Kept in its own file (rather than inline in job_agents.py or the DAG) for
two reasons:
  1. It's the one part of this pipeline you'll actually edit over time —
     as your resume changes, you only touch this file.
  2. It keeps job_agents.py and the DAG readable — neither should scroll
     past a page of resume markdown to find the actual logic.

HOW TO FILL THIS IN:
  Copy RESUME_MARKDOWN and KEY_STRENGTHS_TO_WEIGHT exactly from the
  `system_instrucions` string in your original LinkedInJobScraper script —
  the big fenced ```markdown block is RESUME_MARKDOWN, and the
  "KEY STRENGTHS TO WEIGHT HEAVILY" bullet list is KEY_STRENGTHS_TO_WEIGHT.
  Nothing needs to change about the content itself.
"""

# Paste the full resume markdown block from your original script here,
# unchanged (the big multi-page ```markdown ... ``` section under
# "### CANDIDATE RESUME DATA").
RESUME_MARKDOWN = """
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
"""

# Paste the "KEY STRENGTHS TO WEIGHT HEAVILY" bullet list here, unchanged.
KEY_STRENGTHS_TO_WEIGHT = """
- GCP Ecosystem (BigQuery, Vertex AI, Gemini 1.5 Pro).
- Python Automation (Selenium, Django, Airflow).
- Business Impact ($1M cost savings, A/B testing optimization).
- Background: MBA in Business Analytics + Industrial Engineering.
"""
