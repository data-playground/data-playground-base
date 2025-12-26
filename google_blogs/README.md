# Google Blog Data Aggregation & AI Enrichment Pipeline

## Portfolio Context
This project, `google_blogs`, is a core component of a larger data-playground initiative focused on extracting, processing, and enriching publicly available data. It specifically targets various Google blogs to collect valuable insights, making it suitable for content analysis, trend tracking, or building custom news feeds. The design prioritizes modularity and ease of integration into orchestrated data pipelines, such as those built with Apache Airflow.

## Difficulty Level
**Advanced**

This project demonstrates advanced capabilities in web scraping, API integration, data engineering, and secure credential management. It leverages headless browser automation (Selenium) for historical data, integrates with Google Secret Manager for secure API key handling, and utilizes Google's Gemini API for sophisticated AI-driven content enrichment, including summarization and tag generation. The architecture is also designed for production deployment within environments like Airflow.

## Tools Used

*   **Core Language**: Python
*   **Web Scraping & Parsing**:
    *   `requests`
    *   `BeautifulSoup4`
    *   `feedparser`
    *   `Selenium WebDriver` (for historical data collection)
    *   `xml.etree.ElementTree` (for sitemap parsing)
*   **Cloud Services & APIs**:
    *   `Google Cloud Secret Manager` (for secure credential storage)
    *   `Google Gemini API` (`google-generative-ai` library)
*   **Data Handling**:
    *   `json` (for structured data output)
    *   `hashlib` (for data integrity/deduplication)
    *   `re` (Regular Expressions for text processing)
    *   `datetime` (for date manipulation)
*   **Utilities**:
    *   `tqdm` (for progress bars)
    *   `unicodedata` (for text normalization)
    *   `secrets` (for secure random generation)
*   **Deployment Target (Implied)**: Apache Airflow
*   **Version Control**: Git (for submodule updates)

## Data Interaction

The `data_gathering.py` script serves as the central orchestrator for all data operations. It actively generates, updates, and maintains the accompanying JSON data files:

*   **Output Files**: The script populates and manages `GOOGLE_APPS_UPDATES.json`, `GOOGLE_BLOG.json`, `GOOGLE_CLOUD_BLOG.json`, `GOOGLE_DEEPMIND_BLOG.json`, `GOOGLE_DEVELOPERS_BLOG.json`, `GOOGLE_RESEARCH_BLOG.json`, `GOOGLE_TECHNOLOGY_BLOG.json`, and `worskpace_data.json`.
*   **JSON Structure**: Each JSON file stores a list of dictionaries, where each dictionary represents a blog post with a consistent schema, including keys like `website`, `link`, `title`, `thumbnail`, `author`, `track`, `description`, and `published_date`.
*   **Data Flow**: The script fetches raw content from various Google blog RSS feeds, XML sitemaps, and direct HTML pages, processes this content, extracts key metadata, and then serializes the structured data into these JSON files. Deduplication and merging operations ensure data quality and prevent redundancy across updates.
*   **Enrichment**: AI-generated summaries and tags are added to the existing data, enhancing the value of the stored blog post entries.

## Key Features

*   **Multi-Source Data Collection**: Gathers blog post data from a wide array of Google blogs, including Google Workspace, Developers, Apps Updates, Cloud, DeepMind, Research, Technology, and the general Google Blog.
*   **Dynamic Acquisition Methods**: Supports multiple data acquisition strategies:
    *   **RSS Feed Parsing**: Efficiently fetches the latest posts from blogs offering RSS feeds.
    *   **Sitemap Scraping**: Discovers blog post URLs and metadata by parsing XML sitemaps.
    *   **Direct HTML Scraping**: Extracts details from individual blog post pages using `requests` and `BeautifulSoup`.
*   **Historical Data Retrieval**: Leverages `Selenium WebDriver` to navigate and scrape historical blog posts from dynamic web pages, ensuring comprehensive data coverage.
*   **Secure Credential Management**: Integrates with Google Secret Manager to securely retrieve API keys (e.g., for Gemini API), enhancing security and maintainability.
*   **AI-Powered Content Enrichment**: Utilizes the Gemini API to automatically generate concise summaries and relevant tags for blog post content, adding valuable metadata for analysis and search.
*   **Robust Data Processing**: Includes functions for:
    *   Cleaning unwanted HTML attributes from content.
    *   Deduplicating lists of dictionaries based on specified keys.
    *   Merging data from different sources using left join logic.
*   **Airflow-Ready Design**: The modular function-based structure, avoiding traditional classes, makes the scripts highly adaptable for integration into Apache Airflow DAGs.
*   **Structured Output**: Stores all collected and enriched data in well-structured JSON files, ready for downstream consumption, analytics, or archival.

## How to Run

To set up and run this data collection pipeline, follow these steps:

1.  **Clone the Repository**:
    bash
    git clone https://github.com/Llubr/data-playground-base.git
    cd data-playground-base/google_blogs
    

2.  **Create and Activate a Virtual Environment**:
    bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    

3.  **Install Dependencies**:
    bash
    pip install google-cloud-secret-manager google-generativeai requests beautifulsoup4 feedparser selenium tqdm lxml
    

4.  **Set up Google Cloud Credentials & Secret Manager**:
    *   Ensure you have a Google Cloud project configured.
    *   Authenticate your environment for Google Cloud services (e.g., `gcloud auth application-default login`).
    *   Store your Gemini API key (or any other sensitive API key) in Google Secret Manager. The script expects a secret named `GEMINI_API_KEY` by default.

5.  **Install Selenium WebDriver**:
    *   Download the appropriate WebDriver for your browser (e.g., ChromeDriver for Google Chrome, GeckoDriver for Firefox). Place the WebDriver executable in a directory included in your system's PATH, or specify its path in the `start_dict` configuration.

6.  **Execute the Data Gathering Script**:
    The `data_gathering.py` script contains an `init()` function, suggesting it's designed to be called with a `start_dict` parameter for configuration. While a direct command-line execution requires a `main` block to initialize this `start_dict`, a common way to run it is by calling specific functions or wrapping them:

    ```python
    # Example of how you might run parts of the script (requires modification to data_gathering.py
    # to have a main execution block or be imported into another script).
    # Assuming start_dict is configured, e.g., for a specific site:

    # In a new `run_pipeline.py` file:
    # from data_gathering import init, workspace_data_get, get_ai_summary_tags, update_submodule_direct
    #
    # if __name__ == "__main__":
    #     config = {
    #         "project_id": "your-gcp-project-id",
    #         "secret_name_gemini": "GEMINI_API_KEY",
    #         "output_dir": ".", # Current directory for JSON files
    #         "webdriver_path": "/path/to/your/chromedriver", # Optional, if not in PATH
    #         # Other configurations...
    #     }
    #     start_dict = init(config)
    #
    #     # Example: Fetch Workspace data
    #     workspace_data_get(start_dict)
    #
    #     # Example: Enrich existing data (assuming 'worskpace_data.json' exists)
    #     # from_json = json.load(open('worskpace_data.json'))
    #     # get_ai_summary_tags(start_dict, from_json, 'blog_post')
    #
    #     # To run the full data collection, you would call each relevant function sequentially.
    ```
    
    Without a top-level execution block provided in `data_gathering.py`, you would typically import its functions into another script (e.g., `main.py` or an Airflow DAG) and call them with the appropriate `start_dict` configuration.