# NBA Data Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Requests](https://img.shields.io/badge/Requests-library-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-data%20analysis-orange.svg)
![lxml](https://img.shields.io/badge/lxml-parsing-purple.svg)
![Google Cloud BigQuery](https://img.shields.io/badge/Google%20Cloud-BigQuery-yellowgreen.svg)

---

## Portfolio Context

This project resides in the `nba` subdirectory of a larger `data-playground-base` repository, demonstrating capabilities in data acquisition, processing, and storage specifically tailored for NBA-related information. It showcases an object-oriented approach to interact with external data sources.

---

## Difficulty Level
**Intermediate**

This project utilizes classes (`NBA()`) for an object-oriented design, interacts with web APIs and HTML content, processes data with Pandas, and integrates with Google BigQuery, indicating a solid understanding of modern Python development practices and cloud services.

## Tools Used
*   **Python**: Core programming language.
*   **Requests**: For making HTTP requests to fetch data from web sources.
*   **Pandas**: For efficient data manipulation and analysis.
*   **lxml.html**: For parsing HTML content, likely used in web scraping scenarios.
*   **Google Cloud BigQuery**: For robust data storage and querying in the cloud.
*   **json**: For handling JSON formatted data, typically from API responses.
*   **dataclasses**: For creating structured data models with less boilerplate.
*   **datetime, time, itertools, operator**: Standard Python libraries for various utility functions.

## Data Interaction
The primary script, `nba_game_summary.py`, interacts with data in several ways:
*   **API/Web Scraping**: It leverages the `requests` library to fetch data from external web sources (likely NBA APIs or websites) and uses `lxml.html` to parse HTML content, and `json` to process JSON responses from these sources.
*   **Data Processing**: Acquired data is then processed and transformed using the `pandas` library, preparing it for analysis or storage.
*   **Cloud Integration**: The script is designed to interact with **Google Cloud BigQuery**, suggesting that processed NBA data is intended for persistent storage and subsequent advanced analytics within a cloud data warehouse.

## Key Features
*   **Modular Data Acquisition**: Employs an `NBA()` class for organized and reusable data fetching logic.
*   **Web Data Extraction**: Capable of sending HTTP requests and parsing both JSON API responses and HTML content from various web sources.
*   **Data Transformation**: Utilizes Pandas for efficient cleaning, manipulation, and structuring of raw NBA data.
*   **Cloud Data Integration**: Includes functionality for loading processed data into Google Cloud BigQuery, enabling scalable data warehousing.
*   **Date & Time Management**: Integrates Python's `datetime` and `time` modules for handling time-sensitive data and operations.

## How to Run
To run this project, ensure you have Python 3.8+ installed and set up a virtual environment.

1.  **Clone the Repository (if not already done):**
    bash
    git clone https://github.com/your_username/data-playground-base.git
    cd data-playground-base/nba
    

2.  **Create and Activate a Virtual Environment:**
    bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    

3.  **Install Dependencies:**
    bash
    pip install requests pandas lxml google-cloud-bigquery
    
    *Note: Additional setup for Google Cloud authentication (e.g., setting `GOOGLE_APPLICATION_CREDENTIALS` environment variable) may be required if interacting with BigQuery.*

4.  **Execute the Main Script:**
    bash
    python nba_game_summary.py
    
    *Refer to the `nba_game_summary.py` script for specific command-line arguments or configuration options if available.*
