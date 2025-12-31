# 🎬 Intelligent Movie & TV Recommender System

This project presents a comprehensive content-based recommendation system for movies and TV shows, combining robust data engineering practices with machine learning models and an interactive web interface. It demonstrates a full pipeline from data acquisition and processing to model deployment and user interaction.

## Portfolio Context

This project serves as a showcase of end-to-end data science and engineering capabilities, including:
*   **Data Engineering**: Implementing ETL processes for external API data, data transformation, and cloud data warehousing (BigQuery).
*   **Machine Learning**: Developing and deploying content-based recommendation algorithms using TF-IDF and advanced sentence embeddings.
*   **Web Development**: Building an interactive user interface with Streamlit to serve real-time recommendations.
*   **Cloud Integration**: Leveraging Google BigQuery for scalable data storage and management.

## Difficulty Level

**Advanced**
The project integrates multiple complex components including API interaction, cloud database integration (BigQuery), natural language processing (TF-IDF, Sentence Embeddings), machine learning model development (cosine similarity), and a full-stack Streamlit application.

## Tools Used

| Category          | Tools & Libraries                                        |
| :---------------- | :------------------------------------------------------- |
| **ETL & Data**    | `pandas`, `requests`, `google-cloud-bigquery`, `tqdm`    |
| **Machine Learning** | `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), `numpy`, `SentenceTransformer` |
| **Web Application** | `streamlit`, `st_javascript`                           |
| **Utilities**     | `json`, `re`, `os`, `datetime` (`date`, `relativedelta`), `copy` |
| **Data Storage**  | Google BigQuery, Local JSON files                        |

## Data Interaction

The project meticulously handles data from various sources and formats:

1.  **External API Interaction**: The `movie_recommender_etl.py` script leverages `requests` to interact with an external movie/TV database API (inferred to be a popular one like TMDB based on function names). It fetches detailed information including movie/TV details, genres, watch providers, credits, and trending content.
2.  **Data Transformation and Loading**: The fetched raw data is processed and transformed using `pandas` to clean, standardize, and enrich the datasets. The `func_load_data` function then efficiently loads this processed data into **Google BigQuery** tables, facilitating scalable storage and querying.
3.  **Local Data Caching/Processing**:
    *   `movies_tv.json`: This file likely stores a curated dataset of movies and TV shows, including details such as `content_type`, `id`, `title`, `poster_path`, `info`, `country`, `watch_type`, and `provider_name`. This structured JSON is used by the Streamlit application to display content details and watch options.
    *   `movies_tv_4_recs.json`: This specialized JSON file contains data specifically prepped for the recommendation engine, including `id`, `title`, `overview`, `keywords`, `genres`, and a concatenated `soup` field. The `soup` field is crucial for the TF-IDF vectorization and content similarity calculations.
4.  **Recommendation Model Input**: The `movie_recommender_models.py` and `app.py` scripts utilize these processed datasets (either from BigQuery or the local JSON files) to build TF-IDF matrices and calculate cosine similarities for generating recommendations. The `load_and_process_data` function in `app.py` is responsible for loading these datasets into pandas DataFrames and preparing them for the recommendation engine, often caching results for performance.

## Key Features

*   **Automated ETL Pipeline**: Extracts movie and TV show data from external APIs, transforms it, and loads it into Google BigQuery.
*   **Content-Based Recommendation Engine**: Implements two distinct recommendation approaches:
    *   **TF-IDF**: Uses Term Frequency-Inverse Document Frequency to find semantic similarities between content descriptions.
    *   **Sentence Embeddings**: Leverages advanced pre-trained models (`SentenceTransformer`) for a deeper understanding of content context.
*   **BigQuery Integration**: Utilizes BigQuery for efficient and scalable storage of structured movie and TV show data.
*   **Interactive Streamlit Application**: Provides a user-friendly web interface to get personalized movie and TV show recommendations based on watched titles.
*   **Watch Provider Integration**: Displays available streaming/rental/buy options and their respective providers (e.g., Netflix, Amazon Prime Video).
*   **Robust Data Merging**: Includes a `merge_dictionaries` function for recursively combining complex nested dictionary structures, ensuring comprehensive data aggregation.
*   **Handles Movies & TV Shows**: Capable of recommending both movie and television series content.

## How to Run

To set up and run this project locally, follow these steps:

1.  **Clone the Repository**:
    bash
    git clone https://github.com/your_username/data-playground-base.git
    cd data-playground-base/movie_recommender
    

2.  **Set Up Python Environment**:
    It's recommended to use a virtual environment.
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    

3.  **Install Dependencies**:
    While a `requirements.txt` is not provided in the input, based on the imports, you would typically install the following:
    bash
    pip install pandas requests google-cloud-bigquery tqdm scikit-learn numpy sentence-transformers streamlit st-javascript
    
    *Note: Ensure `google-cloud-bigquery` is correctly configured with your GCP credentials if you intend to run the ETL script.*

4.  **Configure API Keys and BigQuery (ETL Specific)**:
    *   You will need an API key for the movie/TV database (e.g., TMDB API). Set this as an environment variable (e.g., `TMDB_API_KEY`) or similar configuration.
    *   For BigQuery, ensure your Google Cloud credentials are set up (e.g., via `gcloud auth application-default login` or by providing a service account key path).
    *   Update `movie_recommender_etl.py` with your `project_id`, `table`, and `schema` details for BigQuery if necessary.

5.  **Run the ETL Process (Optional, if starting fresh data)**:
    If you need to fetch fresh data and load it into BigQuery or generate the initial JSON files, execute the ETL script:
    bash
    python movie_recommender_etl.py
    
    *This step populates your BigQuery tables and/or generates the local `movies_tv.json` and `movies_tv_4_recs.json` files that the Streamlit app will use.*

6.  **Run the Streamlit Application**:
    Once the data (or pre-processed JSON files) are available, you can launch the interactive recommender application:
    bash
    streamlit run app.py
    
    This command will open the application in your default web browser.