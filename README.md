# NewsLens NLP Dashboard

A Flask-based news intelligence dashboard for query-driven news retrieval, NLP preprocessing, TF-IDF feature extraction, event clustering, cluster analysis, evaluation, and extractive multi-document summarization.

## Features

- Retrieves articles for the user's query through NewsAPI or configured RSS feeds
- Collects and normalizes articles from NewsAPI and configured RSS feeds
- Cleans HTML, URLs, punctuation, and stop words in a dedicated preprocessing module
- Extracts sparse TF-IDF unigram and bigram features
- Groups related articles using cosine-distance threshold clustering by default; `clustering.py` also provides K-Means when a fixed cluster count is required
- Labels clusters from topic keyword scores and extracts representative keywords
- Evaluates valid clusterings with Silhouette and Davies-Bouldin scores
- Produces PCA coordinates for a frontend or Plotly visualization
- Produces one extractive summary per event cluster
- Displays source comparison, article metadata, and event clusters in a responsive dashboard

## Run

1. Open a terminal in this project folder.
2. Install dependencies:

   python -m pip install -r requirements.txt

3. Start the app:

   python app.py

4. Open the browser at:

   http://127.0.0.1:5000

For live search, copy `.env.example` to `.env` and set `NEWS_API_KEY`. If no key is configured, the app searches the RSS feeds in `data/feeds.json`.

## Pipeline modules

- feeds_loader.py — NewsAPI/RSS collection and article normalization
- preprocessing.py — cleaning, tokenization, and stop-word removal
- feature_extraction.py — TF-IDF vectorization
- clustering.py — similarity clustering and evaluation metrics
- cluster_analysis.py — topic labels and top keywords
- visualization.py — PCA coordinates for cluster plots
- news_pipeline.py — orchestration of the complete NLP pipeline

## Main files

- app.py — Flask routes and web integration
- templates/index.html — dashboard template
- static/style.css — responsive styling
- data/feeds.json — configured live RSS sources used by search and Latest News
