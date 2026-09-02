# NewsLens NLP Dashboard

A Flask-based news intelligence dashboard for query-driven news retrieval, semantic event clustering, and extractive multi-document summarization.

## Features

- Retrieves articles for the user's query through NewsAPI or configured RSS feeds
- Cleans and preprocesses text
- Generates semantic embeddings with Sentence-BERT
- Groups related articles using cosine-distance threshold clustering; the number of clusters is data-dependent
- Produces one extractive summary per event cluster
- Generates cluster titles from the cluster's article content
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

## Main files

- app.py — NLP pipeline and Flask routes
- feeds_loader.py — NewsAPI and RSS retrieval
- templates/index.html — dashboard template
- static/style.css — responsive styling
- data/feeds.json — configured live RSS sources used by search and Latest News
