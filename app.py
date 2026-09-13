import os
import re
import logging
from collections import defaultdict

import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from feeds import parse_rss_feed
from article_extractor import ArticleExtractionError, extract_article_text
from feeds_loader import fetch_articles_for_query, fetch_latest_articles
from news_pipeline import run_news_pipeline, run_saved_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEED_CONFIG_PATH = os.path.join(BASE_DIR, "data", "feeds.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_DISTANCE_THRESHOLD = 0.4


if load_dotenv:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
logger.info("[CONFIG] .env exists: %s", "YES" if os.path.exists(os.path.join(BASE_DIR, ".env")) else "NO")
logger.info("[CONFIG] python-dotenv available: %s", "YES" if load_dotenv else "NO")
logger.info("[CONFIG] NEWS_API_KEY loaded: %s", "YES" if os.getenv("NEWS_API_KEY") else "NO")


def ensure_nltk_resources():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()


def load_articles(path=None, query=None):
    query = (query or "").strip()
    if query:
        articles, _ = fetch_articles_for_query(
            query,
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
        if articles:
            return articles
        return []

    articles, _ = fetch_latest_articles(
        api_key=os.getenv("NEWS_API_KEY"),
        config_path=FEED_CONFIG_PATH,
        max_articles=30,
    )
    return articles


def filter_articles_by_query(articles, query):
    if not query:
        return articles

    keyword = query.strip().lower()
    filtered = []
    for article in articles:
        text = " ".join([
            article.get("title", ""),
            article.get("content", ""),
            article.get("source", "")
        ]).lower()
        if keyword in text:
            filtered.append(article)
    return filtered


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def generate_embeddings(texts):
    if not texts:
        return np.array([]).reshape((0, 0))

    logger.info("[NLP] Generating embeddings for %d articles...", len(texts))
    try:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info("[NLP] Embeddings generated: %d", len(embeddings))
        return np.asarray(embeddings)
    except Exception:
        logger.warning("[NLP] SBERT unavailable; using TF-IDF fallback")
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray()


def generate_cluster_title(cluster_articles):
    if not cluster_articles:
        return "News Event"

    combined_text = " ".join(
        f"{article.get('title', '')} {article.get('content', '')}"
        for article in cluster_articles
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8)
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = tfidf_matrix.toarray()[0]

    important_words = []
    for word, score in sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True):
        normalized = word.lower()
        if len(normalized) <= 3 or normalized in {"news", "said", "would", "also", "from", "with", "into", "after", "over", "about", "new"}:
            continue
        important_words.append(word)
        if len(important_words) >= 3:
            break

    if important_words:
        return " ".join(word.capitalize() for word in important_words)

    title = (cluster_articles[0].get("title") or "News Event").strip()
    return title[:60].strip() or "News Event"


def _extractive_summary(text, title="", max_sentences=3, max_length=300):
    source_text = " ".join(part.strip() for part in (title, text) if part and part.strip())
    if len(source_text.split()) < 12:
        return "Insufficient article text is available for a reliable summary."
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        if len(sentence.strip().split()) >= 5
    ]
    if not sentences:
        return "Insufficient article text is available for a reliable summary."
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = matrix.toarray().sum(axis=1)
    selected = [sentences[index] for index in np.argsort(scores)[::-1][:max_sentences]]
    summary = " ".join(selected)
    if len(summary) > max_length:
        summary = summary[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return summary


def summarize_article(article_text, title=""):
    """Create an extractive summary from one article only."""
    return _extractive_summary(article_text, title=title, max_sentences=3)


def select_related_articles(cluster_articles, similarity_threshold=0.6):
    """Keep articles sufficiently close to their cluster's content centre."""
    if len(cluster_articles) <= 1:
        return cluster_articles
    texts = [article.get("processed_text") or clean_text(
        f"{article.get('title', '')} {article.get('content', '')}"
    ) for article in cluster_articles]
    if not any(texts):
        return cluster_articles[:1]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    centre = np.asarray(matrix.mean(axis=0))
    similarities = cosine_similarity(matrix, centre).ravel()
    selected = [article for article, score in zip(cluster_articles, similarities) if score >= similarity_threshold]
    return selected or [cluster_articles[int(np.argmax(similarities))]]


def summarize_cluster(cluster_articles):
    if not cluster_articles:
        return "No articles available for summary."
    related = select_related_articles(cluster_articles)
    source_text = " ".join(
        f"{article.get('title', '')}. {article.get('content', '')}"
        for article in related
    )
    return _extractive_summary(source_text, max_sentences=min(3, len(related) + 1))


def build_event_clusters(articles, use_saved_model=False, pipeline_result=None):
    if not articles:
        logger.info("[NLP] Articles clustered: 0; clusters created: 0")
        return []

    if all("event_label" in article for article in articles) and not use_saved_model:
        grouped = defaultdict(list)
        for article in articles:
            grouped[article.get("event_label", "unknown")].append(article)
        clusters = []
        for index, cluster_articles in enumerate(grouped.values(), start=1):
            cluster_title = generate_cluster_title(cluster_articles)
            summary = summarize_cluster(cluster_articles)
            sources = sorted({article["source"] for article in cluster_articles})
            clusters.append({
                "cluster_id": index,
                "name": cluster_title,
                "title": cluster_title,
                "number": index,
                "summary": summary,
                "article_count": len(cluster_articles),
                "source_count": len(sources),
                "sources": sources,
                "articles": cluster_articles,
                "source_articles": {
                    source: [article for article in cluster_articles if article["source"] == source]
                    for source in sources
                },
            })
        return sorted(clusters, key=lambda cluster: cluster["title"].lower())

    if pipeline_result is None:
        pipeline_result = run_saved_pipeline(articles) if use_saved_model else run_news_pipeline(articles)
    grouped = defaultdict(list)
    cluster_metadata = {}
    for pipeline_cluster in pipeline_result["clusters"]:
        cluster_metadata[pipeline_cluster["cluster_id"] - 1] = pipeline_cluster
        for article in pipeline_cluster["articles"]:
            grouped[article["cluster_label"]].append(article)

    clusters = []
    for index, cluster_articles in enumerate(grouped.values(), start=1):
        cluster_title = generate_cluster_title(cluster_articles)
        summary = summarize_cluster(cluster_articles)
        sources = sorted({article["source"] for article in cluster_articles})
        cluster = {
            "cluster_id": index,
            "name": cluster_title,
            "title": cluster_title,
            "number": index,
            "summary": summary,
            "article_count": len(cluster_articles),
            "source_count": len(sources),
            "sources": sources,
            "articles": cluster_articles,
            "source_articles": {
                source: [article for article in cluster_articles if article["source"] == source]
                for source in sources
            },
            "label": cluster_metadata.get(index - 1, {}).get("label", "News"),
            "keywords": cluster_metadata.get(index - 1, {}).get("keywords", []),
            "silhouette_score": pipeline_result["metrics"]["silhouette_score"],
            "davies_bouldin_score": pipeline_result["metrics"]["davies_bouldin_score"],
            "model_used": pipeline_result.get("model_used", "dynamic_pipeline"),
        }
        clusters.append(cluster)

    logger.info("[NLP] Articles clustered: %d; clusters created: %d", len(articles), len(clusters))
    return sorted(clusters, key=lambda cluster: cluster["title"].lower())


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    latest = request.args.get("latest") == "1" or not query
    retrieval_errors = []
    logger.info("[NEWS] Flask query received: %s", query or "<none>")
    if query:
        articles, retrieval_errors = fetch_articles_for_query(
            query,
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
    elif latest:
        articles, retrieval_errors = fetch_latest_articles(
            api_key=os.getenv("NEWS_API_KEY"),
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
    else:
        articles, retrieval_errors = fetch_latest_articles(
            api_key=os.getenv("NEWS_API_KEY"),
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
    pipeline = run_saved_pipeline(articles) if articles else {
        "metrics": {"silhouette_score": None, "davies_bouldin_score": None},
        "visualization": [],
    }
    clusters = build_event_clusters(articles, use_saved_model=True, pipeline_result=pipeline)
    source_names = sorted({article["source"] for article in articles})
    return render_template(
        "index.html",
        articles=articles,
        clusters=clusters,
        article_count=len(articles),
        cluster_count=len(clusters),
        source_count=len(source_names),
        query=query,
        latest=latest,
        retrieval_errors=retrieval_errors,
        metrics=pipeline["metrics"],
        visualization=pipeline["visualization"],
        model_used=pipeline.get("model_used", "dynamic_pipeline"),
    )


@app.post("/api/summarize")
def summarize_article_api():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    description = (payload.get("description") or "").strip()

    try:
        article_text = extract_article_text(url)
        summary = summarize_article(article_text, title=payload.get("title", ""))
        return jsonify({"summary": summary, "basis": "Full article text"})
    except ArticleExtractionError as error:
        if description:
            summary = summarize_article(description, title=payload.get("title", ""))
            return jsonify({
                "summary": summary,
                "basis": "Publisher description/snippet only; full article text was unavailable.",
            })
        return jsonify({"error": str(error)}), 422


@app.route("/clusters")
def clusters_api():
    query = request.args.get("q", "").strip()
    latest = request.args.get("latest") == "1"
    if query:
        articles, errors = fetch_articles_for_query(
            query,
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
    elif latest:
        articles, errors = fetch_latest_articles(
            api_key=os.getenv("NEWS_API_KEY"),
            config_path=FEED_CONFIG_PATH,
            max_articles=30,
        )
    else:
        articles, errors = [], ["A query or latest=1 is required"]
    clusters = build_event_clusters(articles, use_saved_model=True)
    response = {"articles": articles, "clusters": clusters, "errors": errors}
    if articles:
        pipeline = run_saved_pipeline(articles)
        response["metrics"] = pipeline["metrics"]
        response["visualization"] = pipeline["visualization"]
        response["model_used"] = pipeline["model_used"]
    return jsonify(response)


@app.route("/cluster/<int:cluster_id>")
def cluster_detail(cluster_id):
    query = request.args.get("q", "").strip()
    articles = load_articles(query=query)
    cluster = next((item for item in build_event_clusters(articles, use_saved_model=True) if item["cluster_id"] == cluster_id), None)
    if cluster is None:
        return "Cluster not found", 404
    return render_template("cluster_detail.html", cluster=cluster)


@app.route("/article/<article_id>")
def article_detail(article_id):
    query = request.args.get("q", "").strip()
    articles = load_articles(query=query)
    article = next((item for item in articles if item["article_id"] == article_id), None)
    if article is None:
        return "Article not found", 404

    cluster = next((item for item in build_event_clusters(articles, use_saved_model=True) if any(a["article_id"] == article_id for a in item["articles"])), None)
    if cluster:
        article["cluster_name"] = cluster["title"]
    else:
        article["cluster_name"] = "Unassigned"

    return render_template("article.html", article=article)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
