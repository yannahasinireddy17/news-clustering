"""End-to-end news preprocessing, training, inference, and analysis."""

import json
import logging
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from cluster_analysis import analyze_centroid_clusters, analyze_cluster
from clustering import cluster_features, evaluate_clusters
from feature_extraction import extract_tfidf_features
from preprocessing import combine_article_text, preprocess_articles
from visualization import project_clusters


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
AG_NEWS_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}
logger = logging.getLogger(__name__)


def _article_records(frame):
    return [
        {
            "title": str(row["Title"]),
            "description": str(row["Description"]),
            "content": combine_article_text(row["Title"], row["Description"]),
            "class_index": int(row["Class Index"]),
        }
        for _, row in frame.iterrows()
    ]


def _build_result(processed, features, labels, cluster_info=None, true_labels=None):
    grouped = defaultdict(list)
    for article, label in zip(processed, labels):
        article["cluster_label"] = int(label)
        article["event_label"] = (cluster_info or {}).get(int(label), {}).get("label", "News")
        grouped[int(label)].append(article)

    clusters = []
    for cluster_id, cluster_articles in enumerate(grouped.values(), start=1):
        numeric_label = cluster_articles[0]["cluster_label"]
        topic, keywords = analyze_cluster(cluster_articles)
        info = (cluster_info or {}).get(numeric_label, {})
        label = info.get("label", topic)
        keywords = info.get("keywords", keywords)
        for article in cluster_articles:
            article["event_label"] = label
        clusters.append({
            "cluster_id": cluster_id,
            "articles": cluster_articles,
            "label": label,
            "keywords": keywords,
        })

    metrics = evaluate_clusters(features, np.asarray(labels))
    if true_labels is not None:
        metrics["adjusted_rand_score"] = round(float(adjusted_rand_score(true_labels, labels)), 3)
        metrics["normalized_mutual_info_score"] = round(float(normalized_mutual_info_score(true_labels, labels)), 3)
    return {
        "articles": processed,
        "clusters": clusters,
        "feature_count": features.shape[1] if features is not None else 0,
        "metrics": metrics,
        "visualization": project_clusters(features, labels),
    }


def run_news_pipeline(articles):
    processed = preprocess_articles(articles)
    vectorizer, features = extract_tfidf_features(processed)
    labels = cluster_features(features)
    if len(labels) == 0 and processed:
        labels = [0] * len(processed)
    return _build_result(processed, features, labels)


def refine_live_labels(articles, coarse_labels):
    """Refine coarse saved-model buckets using each article's topic evidence."""
    topic_ids = {}
    refined = []
    for article in articles:
        topic, _ = analyze_cluster([article])
        key = topic if topic != "News" else "News"
        topic_ids.setdefault(key, len(topic_ids))
        refined.append(topic_ids[key])
    logger.info(
        "[NLP] Live content refinement: %d coarse clusters -> %d topic clusters",
        len(set(int(label) for label in coarse_labels)),
        len(set(refined)),
    )
    return np.asarray(refined, dtype=int)


def train_ag_news_pipeline(train_path, test_path, model_dir=DEFAULT_MODEL_DIR, n_clusters=4, max_features=20000):
    """Fit K-Means on Title+Description text only and persist reusable artifacts."""
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    required = {"Class Index", "Title", "Description"}
    if not required.issubset(train_frame.columns) or not required.issubset(test_frame.columns):
        raise ValueError("AG News files must contain Class Index, Title, and Description columns")

    train_articles = preprocess_articles(_article_records(train_frame))
    test_articles = preprocess_articles(_article_records(test_frame))
    vectorizer, train_features = extract_tfidf_features_with_limit(train_articles, max_features)
    test_features = vectorizer.transform([article["processed_text"] for article in test_articles])
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    ).fit(train_features)
    train_predictions = kmeans.labels_

    # Class Index is used below only for external evaluation, never for labels or features.
    cluster_info = analyze_centroid_clusters(kmeans, vectorizer)
    os.makedirs(model_dir, exist_ok=True)
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "kmeans_model.joblib")
    metadata_path = os.path.join(model_dir, "cluster_metadata.json")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(kmeans, model_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"cluster_info": cluster_info, "n_clusters": n_clusters}, handle, indent=2)

    sample_size = min(10000, train_features.shape[0])
    sample_indices = np.linspace(0, train_features.shape[0] - 1, sample_size, dtype=int)
    train_metrics = evaluate_clusters(train_features[sample_indices], train_predictions[sample_indices])
    test_predictions = kmeans.predict(test_features)
    test_metrics = {
        "adjusted_rand_score": round(float(adjusted_rand_score(test_frame["Class Index"], test_predictions)), 3),
        "normalized_mutual_info_score": round(float(normalized_mutual_info_score(test_frame["Class Index"], test_predictions)), 3),
    }
    return {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "feature_count": len(vectorizer.get_feature_names_out()),
        "metrics": {"train": train_metrics, "test": test_metrics},
        "artifacts": {"vectorizer": vectorizer_path, "model": model_path, "metadata": metadata_path},
        "model_type": "KMeans",
    }


def extract_tfidf_features_with_limit(articles, max_features):
    texts = [article.get("processed_text", "") for article in articles]
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2)
    return vectorizer, vectorizer.fit_transform(texts)


def load_saved_pipeline(model_dir=DEFAULT_MODEL_DIR):
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "kmeans_model.joblib")
    metadata_path = os.path.join(model_dir, "cluster_metadata.json")
    if not all(os.path.exists(path) for path in (vectorizer_path, model_path, metadata_path)):
        return None
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError, KeyError, EOFError) as error:
        logger.warning("[NLP] Saved model could not be loaded: %s", error)
        return None
    return vectorizer, model, {int(key): value for key, value in metadata["cluster_info"].items()}


def run_saved_pipeline(articles, model_dir=DEFAULT_MODEL_DIR):
    """Transform live articles with the fitted AG News vectorizer and predict K-Means clusters."""
    saved = load_saved_pipeline(model_dir)
    if saved is None:
        logger.warning("[NLP] Model used: dynamic fallback; saved AG News artifacts unavailable")
        result = run_news_pipeline(articles)
        result["model_used"] = "dynamic_fallback"
        return result
    logger.info("[NLP] Model used: saved AG News TF-IDF + K-Means")
    vectorizer, model, cluster_info = saved
    processed = preprocess_articles(articles)
    features = vectorizer.transform([article.get("processed_text", "") for article in processed])
    coarse_labels = model.predict(features)
    labels = refine_live_labels(processed, coarse_labels)
    result = _build_result(processed, features, labels)
    result["model_used"] = "saved_ag_news_kmeans+content_refinement"
    logger.info("[NLP] Saved model cluster assignments: %d clusters", len(result["clusters"]))
    return result