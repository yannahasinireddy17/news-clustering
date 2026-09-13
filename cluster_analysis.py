"""Cluster keywords and human-readable topic labels."""

from collections import Counter

import numpy as np


TOPIC_KEYWORDS = {
    "World": {"world", "country", "international", "war", "conflict", "border", "refugee", "diplomatic"},
    "Technology": {"ai", "software", "technology", "chip", "cloud", "data", "internet"},
    "Business": {"business", "market", "company", "investment", "economy", "trade", "stock"},
    "Sports": {"sports", "game", "team", "match", "player", "league", "cricket", "football"},
    "Politics": {"government", "president", "minister", "election", "policy", "parliament", "politics", "senate", "congress"},
    "Entertainment": {"film", "movie", "music", "actor", "show", "entertainment", "celebrity"},
    "Science / Space / NASA": {"science", "scientist", "space", "nasa", "planet", "mars", "satellite", "astronomy", "research"},
    "Health": {"health", "medical", "medicine", "hospital", "doctor", "disease", "cancer", "patient", "drug", "vaccine"},
}


def analyze_cluster(articles):
    counts = Counter()
    for article in articles:
        counts.update(article.get("processed_text", "").split())
    keywords = [word for word, _ in counts.most_common(5)]
    topic_scores = {
        topic: sum(counts[word] for word in words)
        for topic, words in TOPIC_KEYWORDS.items()
    }
    label = max(topic_scores, key=topic_scores.get) if max(topic_scores.values(), default=0) else "News"
    return label, keywords


def analyze_centroid_clusters(model, vectorizer, class_labels=None, top_n=8):
    """Create labels and keywords from K-Means centroids, without using labels as features."""
    terms = np.asarray(vectorizer.get_feature_names_out())
    results = {}
    for cluster_id, centroid in enumerate(model.cluster_centers_):
        indices = np.argsort(centroid)[::-1][:top_n]
        keywords = terms[indices].tolist()
        label = class_labels.get(cluster_id) if class_labels else None
        if not label:
            label, _ = analyze_cluster([{"processed_text": " ".join(keywords)}])
        results[cluster_id] = {"label": label, "keywords": keywords}
    return results