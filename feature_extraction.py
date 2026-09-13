"""TF-IDF feature extraction for normalized news text."""

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(articles):
    texts = [article.get("processed_text", "") for article in articles]
    if not any(texts):
        return TfidfVectorizer(), None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    return vectorizer, vectorizer.fit_transform(texts)