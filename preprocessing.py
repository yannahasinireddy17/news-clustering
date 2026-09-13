"""Text cleaning and normalization for news articles."""

import re

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer


def _load_nltk_resources():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


try:
    _load_nltk_resources()
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
except LookupError:  # pragma: no cover - offline fallback
    STOP_WORDS = set()

LEMMATIZER = WordNetLemmatizer()


STOP_WORDS.update({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "will", "with", "would",
})


def preprocess_text(text):
    """Return normalized tokens after removing markup, URLs, and stop words."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
    tokens = [token for token in wordpunct_tokenize(text) if re.fullmatch(r"[a-z]+", token)]
    return [LEMMATIZER.lemmatize(token) for token in tokens if token not in STOP_WORDS and len(token) > 1]


def preprocess_articles(articles):
    """Add normalized text without changing source article metadata."""
    processed = []
    for article in articles:
        item = dict(article)
        item["processed_text"] = " ".join(
            preprocess_text(f"{article.get('title', '')} {article.get('content', '')}")
        )
        processed.append(item)
    return processed


def combine_article_text(title="", description="", content=""):
    """Use dataset Title+Description and live article title+content consistently."""
    return " ".join(part for part in (title, description or content) if part)