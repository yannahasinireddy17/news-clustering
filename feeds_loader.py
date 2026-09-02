import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from feeds import parse_rss_feed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


DEFAULT_FEEDS = [
    {
        "name": "BBC News",
        "url": "https://feeds.bbci.co.uk/news/rss.xml",
    },
    {
        "name": "NPR News",
        "url": "https://feeds.npr.org/1001/rss.xml",
    },
    {
        "name": "NASA News",
        "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    },
]


def load_feed_config(config_path):
    if not os.path.exists(config_path):
        return DEFAULT_FEEDS
    with open(config_path, "r", encoding="utf-8") as handle:
        feeds = json.load(handle)
    return feeds if isinstance(feeds, list) else DEFAULT_FEEDS


def fetch_feed(feed_url, source_name, timeout=8):
    logger.info("[NEWS] Calling RSS source: %s (%s)", source_name, feed_url)
    request = Request(
        feed_url,
        headers={"User-Agent": "NewsLens/1.0 RSS reader"},
    )
    with urlopen(request, timeout=timeout) as response:
        logger.info("[NEWS] RSS status for %s: %s", source_name, response.status)
        payload = response.read()
    articles = parse_rss_feed(payload.decode("utf-8", errors="replace"), source_name)
    logger.info("[NEWS] RSS articles parsed from %s: %d", source_name, len(articles))
    return articles


def is_real_source_url(url):
    if not url:
        return False
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and bool(hostname) and not hostname.endswith("example.com")


def normalize_article(article, source_name=None):
    url = (article.get("url") or article.get("link") or "").strip()
    if not is_real_source_url(url):
        return None

    description = (article.get("description") or article.get("content") or "").strip()
    title = (article.get("title") or "").strip()
    if not title:
        return None

    source = article.get("source")
    if isinstance(source, dict):
        source = source.get("name")
    source = (source or source_name or "Unknown source").strip()
    published_date = article.get("published_date") or article.get("publishedAt") or article.get("date") or "Unknown"
    return {
        "article_id": article.get("article_id") or f"article_{abs(hash(url))}",
        "title": title,
        "description": description,
        "content": description or title,
        "source": source,
        "published_date": str(published_date).replace("T", " ").replace("Z", "").strip(),
        "url": url,
        "image_url": (article.get("image_url") or article.get("urlToImage") or "").strip(),
    }


def deduplicate_articles(articles):
    unique = {}
    for article in articles:
        article = normalize_article(article)
        if not article:
            continue
        title = article.get("title", "").strip().lower()
        url = article.get("url", "").strip().lower()
        key = url or title
        if key and key not in unique:
            unique[key] = article
    return list(unique.values())


def filter_articles_by_query(articles, query):
    if not query:
        return articles

    keyword = query.strip().lower()
    filtered = []
    for article in articles:
        haystack = " ".join([
            article.get("title", ""),
            article.get("content", ""),
            article.get("source", "")
        ]).lower()
        if keyword in haystack:
            filtered.append(article)
    return filtered


def fetch_newsapi_articles(query, api_key, max_results=20):
    if not api_key or not query:
        return []

    request_params = {
        "q": query,
        "pageSize": max_results,
        "language": "en",
        "sortBy": "publishedAt",
    }
    params = urlencode({
        **request_params,
        "apiKey": api_key,
    })
    endpoint = f"https://newsapi.org/v2/everything?{params}"
    safe_endpoint = f"https://newsapi.org/v2/everything?{urlencode(request_params)}"
    logger.info("[NEWS] Calling NewsAPI: %s", safe_endpoint)
    request = Request(endpoint, headers={"User-Agent": "NewsLens/1.0 NewsAPI client"})

    with urlopen(request, timeout=12) as response:
        raw_response = response.read().decode("utf-8", errors="replace")
        logger.info("[NEWS] API status: %s; response bytes: %d", response.status, len(raw_response))
        payload = json.loads(raw_response)

    logger.info(
        "[NEWS] API response status field: %s; keys: %s; raw articles: %d",
        payload.get("status", "missing"),
        sorted(payload.keys()),
        len(payload.get("articles", [])),
    )

    articles = []
    for item in payload.get("articles", []):
        if not item.get("title"):
            continue
        article = normalize_article({
            "article_id": f"api_{abs(hash(item.get('url') or item.get('title')))}",
            "title": item.get("title", "").strip(),
            "description": item.get("description") or item.get("content") or "",
            "source": item.get("source"),
            "url": item.get("url", "").strip(),
            "publishedAt": item.get("publishedAt") or "Unknown",
            "image_url": item.get("urlToImage") or "",
        })
        if article:
            article["query_topic"] = query
            articles.append(article)
    logger.info("[NEWS] Valid articles after filtering: %d", len(articles))
    if not articles:
        logger.error("[NEWS ERROR] API returned no usable articles for query: %s", query)
    return articles


def fetch_latest_articles(api_key=None, config_path=None, max_articles=30):
    errors = []
    logger.info("[NEWS] Latest News requested")
    logger.info("[CONFIG] NEWS_API_KEY loaded: %s", "YES" if api_key else "NO")
    if api_key:
        params = urlencode({
            "pageSize": max_articles,
            "language": "en",
            "country": "us",
            "apiKey": api_key,
        })
        request = Request(
            f"https://newsapi.org/v2/top-headlines?{params}",
            headers={"User-Agent": "NewsLens/1.0 NewsAPI client"},
        )
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            articles = [normalize_article(item) for item in payload.get("articles", [])]
            articles = [article for article in articles if article]
            if articles:
                return deduplicate_articles(articles)[:max_articles], errors
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            errors.append(f"NewsAPI: {error}")

    rss_articles, rss_errors = load_live_articles(
        config_path or os.path.join(os.getcwd(), "data", "feeds.json"),
        timeout=10,
    )
    errors.extend(rss_errors)
    return deduplicate_articles(rss_articles)[:max_articles], errors


def load_live_articles(config_path, timeout=8):
    articles = []
    errors = []
    for feed in load_feed_config(config_path):
        name = feed.get("name", "RSS Feed")
        url = feed.get("url", "").strip()
        if not url:
            continue
        try:
            articles.extend(fetch_feed(url, name, timeout=timeout))
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            logger.error("[NEWS ERROR] RSS source failed for %s: %s", name, error)
            errors.append(f"{name}: {error}")
    valid_articles = deduplicate_articles(articles)
    logger.info("[NEWS] Valid RSS articles after deduplication: %d", len(valid_articles))
    return valid_articles, errors


def fetch_articles_for_query(query, config_path=None, max_articles=20):
    query = (query or "").strip()
    if not query:
        return [], []

    api_key = os.getenv("NEWS_API_KEY")
    errors = []
    logger.info("[NEWS] Query received: %s", query)
    logger.info("[CONFIG] NEWS_API_KEY loaded: %s", "YES" if api_key else "NO")

    if api_key:
        try:
            articles = fetch_newsapi_articles(query, api_key, max_results=max_articles)
            if articles:
                return deduplicate_articles(articles), errors
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            if isinstance(error, HTTPError):
                logger.error("[NEWS ERROR] API status: %s", error.code)
                if error.code == 401:
                    logger.error("[NEWS ERROR] Invalid API key")
            else:
                logger.error("[NEWS ERROR] Connection or response failure: %s", error)
            errors.append(f"NewsAPI: {error}")
    else:
        logger.error("[NEWS ERROR] NEWS_API_KEY is not configured; using RSS fallback")
        errors.append("NEWS_API_KEY is not configured; using RSS fallback")

    rss_articles, rss_errors = load_live_articles(config_path or os.path.join(os.getcwd(), "data", "feeds.json"), timeout=10)
    filtered = filter_articles_by_query(rss_articles, query)
    logger.info("[NEWS] RSS articles matching query '%s': %d", query, len(filtered))
    if rss_errors:
        errors.extend(rss_errors)
    if not filtered:
        logger.error("[NEWS ERROR] No articles returned for query: %s", query)
        errors.append(f"No live articles returned for query: {query}")
        return [], errors
    return deduplicate_articles(filtered)[:max_articles], errors
