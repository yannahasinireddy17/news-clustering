import hashlib
import json
import logging
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from feeds import parse_rss_feed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _stable_article_id(prefix, value):
    digest = hashlib.md5((value or "").encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}_{digest}"


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

DEFAULT_CATEGORY_FEEDS = {
    "cricket": [
        {"name": "BBC Sport Cricket", "url": "https://feeds.bbci.co.uk/sport/cricket/rss.xml", "category": "cricket"},
        {"name": "ESPN Cricinfo", "url": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml", "category": "cricket"},
    ],
    "football": [
        {"name": "BBC Sport Football", "url": "https://feeds.bbci.co.uk/sport/football/rss.xml", "category": "football"},
    ],
    "sports": [
        {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/rss.xml", "category": "sports"},
        {"name": "NPR Sports", "url": "https://feeds.npr.org/1055/rss.xml", "category": "sports"},
    ],
    "technology": [
        {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "category": "technology"},
        {"name": "NPR Technology", "url": "https://feeds.npr.org/1019/rss.xml", "category": "technology"},
    ],
    "science": [
        {"name": "NASA News", "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss", "category": "science"},
        {"name": "BBC Science", "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "category": "science"},
        {"name": "NPR Science", "url": "https://feeds.npr.org/1007/rss.xml", "category": "science"},
    ],
    "business": [
        {"name": "BBC Business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml", "category": "business"},
        {"name": "NPR Business", "url": "https://feeds.npr.org/1006/rss.xml", "category": "business"},
    ],
    "entertainment": [
        {"name": "BBC Entertainment", "url": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml", "category": "entertainment"},
        {"name": "NPR Arts & Culture", "url": "https://feeds.npr.org/1008/rss.xml", "category": "entertainment"},
    ],
    "health": [
        {"name": "BBC Health", "url": "https://feeds.bbci.co.uk/news/health/rss.xml", "category": "health"},
        {"name": "NPR Health", "url": "https://feeds.npr.org/1128/rss.xml", "category": "health"},
    ],
    "politics": [
        {"name": "BBC Politics", "url": "https://feeds.bbci.co.uk/news/politics/rss.xml", "category": "politics"},
        {"name": "NPR Politics", "url": "https://feeds.npr.org/1014/rss.xml", "category": "politics"},
    ],
}

# Broad topic queries that should keep articles from their mapped category feeds.
CATEGORY_TOPIC_QUERIES = {
    "sports", "sport", "technology", "tech", "science", "business", "finance",
    "entertainment", "health", "medical", "politics", "political", "government",
    "elections", "election", "cricket", "football", "soccer",
}

CATEGORY_KEYWORDS = {
    "cricket": [
        "cricket", "cricketer", "cricketers", "ipl", "bcci", "icc", "t20", "odi", "wicket", "batsman", "batter",
        "bowler", "test match", "ashes", "ranji", "cricinfo", "kohli", "rohit", "dhoni", "bumrah"
    ],
    "football": [
        "football", "footballer", "soccer", "premier league", "la liga", "serie a", "bundesliga",
        "champions league", "fifa", "uefa", "nfl", "messi", "ronaldo", "haaland", "mbappe", "striker",
        "manchester", "arsenal", "liverpool", "chelsea", "barcelona", "real madrid"
    ],
    "sports": [
        "sports", "sport", "tennis", "badminton", "basketball", "nba", "f1", "formula 1", "olympics",
        "athletics", "golf", "rugby", "hockey", "wimbledon"
    ],
    "technology": [
        "technology", "tech", "ai", "artificial intelligence", "machine learning", "deep learning",
        "robotics", "software", "hardware", "cyber", "cybersecurity", "chips", "semiconductor",
        "semiconductors", "computing", "computer", "algorithm", "algorithms", "app", "apps",
        "cloud", "google", "apple", "microsoft", "meta", "nvidia", "intel", "amd", "openai",
        "chatgpt", "anthropic", "gemini", "crypto", "cryptocurrency", "bitcoin", "blockchain"
    ],
    "science": [
        "science", "nasa", "space", "astronomy", "astrophysics", "mars", "moon", "lunar", "isro",
        "esa", "telescope", "james webb", "hubble", "satellite", "satellites", "rocket", "rockets",
        "spacex", "artemis", "cosmic", "cosmos", "galaxy", "physics", "quantum", "planet", "planets",
        "asteroid", "meteor", "supernova", "climate", "environment"
    ],
    "business": [
        "business", "finance", "economy", "economic", "market", "markets", "stock", "stocks",
        "shares", "wall street", "nasdaq", "dow jones", "banking", "bank", "banks", "inflation",
        "interest rate", "interest rates", "gdp", "investing", "investor", "investors", "commerce",
        "trade", "earnings", "revenue", "profit", "recession", "fintech"
    ],
    "entertainment": [
        "entertainment", "movies", "movie", "cinema", "film", "films", "hollywood", "bollywood",
        "actor", "actress", "actors", "actresses", "oscar", "oscars", "emmy", "emmys", "grammy",
        "grammys", "music", "musician", "musicians", "celebrity", "celebrities", "theatre", "theater",
        "box office", "pop culture", "arts", "television", "tv show", "series", "album", "songs", "song"
    ],
    "health": [
        "health", "medical", "medicine", "hospital", "hospitals", "doctor", "doctors", "patient",
        "patients", "disease", "diseases", "virus", "infection", "vaccine", "vaccines", "vaccination",
        "healthcare", "wellness", "mental health", "cancer", "fda", "who", "pharma", "pharmaceutical",
        "drug", "drugs", "treatment", "therapy", "clinical"
    ],
    "politics": [
        "politics", "political", "election", "elections", "government", "parliament", "congress",
        "senate", "president", "prime minister", "policy", "democrat", "republican", "vote",
        "voting", "campaign", "minister", "legislation", "white house", "downing street"
    ],
}


def detect_query_categories(query):
    if not query:
        return []
    q_norm = query.strip().lower()
    matched = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw == q_norm or re.search(r'\b' + re.escape(kw) + r'\b', q_norm):
                if cat not in matched:
                    matched.append(cat)
                break
    return matched


def load_feed_config(config_path, category=None):
    if isinstance(config_path, list):
        if category:
            feeds = [f for f in config_path if f.get("category") == category]
            return feeds if feeds else DEFAULT_CATEGORY_FEEDS.get(category, [])
        return config_path
    if isinstance(config_path, dict):
        if category:
            return config_path.get("categories", {}).get(category, DEFAULT_CATEGORY_FEEDS.get(category, []))
        return config_path.get("general", DEFAULT_FEEDS)

    if not isinstance(config_path, str) or not os.path.exists(config_path):
        if category:
            return DEFAULT_CATEGORY_FEEDS.get(category, [])
        return DEFAULT_FEEDS

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            feeds = json.load(handle)
    except Exception as e:
        logger.error("[NEWS ERROR] Failed to read feed config %s: %s", config_path, e)
        if category:
            return DEFAULT_CATEGORY_FEEDS.get(category, [])
        return DEFAULT_FEEDS

    if isinstance(feeds, dict):
        if category:
            return feeds.get("categories", {}).get(category, DEFAULT_CATEGORY_FEEDS.get(category, []))
        return feeds.get("general", DEFAULT_FEEDS)
    elif isinstance(feeds, list):
        if category:
            cat_feeds = [f for f in feeds if f.get("category") == category]
            return cat_feeds if cat_feeds else DEFAULT_CATEGORY_FEEDS.get(category, [])
        return feeds
    return DEFAULT_FEEDS


def fetch_feed(feed_url, source_name, timeout=8, category=None):
    logger.info("[NEWS] Calling RSS source: %s (%s)", source_name, feed_url)
    request = Request(
        feed_url,
        headers={"User-Agent": "NewsLens/1.0 RSS reader"},
    )
    with urlopen(request, timeout=timeout) as response:
        logger.info("[NEWS] RSS status for %s: %s", source_name, response.status)
        payload = response.read()
    articles = parse_rss_feed(payload.decode("utf-8", errors="replace"), source_name, category=category)
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
    existing_id = article.get("article_id") or article.get("id")
    if existing_id is not None and str(existing_id).strip():
        article_id = str(existing_id)
        if article_id.isdigit():
            article_id = _stable_article_id("article", url)
    else:
        article_id = _stable_article_id("article", url)

    return {
        "article_id": article_id,
        "title": title,
        "description": description,
        "content": description or title,
        "source": source,
        "category": article.get("category", "World News"),
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


def _article_search_haystack(article):
    """Build searchable text without URL query strings (avoids false matches like 'ai' in 'campaign')."""
    raw_url = (article.get("url") or "").strip()
    parsed = urlparse(raw_url)
    url_text = f"{parsed.netloc} {parsed.path}".strip()
    return " ".join([
        article.get("title", ""),
        article.get("content", ""),
        article.get("description", ""),
        article.get("source", ""),
        article.get("category", ""),
        url_text,
    ]).lower()


def filter_articles_by_query(articles, query):
    if not query:
        return articles

    keyword = query.strip().lower()
    matched_cats = {cat.lower() for cat in detect_query_categories(keyword)}
    allow_category_match = keyword in CATEGORY_TOPIC_QUERIES or keyword in matched_cats

    if keyword == "ai":
        pattern = re.compile(
            r'\b(ai|artificial intelligence|machine learning|deep learning|llm|chatgpt|openai|anthropic|generative ai)\b',
            re.IGNORECASE,
        )
    elif keyword in {"movie", "movies", "film", "films"}:
        pattern = re.compile(
            r'\b(movie|movies|film|films|cinema|hollywood|bollywood|box office|actor|actress|oscar|oscars)\b',
            re.IGNORECASE,
        )
    elif keyword == "cricket":
        pattern = re.compile(
            r'\b(cricket|cricketer|cricketers|wicket|wickets|batsman|batter|bowler|test match|ipl|bcci|icc|t20|odi|cricinfo|innings)\b',
            re.IGNORECASE,
        )
    elif keyword in {"football", "soccer"}:
        pattern = re.compile(
            r'\b(football|footballer|soccer|premier league|fifa|uefa|striker|champions league)\b',
            re.IGNORECASE,
        )
    elif keyword == "nasa":
        pattern = re.compile(
            r'\b(nasa|artemis|james webb|hubble|kennedy space|marshall space|goddard)\b',
            re.IGNORECASE,
        )
    elif keyword == "space":
        pattern = re.compile(
            r'\b(space|nasa|spacex|orbit|orbital|satellite|rocket|astronaut|cosmos|galaxy|mars|lunar|moon)\b',
            re.IGNORECASE,
        )
    else:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

    # Short tokens like "ai" must not use naive substring matching (matches inside "campaign", etc.).
    use_substring = len(keyword) > 2
    words = [w for w in keyword.split() if len(w) > 2]

    filtered = []
    for article in articles:
        haystack = _article_search_haystack(article)
        category = (article.get("category") or "").strip().lower()

        if pattern.search(haystack):
            filtered.append(article)
        elif use_substring and keyword in haystack:
            filtered.append(article)
        elif words and all(w in haystack for w in words):
            filtered.append(article)
        elif allow_category_match and category and category in matched_cats:
            filtered.append(article)
        elif keyword == "nasa" and ("nasa" in (article.get("source") or "").lower() or "nasa.gov" in haystack):
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
        title = (item.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        url = (item.get("url") or "").strip()
        if not is_real_source_url(url):
            continue
        article = normalize_article({
            "article_id": _stable_article_id("api", url or title),
            "title": title,
            "description": item.get("description") or item.get("content") or "",
            "source": item.get("source"),
            "url": url,
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


def load_live_articles(config_path, timeout=8, feeds=None, category=None, query=None):
    articles = []
    errors = []
    feed_list = feeds
    if feed_list is None:
        if category:
            feed_list = load_feed_config(config_path, category=category)
        elif query:
            cats = detect_query_categories(query)
            if cats:
                feed_list = []
                for cat in cats:
                    feed_list.extend(load_feed_config(config_path, category=cat))
            else:
                feed_list = load_feed_config(config_path, category=None)
        else:
            feed_list = load_feed_config(config_path, category=None)

    seen_urls = set()
    unique_feeds = []
    for f in (feed_list or []):
        url = f.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_feeds.append(f)

    for feed in unique_feeds:
        name = feed.get("name", "RSS Feed")
        url = feed.get("url", "").strip()
        feed_cat = feed.get("category", category)
        if not url:
            continue
        try:
            articles.extend(fetch_feed(url, name, timeout=timeout, category=feed_cat))
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
                return deduplicate_articles(articles)[:max_articles], errors
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

    cfg_path = config_path or os.path.join(os.getcwd(), "data", "feeds.json")
    rss_articles, rss_errors = load_live_articles(cfg_path, timeout=10, query=query)
    filtered = filter_articles_by_query(rss_articles, query)
    logger.info("[NEWS] RSS articles matching query '%s': %d", query, len(filtered))
    if rss_errors:
        errors.extend(rss_errors)
    if not filtered:
        logger.error("[NEWS ERROR] No articles returned for query: %s", query)
        errors.append(f"No live articles returned for query: {query}")
        return [], errors
    return deduplicate_articles(filtered)[:max_articles], errors
