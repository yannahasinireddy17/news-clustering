import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, build_event_clusters, filter_articles_by_query, parse_rss_feed
from feeds_loader import fetch_articles_for_query


SAMPLE_RSS = '''
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>AI chips spark new investment</title>
      <link>https://example.com/1</link>
      <pubDate>Tue, 15 Aug 2026 08:00:00 GMT</pubDate>
      <description>Technology firms are investing heavily in AI chips and advanced cloud infrastructure.</description>
    </item>
    <item>
      <title>Chip investment accelerates in cloud markets</title>
      <link>https://example.com/2</link>
      <pubDate>Tue, 15 Aug 2026 09:00:00 GMT</pubDate>
      <description>Cloud providers increase spending on semiconductors and AI data centers.</description>
    </item>
  </channel>
</rss>
'''


def test_parse_rss_feed_extracts_articles():
    articles = parse_rss_feed(SAMPLE_RSS, source_name="Example Feed")
    assert len(articles) == 2
    assert articles[0]["title"]
    assert articles[0]["source"] == "Example Feed"
    assert "chip" in articles[0]["content"].lower()


def test_filter_articles_by_query_matches_topic_terms():
    articles = [
        {"title": "ISRO launches new satellite", "content": "Space mission completes launch", "source": "BBC"},
        {"title": "Weather system hits Hyderabad", "content": "Heavy rain floods city roads", "source": "Reuters"},
    ]

    filtered = filter_articles_by_query(articles, "isro")
    assert len(filtered) == 1
    assert "ISRO" in filtered[0]["title"]


def test_filter_articles_by_query_ignores_ai_in_url_campaign_param():
    articles = [
        {
            "title": "Pubs trial digital ID apps",
            "content": "Venues may accept digital identity apps.",
            "source": "BBC Technology",
            "category": "technology",
            "url": "https://www.bbc.co.uk/news/articles/cm4gl6j53w19o?at_campaign=rss",
        },
        {
            "title": "What is AI and how does it work?",
            "content": "Artificial intelligence systems are expanding quickly.",
            "source": "BBC Technology",
            "category": "technology",
            "url": "https://www.bbc.co.uk/news/articles/c2l799gxjjpo?at_campaign=rss",
        },
    ]

    filtered = filter_articles_by_query(articles, "AI")
    assert len(filtered) == 1
    assert "AI" in filtered[0]["title"]


def test_build_event_clusters_uses_similarity_threshold_not_fixed_cluster_count():
  articles = [
    {"article_id": f"s{i}", "title": f"AI chip investment round {i}", "content": "Large technology firms are increasing investment in AI chips and cloud and data center expansion.", "source": "Reuters"}
    for i in range(1, 7)
  ] + [
    {"article_id": f"d{i}", "title": f"Weather warning in city {i}", "content": "Heavy rain and flooding are expected across urban areas with roads closing for safety.", "source": "BBC"}
    for i in range(1, 4)
  ]

  clusters = build_event_clusters(articles)

  assert len(clusters) == 2
  assert sorted(cluster["article_count"] for cluster in clusters) == [3, 6]


def test_query_retrieval_prefers_newsapi_when_configured(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    expected = [{"article_id": "api_1", "title": "AI policy update", "url": "https://www.reuters.com/world/ai-policy"}]

    monkeypatch.setattr("feeds_loader.fetch_newsapi_articles", lambda query, api_key, max_results: expected)
    monkeypatch.setattr("feeds_loader.load_live_articles", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RSS should not be used")))

    articles, errors = fetch_articles_for_query("AI policy", config_path="unused")

    assert len(articles) == 1
    assert articles[0]["article_id"] == expected[0]["article_id"]
    assert articles[0]["url"] == expected[0]["url"]
    assert articles[0]["content"] == expected[0]["title"]
    assert errors == []


def test_query_rss_fallback_returns_only_matching_articles(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.setattr(
        "feeds_loader.load_live_articles",
        lambda *args, **kwargs: ([
            {"title": "Cricket final reaches last over", "content": "The cricket match went to the final over.", "source": "BBC", "url": "https://www.bbc.com/sport/cricket"},
            {"title": "Technology earnings rise", "content": "Software companies reported stronger earnings.", "source": "BBC", "url": "https://www.bbc.com/news/technology"},
        ], []),
    )

    articles, errors = fetch_articles_for_query("cricket", config_path="unused")

    assert [article["title"] for article in articles] == ["Cricket final reaches last over"]
    assert articles[0]["url"].endswith("/cricket")
    assert errors == ["NEWS_API_KEY is not configured; using RSS fallback"]


def test_clusters_endpoint_returns_retrieval_errors(monkeypatch):
    monkeypatch.setattr(
        "app.fetch_articles_for_query",
        lambda query, config_path, max_articles: ([], ["NEWS_API_KEY is not configured"]),
    )

    response = app.test_client().get("/clusters?q=ISRO")

    assert response.status_code == 200
    assert response.get_json() == {
        "articles": [],
        "clusters": [],
        "errors": ["NEWS_API_KEY is not configured"],
    }
