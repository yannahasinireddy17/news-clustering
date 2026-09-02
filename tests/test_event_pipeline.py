from feeds_loader import is_real_source_url, normalize_article


def test_placeholder_urls_are_rejected():
    assert is_real_source_url("https://www.reuters.com/world/article")
    assert not is_real_source_url("https://example.com/demo")


def test_article_normalization_preserves_real_metadata():
    article = normalize_article({
        "title": "Current news report",
        "description": "A publisher-provided description.",
        "source": {"name": "Reuters"},
        "publishedAt": "2026-08-22T10:30:00Z",
        "url": "https://www.reuters.com/world/article",
        "urlToImage": "https://www.reuters.com/image.jpg",
    })

    assert article["source"] == "Reuters"
    assert article["published_date"] == "2026-08-22 10:30:00"
    assert article["url"].startswith("https://www.reuters.com/")
    assert article["image_url"].endswith("image.jpg")
