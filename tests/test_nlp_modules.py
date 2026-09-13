from cluster_analysis import analyze_cluster
from news_pipeline import run_news_pipeline
from preprocessing import preprocess_text


def test_preprocessing_removes_markup_urls_and_stop_words():
    tokens = preprocess_text("<p>The AI market</p> is growing: https://example.org/story")

    assert tokens == ["ai", "market", "growing"]


def test_pipeline_extracts_features_labels_metrics_and_visualization():
    articles = [
        {"title": "AI chip investment", "content": "Technology firms increase investment in AI chips and cloud data centers."},
        {"title": "Cloud chip spending", "content": "Technology firms increase investment in AI chips and cloud data centers."},
        {"title": "Football match result", "content": "The football team wins a major league match in a sports competition."},
        {"title": "Football league victory", "content": "The football team wins a major league match in a sports competition."},
    ]

    result = run_news_pipeline(articles)

    assert result["feature_count"] > 0
    assert len(result["clusters"]) == 2
    assert {article["event_label"] for article in result["articles"]} == {"Technology", "Sports"}
    assert result["metrics"]["silhouette_score"] is not None
    assert result["metrics"]["davies_bouldin_score"] is not None
    assert len(result["visualization"]) == len(articles)
    assert {"x", "y", "cluster"}.issubset(result["visualization"][0])


def test_cluster_analysis_returns_keywords_and_topic_label():
    label, keywords = analyze_cluster([
        {"processed_text": "government election policy"},
        {"processed_text": "president policy"},
    ])

    assert label == "Politics"
    assert keywords[:2] == ["policy", "government"]
