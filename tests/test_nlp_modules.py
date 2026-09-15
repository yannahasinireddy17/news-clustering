from app import select_related_articles, summarize_article, summarize_cluster
from cluster_analysis import analyze_cluster
from news_pipeline import refine_live_labels, run_news_pipeline
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


def test_individual_summary_uses_only_selected_article():
    summary = summarize_article(
        "The Mars mission sent new images to scientists. The images will support future research.",
        title="NASA Mars mission update",
    )

    assert "Mars" in summary
    assert "unrelated election" not in summary


def test_cluster_summary_filters_unrelated_article():
    articles = [
        {"title": "Storm reaches coast", "content": "A storm reached the coast and caused flooding.", "processed_text": "storm reached coast caused flooding"},
        {"title": "Flooding closes roads", "content": "Flooding from the storm closed roads near the coast.", "processed_text": "flooding storm closed roads coast"},
        {"title": "Football final begins", "content": "The football final begins tonight at the national stadium.", "processed_text": "football final begins tonight national stadium"},
    ]

    related = select_related_articles(articles)
    summary = summarize_cluster(articles)

    assert len(related) == 2
    assert "storm" in summary.lower() or "flood" in summary.lower()
    assert "football" not in summary.lower()


def test_extended_topic_labels_are_supported():
    health, _ = analyze_cluster([{"processed_text": "doctor patient hospital vaccine disease"}])
    science, _ = analyze_cluster([{"processed_text": "nasa space mars satellite research"}])
    finance, _ = analyze_cluster([{"processed_text": "investors stocks markets bank currency"}])
    travel, _ = analyze_cluster([{"processed_text": "tourism flight hotel destination airport"}])

    assert health == "Health"
    assert science == "Science / Space / NASA"
    assert finance == "Finance / Markets"
    assert travel == "Travel"


def test_unsupported_content_is_not_forced_into_a_topic():
    label, _ = analyze_cluster([{"processed_text": "ordinary report with general information"}])

    assert label == "News"


def test_live_refinement_separates_supported_topics_without_fixed_clusters():
    articles = [
        {"processed_text": "president election government policy"},
        {"processed_text": "football team league match player"},
        {"processed_text": "nasa space mars satellite research"},
    ]

    labels = refine_live_labels(articles, [0, 0, 0])

    assert len(set(labels.tolist())) == 3
