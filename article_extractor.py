from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency is installed in normal app setup
    trafilatura = None


class ArticleExtractionError(Exception):
    pass


def extract_article_text(url, timeout=12):
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ArticleExtractionError("The article URL is not valid.")
    if parsed.hostname and parsed.hostname.lower().endswith("example.com"):
        raise ArticleExtractionError("The article URL is a placeholder and was rejected.")
    if trafilatura is None:
        raise ArticleExtractionError("Article text extraction is not installed.")

    request = Request(
        url,
        headers={"User-Agent": "NewsLens/1.0 article reader"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            html = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise ArticleExtractionError("The original source could not be reached.") from error

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text or len(text.split()) < 40:
        raise ArticleExtractionError("The publisher did not expose enough readable article text.")
    return text.strip()
