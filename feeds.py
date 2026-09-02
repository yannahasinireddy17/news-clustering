from datetime import datetime
from xml.etree import ElementTree as ET


def parse_rss_feed(rss_xml, source_name="RSS Feed"):
    """Parse RSS/XML string and return list of article dicts."""
    root = ET.fromstring(rss_xml)
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or item.findtext("content") or "").strip()
        content = description or title
        date_value = "Unknown"
        if published:
            try:
                dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                date_value = dt.strftime("%Y-%m-%d")
            except Exception:
                try:
                    dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    date_value = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_value = published

        items.append({
            "id": abs(hash(f"{source_name}:{title}:{link}")) % (10 ** 12),
            "source": source_name,
            "title": title,
            "date": date_value,
            "category": "World News",
            "content": content,
            "url": link,
        })
    return items
