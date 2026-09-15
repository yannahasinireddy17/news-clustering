from datetime import datetime
from xml.etree import ElementTree as ET


def parse_rss_feed(rss_xml, source_name="RSS Feed", category=None):
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

        item_cat = (item.findtext("category") or "").strip()
        cat_value = item_cat or category or "World News"

        image_url = ""
        enclosure = item.find("enclosure")
        if enclosure is not None and enclosure.get("url"):
            image_url = enclosure.get("url", "").strip()
        if not image_url:
            for child in item:
                if "thumbnail" in child.tag.lower() or "content" in child.tag.lower():
                    u = child.get("url")
                    if u:
                        image_url = u.strip()
                        break

        items.append({
            "id": abs(hash(f"{source_name}:{title}:{link}")) % (10 ** 12),
            "source": source_name,
            "title": title,
            "date": date_value,
            "category": cat_value,
            "content": content,
            "url": link,
            "image_url": image_url,
        })
    return items
