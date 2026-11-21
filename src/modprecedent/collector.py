from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .io import read_jsonl, write_jsonl
from .models import Decision, Outcome

LOG = logging.getLogger(__name__)
BASE_URL = "https://www.oversightboard.com/decision/"
HEADERS = {"User-Agent": "ModPrecedent/0.1 academic-research contact: local-course-project"}


class DecisionCollector:
    """Resumable, rate-limited collector for public Oversight Board decisions."""

    def __init__(self, raw_dir: Path = Path("data/raw"), delay: float = 0.7):
        self.raw_dir = raw_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.pages_dir = raw_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=12), reraise=True)
    def fetch(self, url: str) -> str:
        cache = self.pages_dir / (re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_") + ".html")
        if cache.exists() and cache.stat().st_size > 500:
            return cache.read_text(encoding="utf-8")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        page_text = response.content.decode("utf-8", errors="replace")
        cache.write_text(page_text, encoding="utf-8")
        time.sleep(self.delay)
        return page_text

    def discover(self, max_pages: int = 10) -> list[str]:
        urls: set[str] = set()
        for page in range(1, max_pages + 1):
            html = self.fetch(f"{BASE_URL}?pg={page}")
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.select('a[href*="/decision/"]'):
                url = urljoin(BASE_URL, link.get("href", "")).split("?")[0]
                path = urlparse(url).path.rstrip("/")
                if path != "/decision" and re.search(r"/decision/[^/]+$", path):
                    urls.add(url.rstrip("/") + "/")
            LOG.info("listing page %s: %s decision URLs", page, len(urls))
        return sorted(urls)

    @staticmethod
    def _value_after_heading(soup: BeautifulSoup, label: str) -> list[str]:
        node = soup.find(string=lambda value: value and value.strip().lower() == label.lower())
        if not node:
            return []
        container = node.parent
        sibling = container.find_next_sibling() or container.parent.find_next_sibling()
        if not sibling:
            return []
        return [x.strip() for x in sibling.get_text("|", strip=True).split("|") if x.strip()]

    def parse(self, url: str, html: str) -> Decision:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup
        for tag in main.select("script, style, nav, footer, form"):
            tag.decompose()
        heading = main.find("h1")
        title = heading.get_text(" ", strip=True) if heading else url.rstrip("/").split("/")[-1]
        text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
        low = text.lower()
        outcome = Outcome.OTHER
        if re.search(
            r"\b(overturns?|overturned|reverses?)\b.{0,80}\b(meta|facebook|instagram)", low
        ):
            outcome = Outcome.OVERTURNED
        elif re.search(r"\b(upholds?|upheld)\b.{0,80}\b(meta|facebook|instagram)", low):
            outcome = Outcome.UPHELD
        date_match = re.search(
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            text,
        )
        published = (
            datetime.strptime(date_match.group(), "%B %d, %Y").date() if date_match else None
        )
        type_match = re.search(
            r"Type of Decision\s+(Standard|Summary|Expedited)", text, re.IGNORECASE
        )
        case_match = re.search(r"\b20\d{2}-\d{3}(?:-[A-Z]{2})?-[A-Z]{2}-[A-Z]{2}\b", text)
        slug = url.rstrip("/").split("/")[-1]
        summary_match = re.search(
            r"(?:Case summary|Summary)\s*(.*?)(?=\n(?:Case description|Background|The Board|Meta's|Type of Decision)\b)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        policies = self._value_after_heading(soup, "Community Standard")
        topics = self._value_after_heading(soup, "Topic")
        locations = self._value_after_heading(soup, "Location")
        platform = self._value_after_heading(soup, "Platform")
        minority_match = re.search(
            r"((?:A |The )?minority.*?)(?=\n[A-Z][^\n]{3,80}\n|$)", text, re.IGNORECASE | re.DOTALL
        )
        majority_match = re.search(
            r"((?:A |The )?majority.*?)(?=\n[A-Z][^\n]{3,80}\n|$)", text, re.IGNORECASE | re.DOTALL
        )
        return Decision(
            case_id=case_match.group() if case_match else slug.upper(),
            title=title,
            url=url,
            published_date=published,
            outcome=outcome,
            decision_type=type_match.group(1).title() if type_match else "Unknown",
            platform=platform,
            policies=policies,
            topics=topics,
            locations=locations,
            summary=summary_match.group(1).strip()[:5000] if summary_match else text[:1500],
            full_text=text,
            majority=majority_match.group(1)[:3000] if majority_match else "",
            minority=minority_match.group(1)[:3000] if minority_match else "",
            scraped_at=datetime.now(UTC).isoformat(),
        )

    def collect(self, max_pages: int = 10, limit: int | None = None) -> list[Decision]:
        output = self.raw_dir / "decisions.jsonl"
        existing = {str(d.url): d for d in read_jsonl(output, Decision)}
        urls = self.discover(max_pages)
        if limit:
            urls = urls[:limit]
        for index, url in enumerate(urls, 1):
            if url in existing:
                continue
            try:
                existing[url] = self.parse(url, self.fetch(url))
                write_jsonl(output, existing.values())
                LOG.info("collected %s/%s %s", index, len(urls), url)
            except (requests.RequestException, ValueError) as exc:
                LOG.warning("failed %s: %s", url, exc)
        return list(existing.values())
