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
POLICY_PATTERNS = {
    "Hateful Conduct": r"\b(?:Hateful Conduct|Hate Speech)\b",
    "Violence and Incitement": r"\bViolence and Incitement\b",
    "Dangerous Organizations and Individuals": r"\bDangerous Organizations and Individuals\b",
    "Adult Nudity and Sexual Activity": r"\bAdult Nudity and Sexual Activity\b",
    "Restricted Goods and Services": r"\b(?:Restricted|Regulated) Goods(?: and Services)?\b",
    "Privacy Violations": r"\b(?:Privacy Violations|image privacy rights)\b",
    "Fraud, Scams and Deceptive Practices": r"\b(?:Fraud, Scams and Deceptive Practices|Fraud and Deception)\b",
    "Human Exploitation": r"\bHuman Exploitation\b",
    "Coordinating Harm and Promoting Crime": r"\bCoordinating Harm and Promoting Crime\b",
    "Child Sexual Exploitation, Abuse and Nudity": r"\b(?:Child Sexual Exploitation|Child Nudity and Sexual Exploitation)\b",
    "Bullying and Harassment": r"\bBullying and Harassment\b",
    "Violent and Graphic Content": r"\b(?:Violent and Graphic Content|Graphic Violence)\b",
    "Misinformation": r"\bMisinformation\b",
    "Suicide, Self-Injury and Eating Disorders": r"\bSuicide, Self-Injury and Eating Disorders\b",
}


class DecisionCollector:
    """Resumable, rate-limited collector for public Oversight Board decisions."""

    def __init__(self, raw_dir: Path = Path("data/raw"), delay: float = 0.7):
        """Initialize the DecisionCollector instance and its required dependencies."""
        self.raw_dir = raw_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.pages_dir = raw_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=12), reraise=True)
    def fetch(self, url: str) -> str:
        """Download a page with retries and a persistent HTTP session."""
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
        """Discover decision-page URLs from the official listing pages."""
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
        """Extract the text that follows a named page heading."""
        node = soup.find(string=lambda value: value and value.strip().lower() == label.lower())
        if not node:
            return []
        container = node.parent
        sibling = container.find_next_sibling() or container.parent.find_next_sibling()
        if not sibling:
            return []
        return [x.strip() for x in sibling.get_text("|", strip=True).split("|") if x.strip()]

    def parse(self, url: str, html: str) -> Decision:
        """Parse one official decision page into the normalized data model."""
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup
        for tag in main.select("script, style, nav, footer, form"):
            tag.decompose()
        heading = main.find("h1")
        title = heading.get_text(" ", strip=True) if heading else url.rstrip("/").split("/")[-1]
        text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
        outcome = self._extract_outcome(text)
        date_match = re.search(
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            text,
        )
        published = (
            datetime.strptime(date_match.group(), "%B %d, %Y").date() if date_match else None
        )
        decision_type = self._extract_decision_type(text)
        case_match = re.search(r"\b20\d{2}-\d{3}(?:-[A-Z]{2})?-[A-Z]{2}-[A-Z]{2}\b", text)
        slug = url.rstrip("/").split("/")[-1]
        summary_match = re.search(
            r"(?:Case summary|Summary)\s*(.*?)(?=\n(?:Case description|Background|The Board|Meta's|Type of Decision)\b)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        policies = self._extract_policies(
            text, self._value_after_heading(soup, "Community Standard")
        )
        topics = self._value_after_heading(soup, "Topic")
        locations = self._value_after_heading(soup, "Location")
        platform = self._value_after_heading(soup, "Platform")
        meta_action = self._extract_meta_action(text)
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
            decision_type=decision_type,
            platform=platform,
            policies=policies,
            topics=topics,
            locations=locations,
            summary=summary_match.group(1).strip()[:5000] if summary_match else text[:1500],
            meta_action=meta_action,
            full_text=text,
            majority=majority_match.group(1)[:3000] if majority_match else "",
            minority=minority_match.group(1)[:3000] if minority_match else "",
            scraped_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _extract_outcome(text: str) -> Outcome:
        """Prefer the page's outcome label over mentions of unrelated past cases."""
        lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
        labels = {line for line in lines[:100] if line in {"upheld", "overturned"}}
        if labels == {"upheld"}:
            return Outcome.UPHELD
        if labels == {"overturned"}:
            return Outcome.OVERTURNED
        if len(labels) > 1:
            return Outcome.OTHER
        lead = " ".join(lines[:80])
        patterns = (
            (
                Outcome.OVERTURNED,
                r"the (?:oversight )?board overturns? meta(?:'s|’s) (?:original )?decision",
            ),
            (
                Outcome.UPHELD,
                r"the (?:oversight )?board upholds? meta(?:'s|’s) (?:original )?decision",
            ),
        )
        for outcome, pattern in patterns:
            if re.search(pattern, lead):
                return outcome
        return Outcome.OTHER

    @staticmethod
    def _extract_meta_action(text: str) -> str:
        """Infer Meta's original enforcement action from decision text."""
        lead = " ".join(line.strip() for line in text.splitlines()[:180]).casefold()
        if re.search(
            r"meta(?:'s|’s) (?:original )?decisions? to (?:remove|take down|delete)", lead
        ):
            return "removed"
        if re.search(
            r"meta(?:'s|’s) (?:original )?decisions? to (?:leave|keep) .{0,35}(?:up|online)",
            lead,
        ):
            return "left_up"
        if re.search(
            r"meta(?:'s|’s).{0,45}(?:disable|suspend|ban|restrict).{0,30}(?:account|profile)", lead
        ):
            return "account_action"
        if re.search(r"meta(?:'s|’s).{0,45}(?:warning screen|warning label|age-gat)", lead):
            return "warning"
        return "unknown"

    @staticmethod
    def _extract_decision_type(text: str) -> str:
        """Normalize the Board decision type from page metadata."""
        explicit = re.search(
            r"Type of Decision\s+(Standard|Summary|Expedited)", text, re.IGNORECASE
        )
        if explicit:
            return explicit.group(1).title()
        lead_lines = [line.strip() for line in text.splitlines()[:100] if line.strip()]
        lead = " ".join(lead_lines).casefold()
        if "this is a summary decision" in lead or "summary decisions examine cases" in lead:
            return "Summary"
        for line in lead_lines:
            if line.casefold() in {"standard", "summary", "expedited"}:
                return line.title()
        return "Unknown"

    @staticmethod
    def _extract_policies(text: str, labelled: list[str]) -> list[str]:
        """Extract and normalize policy labels mentioned in a decision."""
        normalized = []
        combined = " ".join(labelled)
        for name, pattern in POLICY_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                normalized.append(name)
        lead = text[:8000]
        for name, pattern in POLICY_PATTERNS.items():
            if name not in normalized and re.search(pattern, lead, re.IGNORECASE):
                normalized.append(name)
        return normalized or labelled

    def refresh_cached(self) -> list[Decision]:
        """Reparse every cached decision page after extraction logic changes."""
        output = self.raw_dir / "decisions.jsonl"
        current = read_jsonl(output, Decision)
        refreshed = [self.parse(str(item.url), self.fetch(str(item.url))) for item in current]
        write_jsonl(output, refreshed)
        return refreshed

    def collect(self, max_pages: int = 10, limit: int | None = None) -> list[Decision]:
        """Collect decision pages and persist normalized records incrementally."""
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
