"""
Module 1 — Email Collection & Parsing
======================================
Accepts raw email input (string, .eml file, or dict) and returns a
structured EmailDocument ready for feature extraction.
"""

import re
import email
import email.policy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from loguru import logger


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmailDocument:
    """Structured representation of a parsed email."""
    raw: str = ""

    # Headers
    sender: str = ""
    sender_name: str = ""
    sender_domain: str = ""
    reply_to: str = ""
    recipients: list[str] = field(default_factory=list)
    subject: str = ""
    date: str = ""
    message_id: str = ""

    # Body
    body_text: str = ""
    body_html: str = ""

    # Derived
    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)

    # Metadata (filled by feature extractor)
    label: Optional[int] = None  # 1 = phishing, 0 = legitimate


# ── Parser ────────────────────────────────────────────────────────────────────

class EmailParser:
    """
    Parses emails from multiple input formats.

    Supported inputs:
        - Raw RFC-2822 string (copy-paste from email client)
        - Path to .eml file
        - Dict with keys: sender, subject, body, [reply_to, date]
        - EmailDocument (pass-through)
    """

    # Regex patterns
    _URL_RE = re.compile(
        r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+',
        re.IGNORECASE
    )
    _EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')

    # Suspicious file extensions for attachments
    _RISKY_EXTS = {
        '.exe', '.js', '.vbs', '.bat', '.cmd', '.ps1',
        '.docm', '.xlsm', '.pptm', '.jar', '.scr', '.hta'
    }

    def parse(self, source) -> EmailDocument:
        """Entry point — auto-detects input format."""
        if isinstance(source, EmailDocument):
            return source
        if isinstance(source, dict):
            return self._from_dict(source)
        if isinstance(source, Path) or (isinstance(source, str) and source.endswith('.eml')):
            return self._from_file(Path(source))
        if isinstance(source, str):
            return self._from_raw_string(source)
        raise ValueError(f"Unsupported input type: {type(source)}")

    # ── Input adapters ─────────────────────────────────────────────────────

    def _from_dict(self, d: dict) -> EmailDocument:
        doc = EmailDocument(raw=str(d))
        doc.sender = d.get("sender", d.get("from", ""))
        doc.subject = d.get("subject", "")
        doc.body_text = d.get("body", d.get("body_text", ""))
        doc.body_html = d.get("body_html", "")
        doc.reply_to = d.get("reply_to", "")
        doc.date = d.get("date", "")
        doc.recipients = self._ensure_list(d.get("to", []))
        self._enrich(doc)
        return doc

    def _from_file(self, path: Path) -> EmailDocument:
        raw = path.read_text(errors="replace")
        return self._from_raw_string(raw)

    def _from_raw_string(self, raw: str) -> EmailDocument:
        doc = EmailDocument(raw=raw)
        try:
            msg = email.message_from_string(raw, policy=email.policy.default)
            doc.sender = str(msg.get("From", ""))
            doc.reply_to = str(msg.get("Reply-To", ""))
            doc.subject = str(msg.get("Subject", ""))
            doc.date = str(msg.get("Date", ""))
            doc.message_id = str(msg.get("Message-ID", ""))
            to_field = msg.get("To", "")
            doc.recipients = [a.strip() for a in str(to_field).split(",") if a.strip()]
            doc.body_text, doc.body_html, doc.attachments = self._extract_body(msg)
        except Exception as exc:
            logger.warning(f"Email parse error, falling back to plain text: {exc}")
            doc.body_text = raw
        self._enrich(doc)
        return doc

    # ── Body extraction ────────────────────────────────────────────────────

    def _extract_body(self, msg) -> tuple[str, str, list[dict]]:
        text_parts, html_parts, attachments = [], [], []

        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get_content_disposition() or "")

            if "attachment" in disp:
                filename = part.get_filename() or "unknown"
                ext = Path(filename).suffix.lower()
                attachments.append({
                    "filename": filename,
                    "extension": ext,
                    "is_risky": ext in self._RISKY_EXTS,
                    "size": len(part.get_payload(decode=True) or b"")
                })
                continue

            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    text_parts.append(payload.decode(errors="replace"))

            elif ctype == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html_parts.append(payload.decode(errors="replace"))

        html_body = "\n".join(html_parts)
        text_body = "\n".join(text_parts)

        # If no plain text, strip HTML
        if not text_body and html_body:
            text_body = BeautifulSoup(html_body, "lxml").get_text(separator=" ")

        return text_body.strip(), html_body.strip(), attachments


    def _enrich(self, doc: EmailDocument) -> None:
        """Derive sender_domain, extract all URLs, resolve domain list."""
        # Sender domain
        doc.sender_domain = self._extract_domain(doc.sender)
        doc.sender_name = self._extract_display_name(doc.sender)

        # URLs from body text + HTML
        combined = doc.body_text + " " + doc.body_html
        doc.urls = self._extract_urls(combined)

        # Unique domains from URLs
        doc.domains = list({
            self._extract_domain_from_url(u)
            for u in doc.urls
            if self._extract_domain_from_url(u)
        })

  

    def _extract_domain(self, address: str) -> str:
        """Extract domain from 'Display Name <user@domain.com>' or 'user@domain.com'."""
        match = self._EMAIL_RE.search(address)
        if match:
            return match.group().split("@")[-1].lower().strip()
        return ""

    def _extract_display_name(self, address: str) -> str:
        match = re.match(r'^"?([^"<]+)"?\s*<', address.strip())
        return match.group(1).strip() if match else ""

    def _extract_urls(self, text: str) -> list[str]:
        return list(dict.fromkeys(self._URL_RE.findall(text)))  # deduplicated, order preserved

    def _extract_domain_from_url(self, url: str) -> str:
        try:
            return urlparse(url if url.startswith("http") else "http://" + url).netloc.lower()
        except Exception:
            return ""

    @staticmethod
    def _ensure_list(val) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []
