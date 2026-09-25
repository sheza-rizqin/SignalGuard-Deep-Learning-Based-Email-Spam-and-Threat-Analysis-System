
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"\')]+', re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(r"\b[a-z0-9][a-z0-9-]{1,60}\.(?:com|net|org|io|co|ru|cn|tk|xyz|top|info|biz|zip|mov|link|click|live|online)\b", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
PUNYCODE_RE = re.compile(r"xn--", re.IGNORECASE)
HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")

SUSPICIOUS_TLDS = frozenset({"zip", "mov", "tk", "top", "xyz", "gq", "cf", "click", "link", "work", "loan", "review"})

CREDENTIAL_RE = re.compile(
    r"\b(password|passcode|pin\b|credentials?|login details|sign in details|username and password|ssn|social security|date of birth|security question|one[- ]time code|otp)\b",
    re.IGNORECASE,
)
FINANCIAL_RE = re.compile(
    r"\b(invoice|wire transfer|bank transfer|payment|refund|billing|credit card|debit card|iban|swift|bitcoin|usdt|crypto wallet|gift card|western union|moneygram|remittance)\b",
    re.IGNORECASE,
)
ACCOUNT_RE = re.compile(
    r"\b(verify your account|confirm your account|account (?:has been|will be) (?:suspended|locked|closed|deactivated)|reactivate|validate your account|update your (?:billing|payment) (?:info|details)|unlock your account|restore access)\b",
    re.IGNORECASE,
)
URGENCY_RE = re.compile(
    r"\b(urgent|immediately|act now|right away|within \d+ (?:hours?|minutes?|days?)|final (?:notice|warning)|last (?:chance|warning)|expires? (?:today|soon|in)|time[- ]sensitive|asap|don'?t delay|limited time)\b",
    re.IGNORECASE,
)
FEAR_RE = re.compile(
    r"\b(suspended|terminated|legal action|law ?suit|court|penalty|fine|authorities|police|arrest|unauthori[sz]ed (?:access|transaction|login)|security breach|compromised|fraud(?:ulent)? (?:activity|transaction)|you will lose)\b",
    re.IGNORECASE,
)
REWARD_RE = re.compile(
    r"\b(congratulations|you (?:have|'ve) won|winner|prize|reward|bonus|claim your|free gift|lottery|jackpot|inheritance|total sum|usd ?\d[\d,]*|million (?:dollars|usd|pounds|euros))\b",
    re.IGNORECASE,
)
AUTHORITY_RE = re.compile(
    r"\b(microsoft|apple|google|amazon|paypal|netflix|dhl|fedex|irs\b|hmrc|bank of|it department|help ?desk|system administrator|security team|technical support|ceo\b|chief executive|hr department|human resources|customs|interpol|fbi)\b",
    re.IGNORECASE,
)
BYPASS_RE = re.compile(
    r"\b(do not (?:contact|tell|inform)|keep this confidential|without (?:telling|informing|notifying)|bypass|no need to (?:verify|confirm)|reply only to this (?:address|email)|do not discuss|between us|strictly confidential)\b",
    re.IGNORECASE,
)

SIGNAL_CATEGORIES: dict[str, tuple[str, re.Pattern[str]]] = {
    "urgency": ("Urgency or deadline pressure", URGENCY_RE),
    "authority": ("Authority or brand impersonation", AUTHORITY_RE),
    "fear": ("Fear or threat framing", FEAR_RE),
    "reward": ("Reward or gain framing", REWARD_RE),
    "credential_request": ("Credential or identity-data request", CREDENTIAL_RE),
    "financial_request": ("Financial or payment request", FINANCIAL_RE),
    "account_verification": ("Account-verification request", ACCOUNT_RE),
    "procedure_bypass": ("Pressure to bypass normal procedure", BYPASS_RE),
}

CREDENTIAL_ACTION_RE = re.compile(r"\b(click|follow|open|visit|download|reply|submit|enter|provide|send)\b", re.IGNORECASE)
@dataclass(frozen=True)
class ObservedSignal:

    category: str
    label: str
    evidence: tuple[str, ...]
    count: int
    source: str = "heuristic"

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "label": self.label,
            "evidence": list(self.evidence),
            "count": self.count,
            "source": self.source,
        }


@dataclass
class SignalReport:
    persuasion: list[ObservedSignal] = field(default_factory=list)
    urls: list[dict] = field(default_factory=list)
    url_findings: list[str] = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    hidden_content: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "persuasion": [signal.as_dict() for signal in self.persuasion],
            "urls": self.urls,
            "url_findings": self.url_findings,
            "structure": self.structure,
            "hidden_content": self.hidden_content,
            "disclaimer": (
                "These are deterministic rule matches over the email text and its HTML source. "
                "They are indicators only: they are produced by heuristics, not by the neural "
                "network, and they do not prove that an email is malicious."
            ),
        }


def observe_persuasion(text: str) -> list[ObservedSignal]:
    """Match the curated persuasion-signal families against the email text."""
    signals: list[ObservedSignal] = []
    for category, (label, pattern) in SIGNAL_CATEGORIES.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        evidence = tuple(dict.fromkeys(str(match).lower() for match in matches))[:6]
        signals.append(
            ObservedSignal(
                category=category,
                label=label,
                evidence=evidence,
                count=len(matches),
            )
        )
    if any(signal.category == "credential_request" for signal in signals) and CREDENTIAL_ACTION_RE.search(text):
        signals.append(
            ObservedSignal(
                category="credential_action_pairing",
                label="Credential request paired with an action instruction",
                evidence=tuple(dict.fromkeys(match.lower() for match in CREDENTIAL_ACTION_RE.findall(text)))[:6],
                count=len(CREDENTIAL_ACTION_RE.findall(text)),
            )
        )
    return signals
def _defang(url: str) -> str:
    """Neutralise a URL for display so it is never clickable in the UI."""
    return url.replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")


def _script_of_character(character: str) -> str | None:
    from .unicode_forensics import script_of

    return script_of(character)


def _url_features(url: str) -> dict:
    candidate = url if "://" in url else f"http://{url}"
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return {"url": _defang(url), "flags": ["unparseable"], "host": ""}
    host = (parts.hostname or "").lower()
    labels = [label for label in host.split(".") if label]
    top_level = labels[-1] if labels else ""
    flags: list[str] = []
    if IP_HOST_RE.match(host):
        flags.append("ip_literal_host")
    if PUNYCODE_RE.search(host):
        flags.append("punycode_host")
    if len(labels) >= 4:
        flags.append("deep_subdomain_chain")
    if top_level in SUSPICIOUS_TLDS:
        flags.append("high_abuse_tld")
    if parts.port:
        flags.append("explicit_port")
    if "@" in parts.netloc:
        flags.append("userinfo_in_authority")
    if HEX_ESCAPE_RE.search(host):
        flags.append("percent_encoded_host")
    if host.count("-") >= 3:
        flags.append("hyphen_heavy_host")
    if len(url) > 120:
        flags.append("excessive_length")
    scripts = {_script_of_character(character) for character in host}
    scripts.discard(None)
    if len(scripts) > 1:
        flags.append("mixed_script_host")
    if parts.scheme.lower() == "http":
        flags.append("plain_http")
    return {
        "url": _defang(url),
        "scheme": parts.scheme.lower(),
        "host": host,
        "registrable_hint": ".".join(labels[-2:]) if len(labels) >= 2 else host,
        "path_depth": len([segment for segment in parts.path.split("/") if segment]),
        "has_query": bool(parts.query),
        "flags": flags,
    }


def extract_anchors(raw_html: str) -> list[dict]:
    if "<" not in raw_html:
        return []
    soup = BeautifulSoup(raw_html, "html.parser")
    anchors: list[dict] = []
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        text = anchor.get_text(" ", strip=True)
        entry = {
            "text": text[:120],
            "href": _defang(href) if not href.lower().startswith(("javascript:", "data:")) else href[:60],
            "scheme": href.split(":", 1)[0].lower() if ":" in href else "",
        }
        if href.lower().startswith(("javascript:", "data:", "vbscript:")):
            entry["active_content"] = True
        host_text = URL_RE.match(text) or BARE_DOMAIN_RE.match(text)
        if host_text:
            try:
                visible_host = urlsplit(text if "://" in text else f"http://{text}").hostname or ""
            except ValueError:
                visible_host = ""
            actual_host = urlsplit(href if "://" in href else f"http://{href}").hostname or ""
            if visible_host and actual_host and visible_host.lower() != actual_host.lower():
                entry["mismatch"] = True
        anchors.append(entry)
    return anchors


def analyse_urls(raw_html: str, text: str) -> tuple[list[dict], list[str]]:
    found: dict[str, dict] = {}
    for match in URL_RE.findall(text):
        found.setdefault(match, _url_features(match))
    for anchor in extract_anchors(raw_html):
        href = anchor["href"].replace("hxxps://", "https://").replace("hxxp://", "http://").replace("[.]", ".")
        if href.lower().startswith(("javascript:", "data:", "vbscript:")):
            continue
        found.setdefault(href, _url_features(href))
    urls = list(found.values())
    findings: list[str] = []
    flagged = {flag for entry in urls for flag in entry["flags"]}
    if "ip_literal_host" in flagged:
        findings.append("At least one URL uses a raw IP address as its host.")
    if "punycode_host" in flagged:
        findings.append("At least one URL uses punycode (xn--) encoding, which can mask a lookalike domain.")
    if "deep_subdomain_chain" in flagged:
        findings.append("At least one URL has an unusually deep subdomain chain.")
    if "high_abuse_tld" in flagged:
        findings.append("At least one URL uses a top-level domain that appears disproportionately in abuse feeds.")
    if "userinfo_in_authority" in flagged:
        findings.append("At least one URL embeds user information before the host, which can disguise the real destination.")
    if "mixed_script_host" in flagged:
        findings.append("At least one URL host mixes Unicode scripts, a homoglyph-domain indicator.")
    if "plain_http" in flagged:
        findings.append("At least one URL uses unencrypted http.")
    return urls, findings
HIDDEN_STYLE_PATTERNS = (
    ("display:none", re.compile(r"display\s*:\s*none", re.IGNORECASE)),
    ("visibility:hidden", re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)),
    ("font-size:0", re.compile(r"font-size\s*:\s*0(?:px|pt|em|%)?", re.IGNORECASE)),
    ("opacity:0", re.compile(r"opacity\s*:\s*0(?:[.;])", re.IGNORECASE)),
    ("height:0", re.compile(r"height\s*:\s*0(?:px|;)", re.IGNORECASE)),
    ("html-hidden-attribute", re.compile(r"<[^>]+\shidden(?:\s|>|=)", re.IGNORECASE)),
    ("aria-hidden", re.compile(r"aria-hidden\s*=\s*[\"']true[\"']", re.IGNORECASE)),
)


def find_hidden_content(raw_html: str) -> list[dict]:
    """Report markup that would hide text from a human reader.

    Nothing is rendered or executed; the raw source is pattern-matched only.
    """
    if "<" not in raw_html:
        return []
    found: list[dict] = []
    for name, pattern in HIDDEN_STYLE_PATTERNS:
        occurrences = pattern.findall(raw_html)
        if occurrences:
            found.append({"technique": name, "occurrences": len(occurrences), "source": "heuristic"})
    if found:
        soup = BeautifulSoup(raw_html, "html.parser")
        forms = soup.find_all("form")
        if forms:
            found.append(
                {
                    "technique": "embedded-html-form",
                    "occurrences": len(forms),
                    "source": "heuristic",
                    "detail": "An embedded form was found. It is never rendered or submitted.",
                }
            )
        hidden_inputs = [element for element in soup.find_all("input") if (element.get("type") or "").lower() == "hidden"]
        if hidden_inputs:
            found.append(
                {
                    "technique": "hidden-form-field",
                    "occurrences": len(hidden_inputs),
                    "source": "heuristic",
                }
            )
    return found


def analyse_structure(raw_text: str, raw_html: str) -> dict:
    length = len(raw_text)
    letters = [character for character in raw_text if character.isalpha()]
    uppercase = [character for character in letters if character.isupper()]
    digits = [character for character in raw_text if character.isdigit()]
    soup = BeautifulSoup(raw_html, "html.parser") if "<" in raw_html else None
    remote_images = 0
    if soup is not None:
        for image in soup.find_all("img"):
            source = (image.get("src") or "")
            if source.lower().startswith(("http://", "https://", "//")):
                remote_images += 1
    return {
        "characters": length,
        "words": len(raw_text.split()),
        "lines": raw_text.count("\n") + 1 if raw_text else 0,
        "uppercase_ratio": (len(uppercase) / len(letters)) if letters else 0.0,
        "digit_ratio": (len(digits) / length) if length else 0.0,
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "currency_symbol_count": sum(raw_text.count(symbol) for symbol in ("$", "€", "£", "₹")),
        "contains_html": "<" in raw_html,
        "remote_image_count": remote_images,
        "mentions_attachment": bool(
            re.search(r"\b(attached|attachment|see attached|attached file|enclosed)\b", raw_text, re.IGNORECASE)
        ),
        "shouting_words": sum(
            1 for word in raw_text.split() if len(word) >= 4 and word.isupper()
        ),
    }


def observe(raw_subject: str, raw_body: str, model_text: str = "") -> SignalReport:

    report = SignalReport()
    combined_html = f"{raw_subject}\n{raw_body}"
    report.persuasion = observe_persuasion(combined_html)
    report.urls, report.url_findings = analyse_urls(combined_html, combined_html)
    anchors = extract_anchors(combined_html)
    mismatched = [anchor for anchor in anchors if anchor.get("mismatch")]
    if mismatched:
        report.url_findings.append(
            f"{len(mismatched)} anchor(s) display one destination but link to another."
        )
    active = [anchor for anchor in anchors if anchor.get("active_content")]
    if active:
        report.url_findings.append(
            f"{len(active)} anchor(s) use a script or data URI scheme instead of a normal link."
        )
    report.structure = analyse_structure(combined_html, combined_html)
    report.structure["model_saw_url_placeholder"] = bool(model_text) and "urltoken" in model_text.lower()
    report.hidden_content = find_hidden_content(combined_html)
    return report
