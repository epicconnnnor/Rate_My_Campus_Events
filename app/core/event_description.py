"""Safe display helpers for descriptions imported from the campus calendar."""

import re
from html import escape
from urllib.parse import urlparse

from markupsafe import Markup


_LINK = re.compile(
    r"(?P<markdown>\[(?P<label>[^\]\n]{1,500})\]\((?P<href>https?://[^\s)]+)\))"
    r"|(?P<url>https?://[^\s<]+)"
)


def _is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _inline_links(text: str) -> str:
    """Escape all text, turning only HTTP(S) links into safe anchors."""
    rendered = []
    position = 0
    for match in _LINK.finditer(text):
        rendered.append(escape(text[position:match.start()]))
        url = match.group("href") or match.group("url")
        label = match.group("label") or url
        trailing = ""
        # Sentence punctuation belongs outside a pasted URL.
        if match.group("url"):
            original_url = url
            url = original_url.rstrip(".,;:!?")
            trailing = original_url[len(url):]
            label = url
        if _is_safe_url(url):
            rendered.append(
                f'<a href="{escape(url, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">{escape(label)}</a>'
            )
        else:
            rendered.append(escape(match.group(0)))
        rendered.append(escape(trailing))
        position = match.end()
    rendered.append(escape(text[position:]))
    return "".join(rendered)


def render_event_description(value: str | None) -> Markup:
    """Render paragraphs and calendar links without allowing arbitrary HTML."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value or "") if part.strip()]
    if not paragraphs:
        return Markup("")
    return Markup("".join(
        f"<p>{_inline_links(paragraph).replace(chr(10), '<br>')}</p>"
        for paragraph in paragraphs
    ))
