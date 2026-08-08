"""Tools available to the research agents: web search and URL scraping."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from tavily import TavilyClient

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
tavily_client = TavilyClient(api_key=settings.tavily_api_key)

_BLOCKED_HOSTNAMES = {"localhost"}


def _is_safe_url(url: str) -> bool:
    """Reject non-http(s) URLs and obvious attempts to reach internal/private
    addresses, so an LLM-chosen URL can't be used to probe internal services
    (basic SSRF guard).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.hostname:
        return False
    if parsed.hostname.lower() in _BLOCKED_HOSTNAMES:
        return False

    try:
        resolved_ip = socket.gethostbyname(parsed.hostname)
        ip = ipaddress.ip_address(resolved_ip)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    except (socket.gaierror, ValueError):
        # DNS resolution failed here; let `requests` raise a clear error later
        # rather than silently blocking (avoids false positives on flaky DNS).
        pass

    return True


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information about a query.
    Returns titles, URLs, and snippets from the search results.
    """
    if not query or not query.strip():
        return "Search Error: empty query."

    try:
        response = tavily_client.search(query=query, max_results=settings.max_search_results)
    except Exception as exc:  # Tavily can raise several distinct exception types
        logger.exception("Tavily search failed for query=%r", query)
        return f"Search Error: {exc}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for result in results:
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = (result.get("content") or "")[: settings.snippet_char_limit]
        lines.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

    return "\n".join(lines)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a given URL for deeper reading.
    """
    url = (url or "").strip()
    if not url:
        return "Error scraping URL: empty URL."

    if not _is_safe_url(url):
        return f"Error scraping URL: '{url}' is not an allowed public http/https address."

    try:
        response = requests.get(
            url,
            timeout=settings.request_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchMindBot/1.0)"},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return f"Error scraping URL: {exc}"

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"Error scraping URL: unsupported content type '{content_type}'."

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    if not text:
        return "Error scraping URL: no readable text content found on the page."

    return text[: settings.scrape_char_limit]