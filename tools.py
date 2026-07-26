from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import requests
import os

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information about a query.
    Returns titles, URLs, and snippets from the search results.
    """

    try:
        search_results = tavily_client.search(
            query=query,
            max_results=5
        )

        output = []

        for result in search_results.get("results", []):
            output.append(
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {result['content'][:300]}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search Error: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from a given URL for deeper reading.
    """

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary HTML elements
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Limit text size
        return text[:3000]

    except Exception as e:
        return f"Error scraping URL: {str(e)}"