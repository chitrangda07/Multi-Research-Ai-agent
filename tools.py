from langchain.tools import tool
import requests 
from bs4 import BeautifulSoup

from tavily import TavilyClient
import os 
from dotenv import load_dotenv
load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query : str) ->str:
    """Search the web for recent and reliable information about a query and return titles , urls and snippet """
    search_results = tavily_client.search(query = query, max_results=5)

    out = []

    for r in search_results['results']:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n\n")

    return "/n".join(out)

print(web_search.invoke("What is the latest news on AI?"))


@tool
def scarpe_url(url :str)-> str:
    """Scrape and return clean text content froma given URL for depper reading."""
    try:
        resp = requests.get(url , timeout = 8, headers = {'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script' , 'style' , 'nav' , "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:3000]
    except Exception as e:
        return f"Error scraping URL: {e}"


