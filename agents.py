"""Agent and chain definitions for the research pipeline."""
from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_settings
from tools import scrape_url, web_search

logger = logging.getLogger(__name__)
settings = get_settings()

llm = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.temperature,
    api_key=settings.openai_api_key,
    timeout=settings.request_timeout,
    max_retries=2,
)

# -----------------------------
# Search Agent
# -----------------------------
SEARCH_AGENT_SYSTEM_PROMPT = (
    "You are a research assistant. Use the web_search tool to find recent, "
    "reliable, and detailed information on the given topic. Prefer credible, "
    "well-known sources over low-quality ones. Always include the source URLs "
    "you found in your final answer."
)


def build_search_agent():
    """Agent that searches the web for information on a topic."""
    return create_agent(model=llm, tools=[web_search], system_prompt=SEARCH_AGENT_SYSTEM_PROMPT)


# -----------------------------
# Reader Agent
# -----------------------------
READER_AGENT_SYSTEM_PROMPT = (
    "You are a research assistant. Given a set of candidate URLs from search "
    "results, choose the single most relevant and credible one, use the "
    "scrape_url tool to read it, and summarize the important information you "
    "find. If scraping fails, say so plainly and summarize whatever "
    "information was already provided instead."
)


def build_reader_agent():
    """Agent that picks the best URL from search results and scrapes it."""
    return create_agent(model=llm, tools=[scrape_url], system_prompt=READER_AGENT_SYSTEM_PROMPT)


# -----------------------------
# Writer Chain
# -----------------------------
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write clear, well-structured, "
            "detailed, and insightful research reports using the provided "
            "information only.",
        ),
        (
            "human",
            """
Write a detailed research report on the following topic.

Topic:
{topic}

Research Material:
{research}

Structure the report as follows:

1. Introduction
2. Key Findings (at least 3 well-explained points)
3. Conclusion
4. Sources (list every URL mentioned in the research)

Guidelines:
- Be factual and accurate.
- Avoid generic statements.
- Explain each finding clearly.
- Use headings and proper formatting.
- Do not invent information that is not present in the research.
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()


# -----------------------------
# Critic Chain
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict and constructive research reviewer. Carefully "
            "evaluate the quality, clarity, factual accuracy, structure, and "
            "completeness of the report.",
        ),
        (
            "human",
            """
Review the following research report.

Report:
{report}

Respond in exactly this format:

Score: X/10

Strengths:
- Point 1
- Point 2
- Point 3

Areas for Improvement:
- Point 1
- Point 2
- Point 3

One-line Verdict:
Provide one concise sentence summarizing your overall assessment.
""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()