from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# -----------------------------
# Search Agent
# -----------------------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )


# -----------------------------
# Reader Agent
# -----------------------------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )


# -----------------------------
# Writer Prompt
# -----------------------------
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write clear, well-structured, detailed, and insightful research reports using the provided information only.",
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
# Critic Prompt
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict and constructive research reviewer. Carefully evaluate the quality, clarity, factual accuracy, structure, and completeness of the report.",
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