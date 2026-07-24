from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini" , temperature = 0)

def build_search_agent():
    return create_agent(
        model = llm ,
        tools = [web_search]
    )

def build_reader_agent():
    return create_agent(
        model = llm ,
        tools = [scrape_url]
    )

writer_prompt = ChatPromptTemplate.from_template([
    ("system" , "You are an expert reseaech writer . Write lear , structured and insigntful reports.")
    ("human" , """Write a detailed research report on the topic below.
    
    Topic : {topic}

    Research Report :{research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed , factual and insightful in your writing. Avoid generic statements and provide specific information based on the research provided.
        """),
])
writer_chain = writer_prompt | llm | StrOutputParser()


critic_prompt =  ChatPromptTemplate.from_template([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.
    
    Report : {report}

    Respond in this exact format:
    Score : X/10
    Strengths : (list at least 3 specific strengths of the report)
    Area of improvement : (list at least 3 specific weaknesses of the report) 
    One line Verdict : (Provide a concise one-line summary of your overall assessment of the report)
    """),
])
critic_chain = critic_prompt | llm | StrOutputParser()