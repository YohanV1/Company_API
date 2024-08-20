# from langchain_groq import ChatGroq # llm1 = ChatGroq(model="llama-3.1-8b-instant", temperature=1, max_retries=2)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from agent_tools.web_page_tool import WebPageTool
from agent_tools.google_search_tool import GoogleSearchTool
from agent_tools.metadesc_tool import MetaDescriptionTool
from agent_tools.company_info_extractor_tool import CompanyInfoExtractorTool
from agent_tools.whois_tool import WhoisTool
import naics_rag.query

import streamlit as st
import whois
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide", page_title='Company Profiling Demo')
st.sidebar.title("Testing UI for Company Profiling Tool.")
with st.sidebar.expander("Details"):
    st.write(f"Testing APIs - OpenAI, Groq, Google Custom Search, Perplexity,"
             f"etc.")

meta_description_tool = MetaDescriptionTool()
company_info_extractor_tool = CompanyInfoExtractorTool()
whois_tool = WhoisTool()
page_getter = WebPageTool()
google_search_tool = GoogleSearchTool(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

url_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a website retriever. Your sole purpose is to provide "
            "the official website URL for the company name given as input. "
            "Use the google_search_tool to find the URL. Respond with ONLY "
            "the URL,"
            "without any additional text, explanation, or formatting. If you "
            "cannot find a definitive URL, respond with 'URL_NOT_FOUND' and "
            "nothing else.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def analyze_company(company_name):
    tools = [google_search_tool]
    website_agent = create_tool_calling_agent(llm, tools, url_prompt)
    website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)
    url_result = website_agent_executor.invoke({"input": company_name})
    url = url_result['output']

    if url != "URL_NOT_FOUND":
        meta_description_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a company analyzer. Your purpose is to provide the meta description of"
                "the given company based on its website content. Use the meta_description_tool."
                "Do not include any additional information or formatting. If you cannot find enough "
                "information for any field, respond with 'Information not available' for that field."
            ),
            ("placeholder", "{chat_history}"),
            ("human", f"Analyze this company: {url}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        meta_tools = [meta_description_tool]
        meta_agent = create_tool_calling_agent(llm, meta_tools, meta_description_prompt)
        meta_agent_executor = AgentExecutor(agent=meta_agent, tools=meta_tools, verbose=True)
        meta_result = meta_agent_executor.invoke({"input": f"Get meta description for: {url}"})
        meta_description = meta_result['output']

        company_info_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a company analyzer. Your purpose is to provide company information. "
                "Use the page_getter tool for this."
                "If you cannot find enough information, "
                "respond with 'Information not available' for that field."
            ),
            ("placeholder", "{chat_history}"),
            ("human", f"Analyze this company: {url}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        info_tools = [page_getter]
        info_agent = create_tool_calling_agent(llm, info_tools, company_info_prompt)
        info_agent_executor = AgentExecutor(agent=info_agent, tools=info_tools, verbose=True)
        info_result = info_agent_executor.invoke({"input": f"Get company information for: {url}"})
        company_info = info_result['output']

        results = naics_rag.query.query_rag(company_info)

        return url, meta_description, company_info, results
    else:
        return url, "Could not find a URL for the company.", "Could not find a URL for the company.", "Could not find a URL for the company."


st.title("Company Profiling")

company_name = st.text_input("Enter company name:")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        url, meta_description, company_info, results = analyze_company(company_name)

    st.write(f"**Company URL**: {url}")

    st.subheader("Meta Description")
    st.write(meta_description)
    st.write(company_info)

    results_content = results.content

    st.subheader("NAICS Data:")
    st.write(results_content)

    st.subheader("Website Age")
    try:
        domain = whois.whois(url)
        creation_date = domain.creation_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            current_date = datetime.now()
            age = current_date - creation_date
            years = age.days // 365
            remaining_days = age.days % 365
            st.write(f"The website {url} is approximately {years} years and {remaining_days} days old.")
        else:
            st.write(f"Unable to determine the age of {url}. Creation date not available.")
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

    st.subheader("Whois Data:")
    st.write(whois.whois(url))
