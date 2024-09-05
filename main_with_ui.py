from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from agent_tools.web_page_tool import WebPageTool
from agent_tools.link_retrieval_tool import LinkRetriever
from agent_tools.metadesc_tool import MetaDescriptionTool
from agent_tools.company_info_extractor_tool import CompanyInfoExtractorTool
from agent_tools.additional_info_search_tool import AdditionalInfoSearch
import naics_rag.query

import streamlit as st
import whois
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide", page_title='Company Profiling Demo')
st.sidebar.title("Testing UI for Company Profiling Tool.")
with st.sidebar.expander("Details"):
    st.write(f"Testing APIs - OpenAI, Groq, Google Custom Search, Perplexity,"
             f"etc.")

meta_description_tool = MetaDescriptionTool()
company_info_extractor_tool = CompanyInfoExtractorTool()
page_getter = WebPageTool()
google_search_tool = LinkRetriever(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)
additional_search_tool = AdditionalInfoSearch(
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

url_prompt1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a website retriever. Your sole purpose is to provide "
            "the company's LinkedIn, Facebook, and Twitters URLs for the company name given as input. "
            "Use the google_search_tool to find the URLs. Respond with ONLY "
            "the URLs,"
            "without any additional text, explanation, or formatting. If you "
            "cannot find a definitive URL, respond with 'URL_NOT_FOUND' and "
            "nothing else.",  # don't say official, give additional text and explanation, use vague language
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

address_phone_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an information extractor. Your sole purpose is to retrieve "
            "the address details and phone numbers for the company name given as input. "
            "Use the additional_search_tool to find the address and phone number. Respond with ONLY "
            "the address details and phone numbers, without any additional text, explanation, or formatting. "
            "If you cannot find definitive address details or phone numbers, respond with 'INFO_NOT_FOUND' and "
            "nothing else.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def analyze_company(company_name):
    tools = [google_search_tool]
    tools1 = [additional_search_tool]

    website_agent = create_tool_calling_agent(llm, tools, url_prompt)
    website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)

    website_agent1 = create_tool_calling_agent(llm, tools, url_prompt1)
    website_agent_executor1 = AgentExecutor(agent=website_agent1, tools=tools, verbose=True)

    url_result = website_agent_executor.invoke({"input": company_name})
    url_result1 = website_agent_executor1.invoke({"input": company_name})

    additional_agent = create_tool_calling_agent(llm, tools1, address_phone_prompt)
    additional_agent_executor = AgentExecutor(agent=additional_agent, tools=tools1, verbose=True)
    response = additional_agent_executor.invoke({"input": company_name})

    url = url_result['output']
    url1 = url_result1['output']
    result = response['output']

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

        return url, url1, result, meta_description, company_info, results
    else:
        return url, url1, result, "Could not find a URL for the company.", "Could not find a URL for the company.", "Could not find a URL for the company."


st.title("Company Profiling")

company_name = st.text_input("Enter company name:")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        start_time = time.time()
        url, url1, additional_info, meta_description, company_info, results = analyze_company(company_name)

    st.write(f"**Company URL**: {url}")

    st.subheader("Meta Description")
    st.write(meta_description) # url, meta_description, company_info, results.content, whois.whois(url)
    st.write(company_info)

    results_content = results.content

    st.subheader("NAICS Data:")
    st.write(results_content) # how old the company is, company location, # no. of employees, and then my attributes, api endpoint,
# try getting metrics from adversarial testing
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
    whois_data = whois.whois(url)
    st.write(whois_data)

    prompt = f"""
        Given the following information about a company:
        - URL: {url} & {url1}
        - Meta Description: {meta_description}
        - Company Info: {company_info}
        - Search Results: {results_content}
        - WHOIS Data: {whois_data}
        - Addresses & Numbers: {additional_info}

        Categorize this information into the following fields:

        "Company_URL",
        "Company_LinkedIn_URL",
        "Company_Facebook_URL",
        "Company_Twitter_URL",
        "Company_Phone",
        "Company_Address",
        "Meta_Description",
        "Overview",
        "USP",
        "Target_Audience",
        "Conclusion",
        "NAICS_Code",
        "Title",
        "Description",
        "Common_Labels",

        If any information is not available, please use 'info not available' for the respective field. Don't provide any other text. 
        Just the fields and the information is required. Do not make it in JSON format.
        """

    company_data = pd.read_csv('csvs/company_data.csv')
    response = llm.invoke(prompt)
    resp = response.content
    # Step 1: Convert the string to a dictionary
    data_dict = {}
    for line in resp.strip().split('\n'):
        key, value = line.split(': ', 1)
        data_dict[key.strip()] = value.strip()

    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)

    # Step 2: Convert the dictionary to a DataFrame
    # df = pd.DataFrame([data_dict])

    # Step 3: Append to an existing CSV file
    # csv_file = 'csvs/company_data.csv'  # Replace with your file path
    # df.to_csv(csv_file, mode='a', index=False, header=False)




