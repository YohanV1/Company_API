from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from agent_tools.web_page_tool import WebPageTool
from agent_tools.link_retrieval_tool import LinkRetriever
from agent_tools.metadesc_tool import MetaDescriptionTool
from agent_tools.company_info_extractor_tool import CompanyInfoExtractorTool
import naics_rag.query
import time

load_dotenv()

app = FastAPI()

meta_description_tool = MetaDescriptionTool()
company_info_extractor_tool = CompanyInfoExtractorTool()
page_getter = WebPageTool()
google_search_tool = LinkRetriever(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)

url_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a website retriever. Your sole purpose is to provide the official website URL for the company name given as input. Use the google_search_tool to find the URL. Respond in the following "
     "format:\n\nCompany_URL: [insert URL here]\n\nIf you cannot find a definitive URL, respond with 'Company_URL: URL_NOT_FOUND'."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

social_media_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a social media URL retriever. Your sole purpose is to provide the company's LinkedIn, Facebook, and Twitter URLs for the company name given as input. Use the google_search_tool to "
     "find the URLs. Respond in the following format:\n\nCompany_LinkedIn_URL: [insert URL here]\nCompany_Facebook_URL: [insert URL here]\nCompany_Twitter_URL: [insert URL here]\n\nIf you cannot "
     "find a definitive URL for any of these, respond with 'URL_NOT_FOUND' for that specific platform."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


class CompanyInfo(BaseModel):
    Company_URL: str
    Company_LinkedIn_URL: str
    Company_Facebook_URL: str
    Company_Twitter_URL: str
    Company_Phone: str
    Company_Address: str
    Meta_Description: str
    Overview: str
    USP: str
    Target_Audience: str
    Conclusion: str
    NAICS_Code: str
    Title: str
    Description: str
    Common_Labels: str
    execution_time: float


@app.get("/lookup/company/{company_name}", response_model=CompanyInfo)
async def lookup_company(company_name: str):
    start_time = time.time()
    def analyze_company(company_name):
        tools = [google_search_tool]

        website_agent = create_tool_calling_agent(llm, tools, url_prompt)
        website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=False)

        website_agent1 = create_tool_calling_agent(llm, tools, social_media_prompt)
        website_agent_executor1 = AgentExecutor(agent=website_agent1, tools=tools, verbose=False)

        url_result = website_agent_executor.invoke({"input": company_name})
        url_result1 = website_agent_executor1.invoke({"input": company_name})

        output = url_result['output']
        url = output.split(': ')[1].strip()
        social_media_url = url_result1['output']

        if url != "URL_NOT_FOUND":
            meta_description_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a company analyzer. Your purpose is to provide the meta description of the given company based on its website content. Use the meta_description_tool. Respond in the "
                 "following format:\n\nMeta_Description: [insert meta description here]\n\nIf you cannot find the meta description, respond with 'Meta_Description: Information not available'."),
                ("placeholder", "{chat_history}"),
                ("human", "Analyze this company: {url}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            meta_tools = [meta_description_tool]
            meta_agent = create_tool_calling_agent(llm, meta_tools, meta_description_prompt)
            meta_agent_executor = AgentExecutor(agent=meta_agent, tools=meta_tools, verbose=False)
            meta_result = meta_agent_executor.invoke({"url": url})
            meta_description = meta_result['output']

            company_info_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a company analyzer. Your purpose is to provide company information. Use the page_getter tool for this. Respond in the following format:\n\nOverview: [insert overview "
                 "here]\nCompany_Phone: [insert phone number here]\nCompany_Address: [insert address here]\nUSP: [insert unique selling proposition here]\nTarget_Audience: [insert target audience here]\nConclusion: [insert conclusion here]\n\nIf you cannot find enough "
                 "information for any field, respond with 'Information not available' for that specific field."),
                ("placeholder", "{chat_history}"),
                ("human", "Analyze this company: {url}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            info_tools = [page_getter]
            info_agent = create_tool_calling_agent(llm, info_tools, company_info_prompt)
            info_agent_executor = AgentExecutor(agent=info_agent, tools=info_tools, verbose=False)
            info_result = info_agent_executor.invoke({"url": url})
            company_info = info_result['output']

            results = naics_rag.query.query_rag(company_info)

            return output, social_media_url, meta_description, company_info, results
        else:
            return output, social_media_url, "Could not find a URL for the company.", "Could not find a URL for the company.", "Could not find a URL for the company."


    url, url1, meta_description, company_info, results = analyze_company(company_name)
    text = url+'\n'+url1+'\n'+meta_description+'\n'+company_info+'\n'+results.content
    lines = text.strip().split('\n')

    # Initialize an empty dictionary
    data_dict = {}

    # Iterate through each line and split into key and value
    for line in lines:
        if ':' in line:
            key, value = line.split(":", 1)  # Split only at the first colon
            data_dict[key.strip()] = value.strip()

    end_time = time.time()
    execution_time = end_time - start_time
    data_dict['execution_time'] = execution_time

    return CompanyInfo(**data_dict)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)



