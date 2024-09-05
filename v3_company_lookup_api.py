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

combined_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a comprehensive company information retriever. Your purpose is to provide the following information for the company name given as input:\n"
     "1. Official website URL\n"
     "2. LinkedIn URL\n"
     "3. Facebook URL\n"
     "4. Twitter URL\n"
     "Use the google_search_tool to find the URLs."
     "Respond in the following format:\n\n"
     "Company_URL: [insert URL here]\n"
     "Company_LinkedIn_URL: [insert URL here]\n"
     "Company_Facebook_URL: [insert URL here]\n"
     "Company_Twitter_URL: [insert URL here]\n"
     "If you cannot find definitive information for any item, respond with 'NOT_FOUND' for that specific item."),
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

        website_agent = create_tool_calling_agent(llm, tools, combined_prompt)
        website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)
        website_links = website_agent_executor.invoke({"input": company_name})

        output = website_links['output']
        lines = output.strip().split('\n')
        company_dict = {}
        for line in lines[:1]:
            key, value = line.split(': ', 1)
            company_dict[key] = value.strip()

        url = company_dict['Company_URL']

        if url != "URL_NOT_FOUND":
            combined_company_analysis_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a comprehensive company analyzer. Your purpose is to provide detailed information about a given company based on its website content. Use the meta_description_tool to fetch the meta description and the page_getter tool for other company information. Respond in the following format:\n\n"
                 "Meta_Description: [insert meta description here]\n"
                 "Company_Phone: [insert phone number here]\n"
                 "Company_Address: [insert address here]\n"
                 "Overview: [insert overview here]\n"
                 "USP: [insert unique selling proposition here]\n"
                 "Target_Audience: [insert target audience here]\n"
                 "Conclusion: [insert conclusion here]\n\n"
                 "If you cannot find enough information for any field, respond with 'Information not available' for that specific field."),
                ("placeholder", "{chat_history}"),
                ("human", "Analyze this company: {url}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            company_tools = [meta_description_tool, page_getter]
            company_agent = create_tool_calling_agent(llm, company_tools, combined_company_analysis_prompt)
            company_agent_executor = AgentExecutor(agent=company_agent, tools=company_tools, verbose=True)
            company_result = company_agent_executor.invoke({"url": url})
            company_info = company_result['output']

            results = naics_rag.query.query_rag(company_info)

            return output, company_info, results
        else:
            return output, "Could not find a URL for the company.", "Could not find a URL for the company."

    output, company_info, results = analyze_company(company_name)
    text = output+'\n'+company_info+'\n'+results.content
    lines = text.strip().split('\n')

    data_dict = {}

    for line in lines:
        if ':' in line:
            key, value = line.split(":", 1)
            data_dict[key.strip()] = value.strip()

    end_time = time.time()
    execution_time = end_time - start_time
    data_dict['execution_time'] = execution_time

    return CompanyInfo(**data_dict)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



