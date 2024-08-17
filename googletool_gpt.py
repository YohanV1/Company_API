from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import Field
from typing import List
from bs4 import BeautifulSoup
import requests
import os

load_dotenv()

os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

llm1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
    max_retries=2,
)


class GoogleSearchTool(BaseTool):
    name = "google_search"
    description = "Useful only for retrieving website links and nothing else."
    api_key: str = Field(..., description="Google API key")
    cse_id: str = Field(..., description="Google Custom Search Engine ID")

    def _run(self, query: str, num_results: int = 5) -> List[str]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': num_results,
        }

        response = requests.get(url, params=params)
        results = response.json().get('items', [])

        links = [item['link'] for item in results]
        return links

    def _arun(self, query: str):
        # This tool does not support async, so we just call the sync version
        return self._run(query)


class WebPageTool(BaseTool):
    name = "get_webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")


page_getter = WebPageTool()

google_search_tool = GoogleSearchTool(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

tools = [google_search_tool]

openai_prompt = ChatPromptTemplate.from_messages(
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

website_agent = create_tool_calling_agent(llm, tools, openai_prompt)
website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)
url_result = website_agent_executor.invoke({"input": "SRM Institute of Science and Technology, Kattankulathur"})
url = url_result['output']
print(url)

if url != "URL_NOT_FOUND":
    company_analyzer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a company analyzer. Your purpose is to determine if the "
            "given company is healthcare-related. "
            "Use the page_getter tool to find information about the "
            "company. Then, provide a response in the "
            "following format: 'Answer: [Yes/No/Maybe], Reason: [Brief "
            "explanation]'. Do not include any additional "
            "information or formatting. If you cannot find enough "
            "information, respond with 'Answer: Maybe, "
            "Reason: Insufficient information available.'"
        ),
        ("placeholder", "{chat_history}"),
        ("human", f"Analyze this company: {url}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [page_getter]

    company_analyzer_agent = create_tool_calling_agent(llm, tools,
                                                       company_analyzer_prompt)
    company_analyzer_executor = AgentExecutor(agent=company_analyzer_agent,
                                              tools=tools, verbose=True)

    analysis_result = company_analyzer_executor.invoke(
        {"input": f"Analyze this company: {url}"})
    print(analysis_result['output'])
else:
    print("Could not find a URL for the company.")
