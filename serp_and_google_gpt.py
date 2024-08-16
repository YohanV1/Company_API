from langchain.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import Field
from typing import List
import requests
import os

load_dotenv()

os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")


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


search = SerpAPIWrapper()

serp_search_tool = StructuredTool.from_function(
    name="Search",
    func=search.run,
    description="Use the website links to go through which website is most likely to be the official website"
)

google_search_tool = GoogleSearchTool(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

tools = [serp_search_tool, google_search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant tasked with finding official company websites. 
    You have access to two tools:
    1. The 'google_search' tool, which retrieves website links based on a query.
    2. The 'Search' tool, which helps determine the official website from a list of links.

    Your process should be:
    1. Use the 'google_search' tool to get a list of website links for the company.
    2. Use the 'Search' tool to analyze these links and determine the official website.
    3. Provide only the URL of the official website as your final answer, with no additional text."""),
    ("human", "{input}"),
    ("assistant", "Certainly! I'll help you find the official website for the company. Let me use the tools at my disposal to search for it."),
    ("human", "Human: {input}"),
    ("assistant", "Assistant: {agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Find the official website for AB Staffing Solutions",
})
print(result)