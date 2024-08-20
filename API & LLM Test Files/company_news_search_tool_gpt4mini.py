from googleapiclient.discovery import build
from langchain.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from typing import Dict, List
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


class CompanyNewsSearch(BaseTool):
    name = "Company_News_Search"
    description = "Searches for company news, events, and updates given a company name and website."
    api_key: str = Field(..., description="Google API key")
    cse_id: str = Field(..., description="Google Custom Search Engine ID")

    def _run(self, company_name: str, website: str) -> List[Dict[str, str]]:
        # Define search query
        query = f"{company_name} (events OR launches OR blogs OR updates OR news) site:{website}"

        # Perform search
        search_engine = build("customsearch", "v1", developerKey=self.api_key)
        result = search_engine.cse().list(q=query, cx=self.cse_id).execute()

        # Extract relevant information
        news_items = [{"title": item["title"], "link": item["link"]} for item in result.get("items", [])]

        return news_items

    async def _arun(self, company_name: str, website: str) -> List[Dict[str, str]]:
        return self._run(company_name, website)


# Initialize the tool with required API keys
google_search_tool = CompanyNewsSearch(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

# Define tools and LLM
tools = [google_search_tool]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)

# Define the prompt template
openai_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a news and updates retriever. Your task is to provide recent news, events, podcasts, blogs, updates, launches, or other relevant information about the company name and website "
            "given as input. Use the google_search_tool to find this information. Respond with a list of relevant titles, URLs, and a brief description or summary of each. The summary should "
            "provide context about the update, such as the main points or why it is significant. If you cannot find any relevant information, respond with 'NO_UPDATES_FOUND' and nothing else."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create agent and executor
website_agent = create_tool_calling_agent(llm, tools, openai_prompt)
website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=False)

# Invoke the agent
r = website_agent_executor.invoke({"input": "AB Staffing Solutions, https://www.abstaffing.com/"})
r = r['output']

print(r)
