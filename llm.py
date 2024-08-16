from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests
import os

load_dotenv()

os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")


def google_search(query, api_key, cse_id, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'num': num_results,
    }

    response = requests.get(url, params=params)
    results = response.json().get('items', [])

    links = [item['link'] for item in results]
    return links


search = SerpAPIWrapper()

serp_search_tool = StructuredTool.from_function(
    name="Search",
    func=search.run,
    description="Use SerpAPI to search the web"
)

tools = [serp_search_tool]

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a website retriever. Your sole purpose is to provide "
#             "the official website URL for the company name given as input. "
#             "Use the SERP API to find the URL. Respond with ONLY the URL, "
#             "without any additional text, explanation, or formatting. If you "
#             "cannot find a definitive URL, respond with 'URL_NOT_FOUND' and "
#             "nothing else.",
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an English to German translator. Your sole purpose is to translate "
            "the English text given as input into German. Provide only the German "
            "translation, without any additional text, explanation, or formatting. "
            "If you cannot translate the text, respond with 'TRANSLATION_NOT_POSSIBLE' "
            "and nothing else.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_executor.invoke({"input": "I love chickens"}))