from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

chat = ChatPerplexity(
    temperature=0, pplx_api_key=os.getenv("PPLX_API_KEY"), model="llama-3.1-sonar-large-128k-online"
)

system = """You are a helpful assistant. Given a company name, provide the following information:

Recent events: (Provide at least 10 news items on the company. Each item MUST include a source URL citation for the news, blog, podcast, or interview from between 2023 and 2024. If you can't find 
exactly 10 recent items with valid source links, provide as many as you can find.)

Provide only the requested information in the specified format. Do not include any additional details or unsourced information."""

human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Recent news/events/updates/podcasts/interviews/launches/blogs/etc with source URL for Accent Health Recruitment"})
print(response.content)