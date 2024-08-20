from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

chat = ChatPerplexity(
    temperature=0, pplx_api_key=os.getenv("PPLX_API_KEY"), model="llama-3.1-sonar-large-128k-online"
)

system = """You are a helpful assistant. Given a company name, provide the NAICS code and the source URL of the information of it and nothing else."""

human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Accent Health Recruitment"})
print(response.content)