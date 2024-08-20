from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

chat = ChatPerplexity(
    temperature=0, pplx_api_key=os.getenv("PPLX_API_KEY"), model="llama-3.1-sonar-large-128k-online"
)

system = """You are a company analyzer. Given a company name, your task is to:
1. Find the official website URL for the company.
2. Analyze the company based on its website content.
3. Provide a response in the following format:

URL: [Official website URL]
Meta Description: [Meta description content]
Company Information Summary: [Brief summary of company information, including About Us, History, etc.]
NAICS Code: [Two or three-digit NAICS code]
NAICS Description: [NAICS code description]
Common Label: [Commonly understood label for the company]
Website Age: [Age of the website]

If you cannot find information for any field, respond with 'Information not available' for that field. Do not include any additional information or formatting."""

human = "Analyze this company: {input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "AB Staffing Solutions"})
print(response.content)