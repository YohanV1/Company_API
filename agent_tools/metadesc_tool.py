from langchain.tools import BaseTool
import requests
from bs4 import BeautifulSoup


class MetaDescriptionTool(BaseTool):
    name = "meta_description_tool"
    description = "Extracts the meta description from a given URL."

    def _run(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_description = soup.find('meta', attrs={'name': 'description'})
        return meta_description['content'] if meta_description else "No meta description found."

    def _arun(self, url: str) -> str:
        raise NotImplementedError("This tool does not support async")