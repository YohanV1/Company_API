from langchain.tools import BaseTool
import requests
from bs4 import BeautifulSoup
import certifi


class MetaDescriptionTool(BaseTool):
    name = "meta_description_tool"
    description = "Extracts the meta description from a given URL."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, verify=certifi.where())
            soup = BeautifulSoup(response.text, 'html.parser')
            meta_description = soup.find('meta', attrs={'name': 'description'})
            return meta_description['content'] if meta_description else "No meta description found."
        except requests.exceptions.RequestException as e:
            return f"Error fetching meta description: {e}"

    def _arun(self, url: str) -> str:
        raise NotImplementedError("This tool does not support async")
