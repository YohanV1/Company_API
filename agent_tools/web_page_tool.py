from langchain.tools import BaseTool
from bs4 import BeautifulSoup
import requests


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