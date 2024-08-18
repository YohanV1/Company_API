from langchain.tools import BaseTool
from langchain.pydantic_v1 import Field
from typing import List
import requests


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