import whois
from datetime import datetime
from langchain.tools import BaseTool


class WhoisTool(BaseTool):
    name = "whois_tool"
    description = "Retrieves the age of a website using WHOIS data."

    def _run(self, url: str) -> str:
        domain = url.split("//")[-1].split("/")[0]
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        age = (datetime.now() - creation_date).days // 365
        return f"The website is approximately {age} years old."

    def _arun(self, url: str) -> str:
        raise NotImplementedError("This tool does not support async")
