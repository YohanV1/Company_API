from langchain.tools import BaseTool


class NAICSCodeIdentifierTool(BaseTool):
    name = "naics_code_identifier"
    description = "Identifies the most appropriate NAICS code for a given company description."

    def _run(self, company_description: str) -> str:
        return query_rag(company_description)

    def _arun(self, company_description: str) -> str:
        raise NotImplementedError("This tool does not support async operations")
