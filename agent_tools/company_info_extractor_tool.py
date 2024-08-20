from langchain.tools import BaseTool
from bs4 import BeautifulSoup
import requests


class CompanyInfoExtractorTool(BaseTool):
    name = "company_info_extractor_tool"
    description = "Extracts company information (About Us, Company, History, etc.) from a given URL and its subpages."

    def _run(self, url: str) -> str:
        visited = set()
        to_visit = [url]
        company_info = []

        while to_visit and len(visited) < 5:  # Limit to 5 pages to avoid excessive crawling
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue

            visited.add(current_url)
            try:
                response = requests.get(current_url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for relevant content
                relevant_content = self._extract_relevant_content(soup)
                if relevant_content:
                    company_info.append(relevant_content)

                # Find more relevant links
                self._add_relevant_links(soup, current_url, to_visit)

            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")

        return "\n\n".join(company_info) if company_info else "No relevant company information found."

    def _extract_relevant_content(self, soup):
        relevant_tags = ['about', 'company', 'history', 'mission', 'vision', 'values', 'team', 'leadership']
        for tag in relevant_tags:
            elements = soup.find_all(['div', 'section', 'article', 'p'],
                                     class_=lambda x: x and tag in x.lower())
            elements += soup.find_all(['div', 'section', 'article'],
                                      id=lambda x: x and tag in x.lower())

            if elements:
                return " ".join(element.get_text(strip=True) for element in elements)

        return None

    def _add_relevant_links(self, soup, base_url, to_visit):
        relevant_keywords = ['about', 'company', 'history', 'mission', 'vision', 'values', 'team', 'leadership']
        for a in soup.find_all('a', href=True):
            link = a['href']
            if any(keyword in link.lower() for keyword in relevant_keywords):
                full_url = urljoin(base_url, link)
                if full_url not in to_visit:
                    to_visit.append(full_url)

    def _arun(self, url: str) -> str:
        raise NotImplementedError("This tool does not support async")
