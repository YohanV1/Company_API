import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from naics_rag.embeddings import get_embedding_function

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "naics_rag/chroma"

NAICS_PROMPT_TEMPLATE = """You are an expert in NAICS (North American Industry Classification System) codes. Your task is to analyze the given company description and determine the most appropriate NAICS code based on the information provided in the context below. The context contains relevant excerpts from the NAICS manual.

Context from NAICS manual:
{context}

---

Company Description: {question}

Based on the company description and the NAICS information provided in the context, please determine the most appropriate NAICS code for this company.

Please format your response as follows:
NAICS_Code: [code]
Title: [title of the code]
Description: [brief description]
Common_Labels: [labels that are more commonly used than the title of the code]
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(NAICS_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)
    response_text = model.invoke(prompt)

    return response_text


if __name__ == "__main__":
    query_text = "Testing Company"
    query_rag(query_text)
