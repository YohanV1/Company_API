import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from naics_rag.embeddings import get_embedding_function

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """You are an expert in NAICS (North American Industry Classification System) codes. Your task is to analyze the given company description and determine the most appropriate NAICS 
code based on the information provided in the context below. The context contains relevant excerpts from the NAICS manual.

Context from NAICS manual:
{context}

---

Company Description: {question}

Based on the company description and the NAICS information provided in the context, please:

1. Determine the most appropriate NAICS code for this company.
2. Provide the title of the NAICS code.
3. Give a brief explanation of why this code is the best fit.
4. If possible, suggest a more specific (longer) NAICS code that might apply, if the information allows for it.

Please format your response as follows:
NAICS Code: [code]
Title: [title of the code]
Description: [brief description]
More Specific Code (if applicable): [longer code]
Specific Code Title (if applicable): [title of the more specific code]
Common Labels: [labels that are more commonly used than the title of the code]

If you cannot determine a specific NAICS code based on the given information, please explain why and what additional information would be needed.
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    query_text = "Testing Company"
    query_rag(query_text)
