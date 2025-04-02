from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from tavily import TavilyClient
from openai import OpenAI
import re
import os
import pandas as pd
import requests

# min number of links to accept 4o-mini-search results
NUM_LINKS_THRESH = 2
NUM_TAVILY_SERACH_RESULTS = 3
NUM_RAG_RESULTS = 3

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ses-papers-textbooks-for-rag"
index = pc.Index(index_name)
print(index.describe_index_stats())

def find_full_doi(doi_suffix, excel_file_path):
    try:
        # Load the Excel file
        df = pd.read_excel(excel_file_path, engine='openpyxl')

        # Find the row where the DOI suffix matches the end part of 'doi' column
        df['doi'] = df['doi'].str.lower()
        match = df[df['doi'].str.endswith(doi_suffix, na=False)]

        if not match.empty:
            return match.iloc[0]['doi']
        else:
            return manually_get_full_doi(doi_suffix)
            #f"No matching full DOI found in DB for suffix: {doi_suffix}"

    except Exception as e:
        return f"An error occurred: {e}"

def get_paper_title(full_doi):
    url = f"https://api.crossref.org/works/{full_doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        title = data['message'].get('title', ["Title not available"])[0]
        doi_url = f"https://doi.org/{full_doi}"

        return f'<a href="{doi_url}" target="_blank" rel="noopener noreferrer">{title}</a> (DOI: {full_doi})'
    else:
        return f'- {full_doi}'
    
def extract_title_doi_from_filename(filename):

    # List of regex patterns to match different filename formats
    patterns = [
        r'/llm_data/papers/rag_papers/9300LIB/(.+)-0\.jsonl$',
        r'/llm_data/papers/rag_papers/Gyuleen_update/(.+)-0\.jsonl$',
        r'/llm_data/papers/3p6_jsonl/(.+)-0\.jsonl$',
        r'/llm_data/papers/textbooks_jsonl/(.+)-0\.jsonl$'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            extracted = match.group(1)
            if "9300LIB" in pattern:
                extracted = find_full_doi(extracted, './data/9300 doi_for_RAG.xlsx')
                title_doi_link = f'- Paper from DB: {get_paper_title(extracted)}'
            if "Gyuleen_update" in pattern:
                extracted = find_full_doi(extracted, './data/Gyuleen_update.xlsx')
                title_doi_link = f'- Paper from DB: {get_paper_title(extracted)}'
            # If the pattern matches the 3p6_jsonl format, replace underscores with slashes
            if "3p6_jsonl" in pattern:
                extracted = extracted.replace("_", "/")
                title_doi_link = f'- Paper from DB: {get_paper_title(extracted)}'
            # If the pattern matches the textbooks_jsonl format, convert hyphens to spaces and capitalize each word
            if "textbooks_jsonl" in pattern:
                title_doi_link = f'- Textbook: {extracted.replace("-", " ").title()}'

            return title_doi_link
    return None

def manually_get_full_doi(suffix):
    # DOI prefixes based on common patterns
    prefix_mapping = {
        "D": "10.1039/",
        "j": "10.1016/",
        "/doi.org/10.1007/": "",
        "s": "10.1038/",
        "anie": "10.1002/",
        "aenm": "10.1002/",
        "ange": "10.1002/",
        "chemrxiv": "10.26434/",
        "smsc": "10.1002/",
        "adfm": "10.1002/",
        "1742-6596": "10.1088/",
        "1945-7111": "10.1149/",
        "acsaem": "10.1021/",
        "nsr": "10.1093/",
        "EMD": "10.1109/",
        "elab": "10.1021/",
        "energymater": "10.1016/",
        "acsnano": "10.1021/",
        "s12598": "10.1007/",
        "j.cnki": "10.1007/"
    }

    # Identify the prefix based on the starting pattern
    prefix = next((v for k, v in prefix_mapping.items() if suffix.startswith(k)), "10.1000/")
    # Generate the full DOI
    full_doi = prefix + suffix.replace("/doi.org/", "")
    return full_doi

async def retrieve_context(query, top_k_chunks, rag_enabled: bool, web_search_enabled: bool, web_search_client:str):
    context = ""
    sources = []

    if web_search_enabled:
        if web_search_client == "Tavily":
            tavily_response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=True,
                include_images=False
            )
            if tavily_response.get("answer"):
                context += f"Web search result: {tavily_response['answer']}\n\n"
                for result in tavily_response["results"]:
                    title = result["title"]
                    url = result["url"]
                    url_link = f'- Web Search: <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'

                if url_link not in sources:
                    sources.append(f'{url_link}')

        elif web_search_client == "OpenAI":
            completion = openai_client.chat.completions.create(
                model = "gpt-4o-mini-search-preview",
                messages = [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
            )
            openai_response = completion.choices[0].message.content
            links = re.findall(r'\(\[([^\]]+)\]\(([^)]+)\)\)', openai_response)
            if len(links) >= NUM_LINKS_THRESH:
                for text, url in links:
                    url_split = url.split("?utm_source=openai")[0]
                    sources.append(f"- {text}: {url_split}")
                # Optionally, include the full response in the context as well:
                context += f"Web search result: {openai_response}\n\n"

    if rag_enabled:
        query_payload = {
            "inputs": {
                "text": f"{query}"
            },
            "top_k": top_k_chunks
        }

        results = index.search(
            namespace="ses_rag",
            query=query_payload
        )

        for (i,hit) in enumerate(results['result']['hits']):
            context_text = hit['fields']['context']
            source_text = hit['fields']['source']
            citation = extract_title_doi_from_filename(source_text)

            if citation:
                context += "Database result " + str(i+1) + ", from " + citation + ": " + context_text + f"\n\n"
            else:
                context += "Database result " + str(i+1) + ": " + context_text + "\n\n"

            sources.append(f'{citation}')

    sources = list(dict.fromkeys(sources))
    sources = "\n".join(sources)
    return context, sources