import os
import re
from io import BytesIO
from datetime import timedelta
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# --- External LLM, Retrieval, and DB Clients (from first file) ---
from huggingface_hub import login
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from tavily import TavilyClient
from pymongo import MongoClient
from openai import OpenAI

# --- RDKit for molecule drawing (from second file) ---
from rdkit import Chem
from rdkit.Chem import Draw

# --- Authentication and Database modules (from second file) ---
from database import get_db, create_tables, User
from auth import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Load environment variables
load_dotenv()

# --- Setup for External Services (RAG, LLM, etc.) ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NUM_LINKS_THRESH = 2

# MongoDB setup for feedback
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
chat_db = mongo_client["chatApp"]

ANSWER_OUTPUT_LENGTH = 1024  # Maximum number of tokens to generate for the answer
MAX_LOOPS = 2  # Maximum number of loops to run the LLM

hf_api_keys = os.getenv("HF_API_KEY")
login(hf_api_keys)

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ses-papers-textbooks-for-rag"
index = pc.Index(index_name)
print(index.describe_index_stats())

# --- Utility Functions from the First File ---
def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination: alpha * dense + (1 - alpha) * sparse."""
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

def dense_sparse_vector(query):
    de = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=query,
        parameters={"input_type": "passage", "truncate": "END"}
    )    
    se = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=query,
        parameters={"input_type": "passage", "return_tokens": True}
    )
    return de[0]['values'], {'indices': se[0]['sparse_indices'], 'values': se[0]['sparse_values']}

def retrieve_context(query, top_k_chunks, rag_enabled: bool, web_search_enabled: bool, web_search_client: str):
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
                    sources.append(f'- <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>')
        elif web_search_client == "OpenAI":
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                messages=[{"role": "user", "content": query}],
            )
            openai_response = completion.choices[0].message.content
            links = re.findall(r'\(\[([^\]]+)\]\(([^)]+)\)\)', openai_response)
            if len(links) >= NUM_LINKS_THRESH:
                for text, url in links:
                    url_split = url.split("?utm_source=openai")[0]
                    sources.append(f"- {text}: {url_split}")
                context += f"Web search result: {openai_response}\n\n"
    
    if rag_enabled:
        query_payload = {
            "inputs": {"text": f"{query}"},
            "top_k": top_k_chunks
        }
        results = index.search(namespace="ses_rag", query=query_payload)
        for i, hit in enumerate(results['result']['hits']):
            context_text = hit['fields']['context']
            full_source = hit['fields']['source'].split("/")[-1].split(".jsonl")[0]
            citation = ""
            print(full_source)
            if full_source[-2:] == "-0":
                doi = full_source[:-2].replace("_", "/")
                try:
                    headers = {"Accept": "text/x-bibliography; style=apa"}
                    response = httpx.get(f"https://doi.org/{doi}", headers=headers, follow_redirects=True)
                    if response.status_code == 200:
                        citation = response.text.strip().split("https://doi.org")[0]
                        source_text = f'{citation} <a href="https://doi.org/{doi}" target="_blank" rel="noopener noreferrer">https://doi.org/{doi}</a>'
                    else:
                        source_text = f"DOI: {doi}"
                except Exception as e:
                    source_text = f"DOI: {doi}"
            else:
                pattern = "(z-lib"
                idx = full_source.lower().find(pattern)
                if idx != -1:
                    source_text = full_source[:idx]
                else:
                    source_text = full_source
                citation = source_text

            if citation:
                context += f"Database result {i+1}, from {citation}: {context_text}\n\n"
            else:
                context += f"Database result {i+1}: {context_text}\n\n"
            sources.append("- " + source_text)
    
    # Remove duplicate sources while preserving order
    sources = list(dict.fromkeys(sources))
    sources = "\n".join(sources)
    return context, sources

async def query_llm(prompt: str, model: str, max_output_len: int = 1024) -> str:
    """
    Query the LLM inference server with the given prompt.
    For model "o3-mini", it uses OpenAI's chat API; for "OmniScience" it uses a local inference URL.
    """
    if model == "o3-mini":
        final_response = openai_client.chat.completions.create(
            model="o3-mini",
            reasoning_effort="high",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
    elif model == "OmniScience":
        inference_url = "http://localhost:8800/v2/models/llama/infer"
        final_response = ""
        current_prompt = prompt
        count = 0

        while True:
            length = max_output_len if count == 0 else ANSWER_OUTPUT_LENGTH
            payload = {
                "inputs": [
                    {
                        "name": "prompts",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [[current_prompt]]
                    },
                    {
                        "name": "max_output_len",
                        "shape": [1, 1],
                        "datatype": "INT64",
                        "data": [[length]]
                    },
                    {
                        "name": "output_generation_logits",
                        "shape": [1, 1],
                        "datatype": "BOOL",
                        "data": [[False]]
                    }
                ],
                "outputs": [{"name": "outputs"}]
            }

            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(inference_url, json=payload)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code,
                                    detail=f"LLM query failed: {response.text}")

            result = response.json()
            try:
                new_output = result["outputs"][0]["data"][0]
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="LLM query returned no output")

            final_response += new_output

            if "<|eot_id|>" in final_response:
                final_response = final_response.replace("<|eot_id|>", "")
                break

            if "<|start_header_id|>answer" not in new_output and count == 0:
                final_response += "\n<|start_header_id|>answer\n"

            current_prompt = prompt + "\n" + final_response
            count += 1
            if count >= MAX_LOOPS:
                break
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified")
    return final_response

# --- Initialize SQLAlchemy Database ---
create_tables()

# --- Create FastAPI App Instance ---
app = FastAPI()

# --- Configure CORS Middleware ---
allowed_origins = list(set([
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
    os.getenv("CORS_ORIGIN", "http://localhost:3000")
]))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models for RAG/LLM Endpoints ---
class RagRequest(BaseModel):
    query: str
    maxOutputLength: int = 2048
    ragEnabled: bool = True
    webSearchEnabled: bool = True
    webSearchClient: str = "OpenAI"
    model: str = "o3-mini"

class FeedbackRequest(BaseModel):
    isPositive: bool
    feedbackText: str
    responseContent: Optional[str] = ""
    collapsibleContent: Optional[str] = ""
    timestamp: Optional[str] = None

# --- Endpoints from the First File ---

@app.post("/api/feedback")
async def save_feedback(feedback: FeedbackRequest):
    if not feedback.feedbackText or feedback.feedbackText.strip() == "":
        raise HTTPException(status_code=400, detail="Feedback text is required")
    try:
        collection = chat_db["feedback"]
        result = collection.insert_one({
            "isPositive": feedback.isPositive,
            "feedbackText": feedback.feedbackText,
            "responseContent": feedback.responseContent,
            "collapsibleContent": feedback.collapsibleContent,
            "timestamp": feedback.timestamp or httpx._utils.get_timestamp()
        })
        return {"message": "Feedback saved successfully", "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.post("/rag")
async def handle_rag(query_req: RagRequest):
    query = query_req.query
    max_output_length = query_req.maxOutputLength
    rag_enabled = query_req.ragEnabled
    web_search_enabled = query_req.webSearchEnabled
    web_search_client = query_req.webSearchClient
    model = query_req.model  # either "OmniScience" or "o3-mini"
    
    # Retrieve context from database and/or web search
    context, sources = retrieve_context(query, 3, rag_enabled, web_search_enabled, web_search_client)
    
    # Construct the prompt by combining the query with the retrieved context
    initial_prompt = f"{query}"
    extended_prompt = ""
    if rag_enabled or web_search_enabled:
        if model == "OmniScience":
            extended_prompt += "\n<|start_header_id|>think\nOkay, let me start by searching "
            if rag_enabled:
                extended_prompt += "my database "
            if web_search_enabled:
                if rag_enabled:
                    extended_prompt += "and "
                extended_prompt += "the internet "
            extended_prompt += "for information about this query. I found the following information:\n"
            extended_prompt += context
            extended_prompt += "Let me think about which parts of this information are useful to answer the query."
        elif model == "o3-mini":
            extended_prompt += "\nThe following are"
            if rag_enabled:
                extended_prompt += " database"
            if web_search_enabled:
                if rag_enabled:
                    extended_prompt += " and"
                extended_prompt += " internet"
            extended_prompt += " search results you might find useful when answering the query:\n"
            extended_prompt += context
            extended_prompt += "These results may or may not be relevant. Think about which parts of this information are useful to answer the query."
    prompt = initial_prompt + extended_prompt
    
    # Query the LLM with the combined prompt
    llm_response = await query_llm(prompt, model, max_output_length)
    if not llm_response:
        raise HTTPException(status_code=500, detail="LLM query failed")
    
    if model == "OmniScience":
        return {"outputs": extended_prompt + llm_response + "\n**Sources:**\n" + sources}
    else:
        return {"outputs": llm_response + "<br><br><strong>Sources:</strong><br><br>" + sources}

# --- Endpoints from the Second File (Authentication & Molecule) ---

@app.post("/register")
def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    hashed_password = get_password_hash(password)
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        permissions="research",  # Default permission
        query_limit=10  # Set query limit to 10
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": new_user.username,
        "email": new_user.email,
        "permissions": new_user.permissions,
        "query_limit": new_user.query_limit
    }

@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "email": user.email,
        "permissions": user.permissions,
        "query_limit": user.query_limit
    }

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "permissions": current_user.permissions,
        "query_limit": current_user.query_limit
    }

@app.get("/verify-token")
def verify_token(current_user: User = Depends(get_current_user)):
    return {
        "valid": True,
        "username": current_user.username,
        "permissions": current_user.permissions,
        "query_limit": current_user.query_limit
    }

@app.get("/query_limit")
def get_query_limit(current_user: User = Depends(get_current_user)):
    """
    Returns the current user's query limit.
    """
    return {
        "username": current_user.username,
        "query_limit": current_user.query_limit
    }

@app.post("/query_limit_update")
def update_query_limit(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Decrements the user's query limit by 1 and updates it in the database.
    """
    if current_user.query_limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query limit already at minimum"
        )
    
    # Decrement the query limit
    current_user.query_limit -= 1
    
    # Update in database
    db.commit()
    
    return {
        "username": current_user.username,
        "query_limit": current_user.query_limit,
        "message": "Query limit decremented successfully"
    }

@app.get("/molecule")
async def get_molecule(smiles: str = Query(..., title="SMILES String")):
    """
    Generate a molecular drawing from a SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}
        img = Draw.MolToImage(mol)
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return Response(content=img_bytes.getvalue(), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

# --- Run the Combined Application ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
