import base64
import os
import re
from io import BytesIO
from datetime import timedelta
import traceback
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from pymongo import MongoClient

# --- Snowflake Client ---
from snowflake.snowpark import Session as SnowflakeSession

# --- RDKit for molecule drawing (from second file) ---
from rdkit import Chem
from rdkit.Chem import Draw

load_dotenv()

# --- Authentication and Database modules (from second file) ---
from database import get_db, create_tables, User
from auth import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from rag import retrieve_context
from chat import query_llm, extract_molecule_names
from util import get_smiles, smiles_to_image, replace_greek_letters

snowflake_auth = {
    "account": "SESAI-MAIN",
    "user": os.getenv("SNOWFLAKE_USER"),
    "authenticator": "externalbrowser",
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": "MATERIAL_WH",
    "database": "UMAP_DATA",
    "schema": "PUBLIC"
}

snowflake_session = SnowflakeSession.builder.configs(snowflake_auth).create()


# MongoDB setup for feedback
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
chat_db = mongo_client["chatApp"]

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
    model = query_req.model  # either OmniScience or o3-mini
    # Run retrieval based on the enabled options
    context, sources = await retrieve_context(query, 3, rag_enabled, web_search_enabled, web_search_client)
    # Construct the prompt by combining the query and retrieval results
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
            extended_prompt += " database search results you might find useful when answering the query:\n"
            extended_prompt += context
            extended_prompt += "These results may or may not be relevant. "
            extended_prompt += "Think about which parts of this information are useful to answer the query. "
            extended_prompt += "If you name specific molecules in your response, make sure to state the full molecule name before using any abbreviations.\n"
    prompt = initial_prompt + extended_prompt
    # Query your LLM with the combined prompt
    llm_response = await query_llm(prompt, model, max_output_length)
    if not llm_response:
        raise HTTPException(status_code=500, detail="LLM query failed")
    
    # Extract molecule names from the LLM response
    original_molecule_list = await extract_molecule_names(llm_response)
    molecule_text = ""
    if original_molecule_list:
        molecule_text = "; ".join(original_molecule_list)
    
    processed_molecule_list = [replace_greek_letters(mol) for mol in original_molecule_list]

    if model == "OmniScience":
        return {"outputs": extended_prompt + llm_response + "\n**Sources:**\n" + sources, "molecules": processed_molecule_list}
    else:
        return {"outputs": llm_response + "<br><br><strong>Sources:</strong><br><br>" + sources +"<br><br><strong>Detected molecules in LLM response:</strong><br><br>" + molecule_text, "molecules": processed_molecule_list}

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
        permissions="research"  # Default permission
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
        "permissions": new_user.permissions
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
        "permissions": user.permissions
    }

@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "permissions": current_user.permissions
    }

@app.get("/verify-token")
def verify_token(current_user: User = Depends(get_current_user)):
    return {
        "valid": True,
        "username": current_user.username,
        "permissions": current_user.permissions
    }

@app.get("/api/find_smiles")
async def find_smiles(molecule: str):
    result = await get_smiles(molecule)
    return result

@app.get("/molecule")
async def get_molecule(smiles: str = Query(..., title="SMILES String")):
    """
    Generate a molecular drawing from a SMILES string.
    """
    try:
        return Response(content=smiles_to_image(smiles), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/api/molecule_details")
async def molecule_details(molecule: str):
    """
    Query the UMAP_DATA.PUBLIC.UMAP_1M_MOLECULAR table for the given molecule (by SMILES).
    If found, return its properties and a base64-encoded image generated by smiles_to_image.
    """
    print(f"Searching for {molecule}...")
    smiles = await get_smiles(molecule)
    if not smiles or smiles["smiles"] == "Not found" or "Error" in smiles["smiles"]:
        return {"found": False, "message": "Molecule not found"}

    query = f'''
        SELECT SMILE, 
            ROUND(HOMO, 2) as HOMO,
            ROUND(LUMO, 2) as LUMO,
            ROUND(ESP_MAX, 2) as ESP_MAX,
            ROUND(ESP_MIN, 2) as ESP_MIN,
            CAST(ENERGY AS INT) as ENERGY
        FROM UMAP_DATA.PUBLIC.UMAP_1M_MOLECULAR
        WHERE SMILE = '{smiles["smiles"]}'
        LIMIT 1
    '''
    print("Constructed query: ", query)
    try:
        result_df = snowflake_session.sql(query).to_pandas()
        if result_df.empty:
            print("Molecule not found.")
            return {"found": False, "message": "Molecule not found"}
        row = result_df.iloc[0].to_dict()
        # Generate molecule image
        image_data = smiles_to_image(row['SMILE'])
        if image_data:
            b64_image = base64.b64encode(image_data).decode('utf-8')
            image_uri = f"data:image/png;base64,{b64_image}"
        else:
            image_uri = None
        row['image'] = image_uri
        row['name'] = molecule
        return {"found": True, "molecule_details": row}
    except Exception as e:
        print("Error querying molecule details:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error querying molecule details: {str(e)}")


# --- Run the Combined Application ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
