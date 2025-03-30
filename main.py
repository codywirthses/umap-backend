from fastapi import FastAPI, Depends, HTTPException, status, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
from fastapi.responses import JSONResponse
from typing import Optional
import os
from dotenv import load_dotenv
from io import BytesIO

# Database and authentication modules
from database import get_db, create_tables, User
from auth import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# RDKit for molecule drawing
from rdkit import Chem
from rdkit.Chem import Draw

# Load environment variables
load_dotenv()

# Initialize the database
create_tables()

# Create FastAPI app instance
app = FastAPI()

# Configure CORS middleware
# Allow origins from both environment variables (if provided), otherwise default to localhost:3000
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

# Authentication endpoints
@app.post("/register")
def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if user already exists
    db_user = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        permissions="basic"  # Default permission
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
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

# Molecule endpoint
@app.get("/molecule")
async def get_molecule(smiles: str = Query(..., title="SMILES String")):
    """Generate a molecular drawing from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}

        # Generate molecular image
        img = Draw.MolToImage(mol)

        # Convert image to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return Response(content=img_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
