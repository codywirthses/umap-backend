# UMAP App Backend Server

This is a simple FastAPI backend for user authentication with SQLite database for the UMAP application.

## Setup

1. Make sure you have Python 3.7+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):
   - The application uses a `.env` file in the server directory
   - You can modify this file to change settings like the port, database URL, and JWT secret
   - Default values will be used if no .env file is found

4. Run the server:
```bash
python main.py
```
or
```bash
uvicorn main:app --reload
```

The server will start at http://localhost:8000 by default (configurable in .env)

## Environment Variables

The following environment variables can be configured in the `.env` file:

```
# Database
DATABASE_URL=sqlite:///./users.db   # SQLite database location
MONGO_URI=mongodb://localhost:27017 # MongoDB connection string for feedback storage

# JWT Authentication
SECRET_KEY=your_secret_key          # Secret key for JWT token generation
ALGORITHM=HS256                     # Algorithm used for JWT
ACCESS_TOKEN_EXPIRE_MINUTES=30      # Token expiration time in minutes

# Server Configuration
FRONTEND_URL=http://localhost:3000  # Frontend URL for CORS
CORS_ORIGIN=http://localhost:3000   # CORS origin

# External Services
OPENAI_API_KEY=your_openai_key      # OpenAI API key for o3-mini model
HF_API_KEY=your_huggingface_key     # HuggingFace API key
TAVILY_API_KEY=your_tavily_key      # Tavily API key for web search
PINECONE_API_KEY=your_pinecone_key  # Pinecone API key for vector database
```

## API Endpoints

### Authentication Endpoints
- `POST /register` - Register a new user
- `POST /login` - Login an existing user
- `GET /users/me` - Get current user information
- `GET /verify-token` - Verify JWT token validity

### RAG (Retrieval Augmented Generation) Endpoints
- `POST /rag` - Query the LLM with context from database and/or web search

### Molecule Related Endpoints
- `GET /molecule` - Generate a molecular drawing from a SMILES string

### Feedback Endpoints
- `POST /api/feedback` - Save user feedback on responses

## Database

The application uses SQLite database (`users.db`) to store user information.

### Database Schema

Table: `users`
- `id` (Integer, Primary Key)
- `username` (String, Unique)
- `email` (String, Unique)
- `hashed_password` (String)
- `permissions` (String, Default: "basic")

### Querying the Database

You can use SQLite CLI or any SQLite browser like DB Browser for SQLite to access the database:

```bash
# Open the database
sqlite3 users.db

# List all tables
.tables

# Show table schema
.schema users

# Query all users
SELECT * FROM users;

# Query user by username
SELECT * FROM users WHERE username = 'your_username';

# Update user permissions
UPDATE users SET permissions = 'admin' WHERE username = 'your_username';

# Delete a user
DELETE FROM users WHERE username = 'your_username';
```

## Permission Levels

The application supports the following permission levels:
- `basic`: Regular user access
- `premium`: Premium features access
- `admin`: Administrative access

By default, all new users are assigned the `basic` permission level.

## Security Notes

For production, make sure to:
1. Change the `SECRET_KEY` in the `.env` file to a secure random string
2. Configure proper CORS settings in the `.env` file
3. Set up HTTPS for secure communication 
