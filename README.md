# UMAP App Backend Server

This is a FastAPI backend for user authentication with PostgreSQL database for the UMAP application.

## Setup

1. Make sure you have Python 3.7+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and configure PostgreSQL:
   - Install PostgreSQL on your system (if not already installed)
   - Create a new database for the application:
   ```bash
   # Connect to PostgreSQL as superuser
   sudo -u postgres psql
   
   # Create a new database
   CREATE DATABASE umap_db;
   
   # Create a user (if needed)
   CREATE USER postgres WITH PASSWORD 'password';
   
   # Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE umap_db TO postgres;
   
   # Exit PostgreSQL
   \q
   ```

4. Configure environment variables:
   - The application uses a `.env` file in the server directory
   - Update the `DATABASE_URL` in the .env file with your PostgreSQL connection details
   - Default values will be used if no .env file is found

5. Run the server:
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
DATABASE_URL=postgresql://postgres:password@localhost:5432/umap_db  # PostgreSQL connection string

# MongoDB
MONGO_URI=mongodb://localhost:27017 # MongoDB connection string for feedback storage

# JWT Authentication
SECRET_KEY=your_secret_key          # Secret key for JWT token generation
ALGORITHM=HS256                     # Algorithm used for JWT
ACCESS_TOKEN_EXPIRE_MINUTES=30      # Token expiration time in minutes

# Server Configuration
FRONTEND_URL=http://localhost:3000  # Frontend URL for CORS
CORS_ORIGIN=http://localhost:3000   # CORS origin

# Stripe Integration
STRIPE_SECRET_KEY=your_stripe_secret_key      # Stripe secret key for payment processing
STRIPE_WEBHOOK_SECRET=your_stripe_webhook_secret  # Secret for verifying Stripe webhook signatures

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

### Query Limit Endpoints
- `GET /query_limit` - Get the current user's query limit
- `POST /query_limit_update` - Decrement the user's query limit by 1
- `POST /admin/reset_query_limits` - Admin endpoint to reset all users' query limits to 10

### RAG (Retrieval Augmented Generation) Endpoints
- `POST /rag` - Query the LLM with context from database and/or web search

### Payments and Subscription Endpoints
- `POST /webhook` - Stripe webhook endpoint for handling payment events (checkout completions, subscription changes)

### Molecule Related Endpoints
- `GET /molecule` - Generate a molecular drawing from a SMILES string

### Feedback Endpoints
- `POST /api/feedback` - Save user feedback on responses

## Database

The application uses PostgreSQL to store user information.

### Database Schema

Table: `users`
- `id` (Integer, Primary Key)
- `username` (String, Unique)
- `email` (String, Unique)
- `hashed_password` (String)
- `permissions` (String, Default: "research")
- `query_limit` (Integer, Default: 10)
- `created_at` (DateTime, Default: current UTC time)

### Query Limit System

The application implements a query limit system:
- Each user has a `query_limit` field (default: 10)
- The query limit decreases by 1 for each query to the RAG system
- Query limits are automatically reset to 10 every 30 days from the user's creation date
- The reset schedule is managed by an APScheduler background scheduler
- Admin users can manually reset all user query limits using the admin endpoint

### Querying the Database

You can use the `psql` command line tool or a PostgreSQL GUI client like pgAdmin to access the database:

```bash
# Connect to the database
psql -U postgres -h localhost -d umap_db

# List all tables
\dt

# Show table schema
\d users

# Query all users
SELECT * FROM users;

# Query user by username
SELECT * FROM users WHERE username = 'your_username';

# Update user permissions
UPDATE users SET permissions = 'admin' WHERE username = 'your_username';

# Update a user's query limit
UPDATE users SET query_limit = 10 WHERE username = 'your_username';

# Check users with exhausted query limits
SELECT id, username, query_limit FROM users WHERE query_limit = 0;

# View all users' query limits and creation dates
SELECT id, username, query_limit, created_at FROM users;

# Delete a user
DELETE FROM users WHERE username = 'your_username';
```

## Permission Levels

The application supports the following permission levels:
- `research`: Regular user access
- `premium`: Premium features access
- `admin`: Administrative access

By default, all new users are assigned the `research` permission level.

## Security Notes

For production, make sure to:
1. Change the `SECRET_KEY` in the `.env` file to a secure random string
2. Configure proper CORS settings in the `.env` file
3. Set up HTTPS for secure communication
4. Use a strong password for your PostgreSQL user
5. Configure PostgreSQL to only accept connections from trusted sources
