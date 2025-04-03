from database import engine, SessionLocal, User
from sqlalchemy import text

# Method 1: Using raw SQL
def delete_user_sql(user_id):
    with engine.connect() as conn:
        result = conn.execute(text(f"DELETE FROM users WHERE id = {user_id}"))
        conn.commit()
        return result.rowcount

# Method 2: Using SQLAlchemy ORM
def delete_user_orm(user_id):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            print(f"Found user: {user.username} with email {user.email}")
            db.delete(user)
            db.commit()
            return True
        else:
            print(f"No user found with ID {user_id}")
            return False
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    # Try to delete user with ID 5
    user_id = 5
    
    # First, try using ORM approach
    print(f"Attempting to delete user with ID {user_id} using ORM...")
    if delete_user_orm(user_id):
        print(f"Successfully deleted user with ID {user_id} using ORM")
    else:
        # Fall back to SQL approach
        print(f"Attempting to delete user with ID {user_id} using SQL...")
        rows_deleted = delete_user_sql(user_id)
        print(f"Deleted {rows_deleted} user(s) with ID {user_id} using SQL") 