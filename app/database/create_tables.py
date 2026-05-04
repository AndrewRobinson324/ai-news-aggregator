from app.database.connection import engine
from app.database.models import Base

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tables created successfully")
