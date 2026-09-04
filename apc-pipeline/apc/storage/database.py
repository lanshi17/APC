from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
Base = declarative_base()
def get_engine(url: str | None = None):
    url = url or os.getenv("APC_DB_URL", "sqlite:///../data/apc.db")
    return create_engine(url, echo=False, future=True)
def get_session(engine=None):
    engine = engine or get_engine()
    return sessionmaker(bind=engine)()
