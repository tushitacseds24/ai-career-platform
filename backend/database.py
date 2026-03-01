from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./career_platform.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)

class Analysis(Base):
    __tablename__ = "analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    target_role = Column(String)
    rule_score = Column(Float)
    ai_score = Column(Float)
    final_score = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)