from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@db/moest_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ForecastResult(Base):
    __tablename__ = "forecast_results"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer)  # Incremented per run
    year = Column(Integer)
    region = Column(String(100))
    council = Column(String(100))
    form_num = Column(Integer)
    subject = Column(String(100))
    enrollment_govt = Column(Integer)
    enrollment_all = Column(Integer)

def init_db():
    Base.metadata.create_all(bind=engine)