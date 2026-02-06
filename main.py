from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import database as db
from model_logic import Pipeline
import uvicorn

app = FastAPI()

def get_db():
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()

@app.on_event("startup")
def startup():
    db.init_db()

@app.post("/run-forecast")
def run_model_batch(db_session: Session = Depends(get_db)):
    # Increment Batch ID
    max_batch = db_session.query(func.max(db.ForecastRecord.batch_id)).scalar() or 0
    new_batch = max_batch + 1
    
    # Run ML Pipeline
    pipeline = Pipeline()
    df_results = pipeline.run()
    
    if df_results is not None:
        # Save to MySQL
        records = [
            db.ForecastRecord(
                batch_id=new_batch,
                year=int(r['YEAR']),
                region=r['REGION'],
                council=r['COUNCIL'],
                subject=r['SUBJECT'],
                form_num=int(r['FORM_NUM']),
                enrollment_govt=int(r['ENROLLMENT_Government']),
                enrollment_all=int(r['ENROLLMENT_All'])
            ) for _, r in df_results.iterrows()
        ]
        db_session.bulk_save_objects(records)
        db_session.commit()
        return {"status": "success", "batch": new_batch, "rows": len(records)}
    return {"status": "failed"}

@app.get("/forecasts")
def get_forecasts(
    region: str = None, 
    council: str = None, 
    year: int = None, 
    subject: str = None,
    db_session: Session = Depends(get_db)
):
    query = db_session.query(db.ForecastRecord)
    if region: query = query.filter(db.ForecastRecord.region == region.upper())
    if council: query = query.filter(db.ForecastRecord.council == council.upper())
    if year: query = query.filter(db.ForecastRecord.year == year)
    if subject: query = query.filter(db.ForecastRecord.subject == subject.upper())
    return query.all()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)