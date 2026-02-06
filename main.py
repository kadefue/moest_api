from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import database as db
from model_logic import Pipeline # Your script wrapped in a class
import uvicorn

app = FastAPI(title="MoEST Enrollment Forecast API")

# Dependency
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
def trigger_forecast(db_session: Session = Depends(get_db)):
    # 1. Determine next Batch ID
    last_batch = db_session.query(func.max(db.ForecastResult.batch_id)).scalar()
    current_batch = (last_batch or 0) + 1
    
    # 2. Run the Pipeline
    pipeline = Pipeline()
    # Modify your run() method to return the final_df instead of just saving CSV
    df_results = pipeline.run() 
    
    # 3. Store to MySQL
    records = []
    for _, row in df_results.iterrows():
        record = db.ForecastResult(
            batch_id=current_batch,
            year=int(row['YEAR']),
            region=row['REGION'],
            council=row['COUNCIL'],
            form_num=int(row['FORM_NUM']),
            subject=row['SUBJECT'],
            enrollment_govt=int(row['ENROLLMENT_Government']),
            enrollment_all=int(row['ENROLLMENT_All'])
        )
        records.append(record)
    
    db_session.bulk_save_objects(records)
    db_session.commit()
    
    return {"message": "Forecast completed", "batch_id": current_batch, "records_added": len(records)}

@app.get("/data")
def get_data(
    region: str = None, 
    council: str = None, 
    year: int = None, 
    subject: str = None,
    db_session: Session = Depends(get_db)
):
    query = db_session.query(db.ForecastResult)
    if region: query = query.filter(db.ForecastResult.region == region.upper())
    if council: query = query.filter(db.ForecastResult.council == council.upper())
    if year: query = query.filter(db.ForecastResult.year == year)
    if subject: query = query.filter(db.ForecastResult.subject == subject.upper())
    
    return query.all()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)