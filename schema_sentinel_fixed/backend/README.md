# Backend (FastAPI)

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Notes
- Uses SQLite by default: `backend/app.db`
- User uploads canonical schema + business metadata + baseline CSV via `/v1/datasets/register`
- Drift events go into `drift_events` table
- Batches routed to:
  - `production_rows` (if low risk)
  - `staging_rows` (if medium/high risk)

## API Docs
Open: http://127.0.0.1:8000/docs
