# SchemaSentinel (Full Stack)

This bundle contains:

- `backend/` — FastAPI + SQLAlchemy drift detection API + Kafka consumer worker
- `frontend/` — Vite + React UI (ported from your HTML prototype)

## 1) Backend setup

### Create a new SQL Server database (recommended)

Using SSMS:

1. Create a database (example): `SchemaSentinelDB`
2. Create a SQL login/user and grant db_owner (or at least create table + read/write)

### Configure backend `.env`

Copy:

```bash
cp .env.example .env
```

Edit `DATABASE_URL` and CORS:

```env
DATABASE_URL=mssql+pyodbc://USER:PASSWORD@HOST:1433/SchemaSentinelDB?driver=ODBC+Driver+17+for+SQL+Server
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPICS=ingest
WINDOW_SECONDS=5
MAX_BUFFER_ROWS=5000
MIN_ROWS_FOR_DRIFT=5
```

> Tip: Set `WINDOW_SECONDS=3` for faster demo.

> Tip: If your batch files have fewer rows, reduce `MIN_ROWS_FOR_DRIFT` (e.g. 1–10).

### Install & run backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Backend: `http://127.0.0.1:8000`

## 2) Kafka setup

Start your Kafka + Zookeeper (docker compose or local install). Make sure:

- Bootstrap server matches `KAFKA_BOOTSTRAP_SERVERS`
- Topic exists (default: `ingest`)

## 3) Start the Kafka consumer worker

In a second terminal:

```bash
cd backend
.\.venv\Scripts\activate
python -m app.workers.kafka_consumer
```

## 4) Frontend setup

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Frontend: `http://localhost:5173`

## 5) Demo flow (matches your prototype)

1. Go to **Datasets**
   - Upload Canonical schema JSON
   - Upload Business metadata JSON
   - Upload Baseline CSV/XLSX (optional)
2. Go to **Live Stream**
   - Upload one or more batch CSV/XLSX files
   - Click **Simulate** on a stored batch
3. Watch **📡 Stream Console**
   - Shows messages being produced to Kafka + `end_batch`
4. Open **Drift Events** / **Staging Queue**
   - You’ll see the worker-generated drift events
   - Approve/Reject/Promote from the UI

