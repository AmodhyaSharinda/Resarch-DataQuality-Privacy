from __future__ import annotations
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Schema Drift Backend"
    ENV: str = "dev"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DATABASE_URL: str = "mssql+pyodbc://@LAPTOP-DDQ61IAS/SchemaDriftDB?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"
    
    STORAGE_DIR: str = "storage"

    RENAME_MODEL_PATH: str | None = None
    RENAME_FEATURES_PATH: str | None = None
    RENAME_THRESHOLD_PATH: str | None = None

    EMB_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Comma-separated list
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
