from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Get the absolute path of this config.py file
CURRENT_FILE = Path(__file__).resolve()

# 2. Go up 4 levels to hit the true workspace root (RagP1)
# config.py (0) -> core (1) -> app (2) -> backend (3) -> RagP1 (4)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent.parent

# 3. Point to the root .env file
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    GROQ_API_KEY: str
    XAI_API_KEY: str
    
    # Correctly resolves to RagP1/backend/chroma_db
    DATABASE_DIR: str = str(PROJECT_ROOT / "backend" / "chroma_db")
    
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, 
        extra="ignore"
    )

settings = Settings()