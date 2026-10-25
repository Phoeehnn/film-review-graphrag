from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore', case_sensitive=False)
    neo4j_uri: str = 'bolt://localhost:17687'
    neo4j_user: str = 'neo4j'
    neo4j_password: str = ''
    neo4j_database: str = 'neo4j'
    deepseek_api_key: str = ''
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: Literal['deepseek-flash'] = 'deepseek-flash'
    embedding_model: Literal['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'] = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embedding_revision: str = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
    embedding_device: Literal['cpu'] = 'cpu'
    vector_backend: Literal['neo4j','chroma'] = 'neo4j'
    vector_index: str = Field(default='comment_embedding',pattern=r'^[A-Za-z_][A-Za-z_0-9]*$')
    data_dir: Path = ROOT / 'export' / 'neo4j'
    embeddings_path: Path = ROOT / 'export_gpu' / 'embeddings.npy'
    preprocessed_path: Path = ROOT / 'export_gpu' / 'df_preprocessed.csv'
    chroma_dir: Path = ROOT / 'var' / 'chroma'
    api_base_url: str = 'http://127.0.0.1:18000'
    database_timeout: float = Field(default=10, gt=0, le=10)
    query_timeout: float = Field(default=90, gt=0, le=90)
    agent_timeout: float = Field(default=120, gt=0, le=120)
    model_timeout: float = Field(default=35, gt=0)
    generation_temperature: float = 0.0
    generation_max_tokens: int = 1400
    max_agent_rounds: int = Field(default=5, ge=1, le=5)
    max_agent_tools: int = Field(default=10, ge=1, le=10)
    max_cypher_rewrites: int = Field(default=2, ge=0, le=2)
    session_dir: Path = ROOT / 'var' / 'sessions'
    session_recent_turns: int = Field(default=3, ge=1, le=6)
    session_context_bytes: int = Field(default=8000, ge=500, le=8000,
        validation_alias=AliasChoices('session_context_bytes','SESSION_CONTEXT_BYTES','session_context_tokens','SESSION_CONTEXT_TOKENS'))
    session_summary_chars: int = Field(default=1200, ge=200, le=3000)
    session_max_turns: int = Field(default=24, ge=8, le=100)
    redis_url: str = 'redis://127.0.0.1:16379/0'
    cache_enabled: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)

    @property
    def session_context_tokens(self):
        """Legacy key; the budget has always measured UTF-8 bytes, not tokens."""
        return self.session_context_bytes


@lru_cache
def get_settings() -> Settings:
    return Settings()
