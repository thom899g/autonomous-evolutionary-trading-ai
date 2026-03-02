"""
Configuration management for Evolutionary Trading AI
Uses Pydantic for validation and environment variable loading
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum
import json

class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class ExchangeConfig(str, Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"

class Settings(BaseSettings):
    """Main configuration settings"""
    
    # Trading Configuration
    TRADING_MODE: TradingMode = Field(default=TradingMode.PAPER)
    EXCHANGE: ExchangeConfig = Field(default=ExchangeConfig.BINANCE)
    SYMBOLS: List[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    TIMEFRAME: str = Field(default="5m")
    
    # RL Model Configuration
    RL_MODEL_TYPE: str = Field(default="PPO")  # PPO, A2C, SAC
    STATE_DIMENSION: int = Field(default=100)
    ACTION_DIMENSION: int = Field(default=3)  # BUY, SELL, HOLD
    GAMMA: float = Field(default=0.99)
    LEARNING_RATE: float = Field(default=0.0003)
    
    # Evolutionary Parameters
    POPULATION_SIZE: int = Field(default=50)
    GENERATIONS: int = Field(default=100)
    MUTATION_RATE: float = Field(default=0.1)
    ELITE_SIZE: int = Field(default=5)
    
    # Risk Management
    MAX_POSITION_SIZE: float = Field(default=0.1)  # 10% of portfolio
    STOP_LOSS_PCT: float = Field(default=0.02)  # 2%
    TAKE_PROFIT_PCT: float = Field(default=0.05)  # 5%
    MAX_DRAWDOWN_PCT: float = Field(default=0.15)  # 15%
    
    # Firebase Configuration
    FIREBASE_PROJECT_ID: str = Field(default="")
    FIREBASE_CREDENTIALS_PATH: str = Field(default="")
    FIRESTORE_COLLECTION: str = Field(default="trading_strategies")
    
    # API Configuration
    API_KEY: Optional[str] = None
    API_SECRET: Optional[str] = None
    
    # System Configuration
    LOG_LEVEL: str = Field(default="INFO")
    DATA_CACHE_TTL: int = Field(default=300)  # 5 minutes
    HEARTBEAT_INTERVAL: int = Field(default=60)  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("FIREBASE_CREDENTIALS_PATH")
    def validate_firebase_creds(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f"Firebase credentials file not found: {v}")
        return v
    
    @validator("SYMBOLS")
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one trading symbol must be specified")
        return v

# Global settings instance
settings = Settings()

def load_strategy_config(strategy_id: str) -> Dict:
    """Load strategy-specific configuration from Firebase"""
    from firebase.firebase_client import FirebaseClient
    client = FirebaseClient()
    try:
        return client.get_strategy_config(strategy_id)
    except Exception as e:
        print(f"Error loading strategy config: {e}")
        return {}