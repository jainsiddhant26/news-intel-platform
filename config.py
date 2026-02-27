"""Configuration management for News Intelligence Platform."""

import os
from typing import List

from dotenv import load_dotenv


class Config:
    """Configuration class using python-dotenv for environment variables."""
    
    def __init__(self) -> None:
        """Initialize configuration by loading environment variables."""
        load_dotenv()
        
        # Required API keys
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.news_api_key: str = os.getenv("NEWS_API_KEY", "")
        
        # Validate required API keys
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY environment variable is required")
        
        # Optional configuration with defaults
        monitored_tickers_str: str = os.getenv("MONITORED_TICKERS", "AAPL,GOOGL,MSFT,TSLA")
        self.monitored_tickers: List[str] = [
            ticker.strip() for ticker in monitored_tickers_str.split(",") if ticker.strip()
        ]
        
        self.alert_sentiment_threshold: str = os.getenv("ALERT_SENTIMENT_THRESHOLD", "negative")
        self.alert_impact_threshold: str = os.getenv("ALERT_IMPACT_THRESHOLD", "high")
        self.refresh_interval_seconds: int = int(os.getenv("REFRESH_INTERVAL_SECONDS", "300"))
    
    @property
    def GROQ_API_KEY(self) -> str:
        """Get Groq API key."""
        return self.groq_api_key
    
    @property
    def NEWS_API_KEY(self) -> str:
        """Get News API key."""
        return self.news_api_key
    
    @property
    def MONITORED_TICKERS(self) -> List[str]:
        """Get list of monitored tickers."""
        return self.monitored_tickers
    
    @property
    def ALERT_SENTIMENT_THRESHOLD(self) -> str:
        """Get alert sentiment threshold."""
        return self.alert_sentiment_threshold
    
    @property
    def ALERT_IMPACT_THRESHOLD(self) -> str:
        """Get alert impact threshold."""
        return self.alert_impact_threshold
    
    @property
    def REFRESH_INTERVAL_SECONDS(self) -> int:
        """Get refresh interval in seconds."""
        return self.refresh_interval_seconds


# Export singleton instance
config = Config()