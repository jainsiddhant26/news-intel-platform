"""News classification agent for categorizing financial news articles."""

from typing import Dict, List

from langchain_groq import ChatGroq

from config import config


class ClassificationAgent:
    """Agent for classifying news articles by topic, company, and region."""
    
    def __init__(self) -> None:
        """Initialize the classification agent with Groq LLM."""
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=config.GROQ_API_KEY,
                temperature=0.1
            )
        except Exception as e:
            print(f"Failed to initialize Groq client: {e}")
            self.llm = None
    
    def run(self, article: Dict[str, str]) -> Dict[str, str]:
        """
        Classify a news article and add classification fields.
        
        Args:
            article: Article dictionary with title, content, url, source, published_at
            
        Returns:
            Enriched article dictionary with topic, company_ticker, and region fields.
        """
        if not self.llm:
            print("Groq client not available, returning default classifications")
            article.update({
                "topic": "other",
                "company_ticker": "UNKNOWN",
                "region": "GLOBAL"
            })
            return article
        
        try:
            # Prepare text for classification
            text = f"Title: {article.get('title', '')}\nContent: {article.get('content', '')}"
            
            # Classify topic
            topic = self._classify_topic(text)
            
            # Classify company ticker
            company_ticker = self._classify_company_ticker(text)
            
            # Classify region
            region = self._classify_region(text)
            
            # Add classifications to article
            article.update({
                "topic": topic,
                "company_ticker": company_ticker,
                "region": region
            })
            
        except Exception as e:
            print(f"Error during classification: {e}")
            # Fallback to default values
            article.update({
                "topic": "other",
                "company_ticker": "UNKNOWN",
                "region": "GLOBAL"
            })
        
        return article
    
    def _classify_topic(self, text: str) -> str:
        """
        Classify the article topic.
        
        Args:
            text: Article title and content
            
        Returns:
            Topic classification: earnings/macro/regulatory/merger_acquisition/other
        """
        try:
            prompt = f"""
            Classify the following financial news article into one of these topics:
            - earnings: Earnings reports, financial results, quarterly/annual results
            - macro: Macroeconomic news, inflation, interest rates, GDP, unemployment
            - regulatory: Regulatory changes, compliance, legal actions, government policy
            - merger_acquisition: Mergers, acquisitions, takeovers, partnerships
            - other: Any other financial news not fitting the above categories
            
            Article text:
            {text}
            
            Respond with only the topic name (lowercase, no spaces, use underscores).
            """
            
            response = self.llm.invoke(prompt)
            topic = response.content.strip().lower()
            
            # Validate and normalize topic
            valid_topics = ["earnings", "macro", "regulatory", "merger_acquisition", "other"]
            return topic if topic in valid_topics else "other"
            
        except Exception as e:
            print(f"Error classifying topic: {e}")
            return "other"
    
    def _classify_company_ticker(self, text: str) -> str:
        """
        Classify the company ticker mentioned in the article.
        
        Args:
            text: Article title and content
            
        Returns:
            Best matching ticker from monitored list or "UNKNOWN"
        """
        try:
            monitored_tickers_str = ", ".join(config.MONITORED_TICKERS)
            
            prompt = f"""
            Identify the primary company ticker mentioned in this financial news article.
            If multiple tickers are mentioned, choose the most prominent one.
            If no monitored ticker is mentioned, respond with "UNKNOWN".
            
            Monitored tickers: {monitored_tickers_str}
            
            Article text:
            {text}
            
            Respond with only the ticker symbol (e.g., "AAPL") or "UNKNOWN".
            """
            
            response = self.llm.invoke(prompt)
            ticker = response.content.strip().upper()
            
            # Validate ticker
            if ticker in config.MONITORED_TICKERS:
                return ticker
            else:
                return "UNKNOWN"
                
        except Exception as e:
            print(f"Error classifying company ticker: {e}")
            return "UNKNOWN"
    
    def _classify_region(self, text: str) -> str:
        """
        Classify the geographic region of the article.
        
        Args:
            text: Article title and content
            
        Returns:
            Region classification: US/EU/APAC/GLOBAL
        """
        try:
            prompt = f"""
            Classify the geographic region focus of this financial news article:
            - US: United States focused news
            - EU: European Union / Europe focused news
            - APAC: Asia-Pacific focused news
            - GLOBAL: Global or multi-regional news
            
            Article text:
            {text}
            
            Respond with only the region code (US, EU, APAC, or GLOBAL).
            """
            
            response = self.llm.invoke(prompt)
            region = response.content.strip().upper()
            
            # Validate and normalize region
            valid_regions = ["US", "EU", "APAC", "GLOBAL"]
            return region if region in valid_regions else "GLOBAL"
            
        except Exception as e:
            print(f"Error classifying region: {e}")
            return "GLOBAL"