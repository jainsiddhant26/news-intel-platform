"""Sentiment analysis agent for financial news using FinBERT and Groq."""

from typing import Dict

import torch
from langchain_groq import ChatGroq
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import config


class SentimentAgent:
    """Agent for analyzing sentiment and market impact of financial news."""
    
    def __init__(self) -> None:
        """Initialize the sentiment agent with FinBERT model and Groq LLM."""
        # Initialize FinBERT model locally
        try:
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print(f"FinBERT model loaded on {self.device}")
        except Exception as e:
            print(f"Failed to load FinBERT model: {e}")
            self.tokenizer = None
            self.model = None
            self.device = None
        
        # Initialize Groq LLM for market impact analysis
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
        Analyze sentiment and market impact of a news article.
        
        Args:
            article: Article dictionary with title, content, and other fields
            
        Returns:
            Enriched article dictionary with sentiment, sentiment_score, and market_impact fields.
        """
        # Run FinBERT sentiment analysis
        sentiment, sentiment_score = self._analyze_finbert_sentiment(article)
        
        # Run Groq market impact analysis
        market_impact = self._analyze_market_impact(article)
        
        # Add sentiment analysis to article
        article.update({
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "market_impact": market_impact
        })
        
        return article
    
    def _analyze_finbert_sentiment(self, article: Dict[str, str]) -> tuple[str, float]:
        """
        Analyze sentiment using FinBERT model.
        
        Args:
            article: Article dictionary
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not self.model or not self.tokenizer:
            print("FinBERT model not available, returning neutral sentiment")
            return "neutral", 0.5
        
        try:
            # Prepare text: title + first 512 characters of content
            title = article.get("title", "")
            content = article.get("content", "")
            text = f"{title} {content[:512]}"
            
            # Tokenize and predict
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Map FinBERT classes to sentiment labels
            # FinBERT classes: 0=negative, 1=neutral, 2=positive
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_map.get(predicted_class, "neutral")
            
            return sentiment, confidence
            
        except Exception as e:
            print(f"Error in FinBERT sentiment analysis: {e}")
            return "neutral", 0.5
    
    def _analyze_market_impact(self, article: Dict[str, str]) -> str:
        """
        Analyze market impact using Groq LLM.
        
        Args:
            article: Article dictionary
            
        Returns:
            Market impact rating: high/medium/low
        """
        if not self.llm:
            print("Groq client not available, returning medium impact")
            return "medium"
        
        try:
            # Prepare text for analysis
            title = article.get("title", "")
            content = article.get("content", "")
            company_ticker = article.get("company_ticker", "UNKNOWN")
            topic = article.get("topic", "other")
            
            prompt = f"""
            Analyze the market impact of this financial news article and rate it as high, medium, or low.
            
            Consider these factors:
            - High impact: Major earnings surprises, significant mergers/acquisitions, regulatory changes, macroeconomic shocks
            - Medium impact: Regular earnings reports, moderate partnerships, standard economic data
            - Low impact: Routine news, minor updates, speculative information
            
            Article details:
            Title: {title}
            Content: {content[:500]}
            Company: {company_ticker}
            Topic: {topic}
            
            Respond with only one word: high, medium, or low
            """
            
            response = self.llm.invoke(prompt)
            impact = response.content.strip().lower()
            
            # Validate and normalize impact
            valid_impacts = ["high", "medium", "low"]
            return impact if impact in valid_impacts else "medium"
            
        except Exception as e:
            print(f"Error analyzing market impact: {e}")
            return "medium"