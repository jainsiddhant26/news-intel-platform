"""Synthesis agent for generating summaries and alerts from processed news articles."""

from typing import Dict, List

from langchain_groq import ChatGroq

from config import config
from rag.retriever import RAGRetriever


class SynthesisAgent:
    """Agent for synthesizing news articles with historical context and alerts."""
    
    def __init__(self) -> None:
        """Initialize the synthesis agent with Groq LLM and RAG retriever."""
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=config.GROQ_API_KEY,
                temperature=0.1
            )
            print("Groq LLM initialized for synthesis")
        except Exception as e:
            print(f"Failed to initialize Groq client: {e}")
            self.llm = None
        
        try:
            # Initialize RAG retriever for historical context
            self.retriever = RAGRetriever()
            print("RAG retriever initialized for historical context")
        except Exception as e:
            print(f"Failed to initialize RAG retriever: {e}")
            self.retriever = None
    
    def run(self, article: Dict[str, str]) -> Dict[str, str]:
        """
        Synthesize an article with summary, historical context, and alert level.
        
        Args:
            article: Processed article dictionary with all previous analysis fields
            
        Returns:
            Article dictionary enriched with summary, historical_context, and alert_level.
        """
        if not self.llm:
            print("Groq client not available, returning default synthesis")
            article.update({
                "summary": "• Unable to generate summary\n• LLM service unavailable\n• Please check configuration",
                "historical_context": "Historical context unavailable due to system limitations",
                "alert_level": "YELLOW"
            })
            return article
        
        try:
            # Get historical context from RAG
            rag_results = self._get_historical_context(article)
            
            # Build synthesis prompt
            prompt = self._build_synthesis_prompt(article, rag_results)
            
            # Generate synthesis
            response = self.llm.invoke(prompt)
            synthesis_text = response.content.strip()
            
            # Parse response
            summary, historical_context, alert_level = self._parse_synthesis_response(synthesis_text)
            
            # Add synthesis to article
            article.update({
                "summary": summary,
                "historical_context": historical_context,
                "alert_level": alert_level
            })
            
            return article
            
        except Exception as e:
            print(f"Error during synthesis: {e}")
            # Fallback values
            article.update({
                "summary": "• Error occurred during synthesis\n• Please try again later\n• Check system logs for details",
                "historical_context": "Historical context unavailable due to processing error",
                "alert_level": "YELLOW"
            })
            return article
    
    def _get_historical_context(self, article: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Retrieve historical context using RAG retriever.
        
        Args:
            article: Article dictionary
            
        Returns:
            List of RAG results with historical context.
        """
        if not self.retriever or not self.retriever.is_available():
            print("RAG retriever not available for historical context")
            return []
        
        try:
            title = article.get("title", "")
            if not title:
                return []
            
            # Retrieve relevant historical documents
            rag_results = self.retriever.retrieve(title, top_k=3)
            return rag_results
            
        except Exception as e:
            print(f"Error retrieving historical context: {e}")
            return []
    
    def _build_synthesis_prompt(self, article: Dict[str, str], rag_results: List[Dict[str, str]]) -> str:
        """
        Build the synthesis prompt with article data and historical context.
        
        Args:
            article: Article dictionary
            rag_results: List of historical context results
            
        Returns:
            Formatted prompt string.
        """
        # Extract article information
        title = article.get("title", "")
        content = article.get("content", "")[:500]  # First 500 chars
        sentiment = article.get("sentiment", "neutral")
        market_impact = article.get("market_impact", "medium")
        verified = article.get("verified", False)
        company_ticker = article.get("company_ticker", "UNKNOWN")
        topic = article.get("topic", "other")
        
        # Format historical context
        rag_text = ""
        if rag_results:
            rag_text = "\nHistorical Context:\n"
            for i, result in enumerate(rag_results, 1):
                rag_text += f"{i}. {result['content'][:200]}...\n"
        
        prompt = f"""
        Analyze this financial news article and provide synthesis:
        
        Article Title: {title}
        Content: {content}
        Company: {company_ticker}
        Topic: {topic}
        Sentiment: {sentiment}
        Market Impact: {market_impact}
        Verified: {verified}
        {rag_text}
        
        Provide a response in this exact format:
        
        SUMMARY:
        • [First key point about the article]
        • [Second key point about the article] 
        • [Third key point about the article]
        
        HISTORICAL_CONTEXT:
        [One sentence linking this to similar past events or patterns]
        
        ALERT_LEVEL:
        [RED/YELLOW/GREEN based on these rules:
        - RED: negative sentiment AND high market impact
        - YELLOW: negative sentiment AND medium market impact
        - GREEN: positive or neutral sentiment OR low impact]
        """
        
        return prompt
    
    def _parse_synthesis_response(self, response: str) -> tuple[str, str, str]:
        """
        Parse the synthesis response into structured components.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Tuple of (summary, historical_context, alert_level).
        """
        try:
            # Initialize defaults
            summary = "• Unable to parse summary\n• Please check response format\n• Default summary provided"
            historical_context = "Historical context could not be parsed"
            alert_level = "YELLOW"
            
            # Parse summary
            if "SUMMARY:" in response:
                summary_section = response.split("SUMMARY:")[1].split("HISTORICAL_CONTEXT:")[0].strip()
                if summary_section:
                    summary = summary_section
            
            # Parse historical context
            if "HISTORICAL_CONTEXT:" in response:
                context_section = response.split("HISTORICAL_CONTEXT:")[1].split("ALERT_LEVEL:")[0].strip()
                if context_section:
                    historical_context = context_section
            
            # Parse alert level
            if "ALERT_LEVEL:" in response:
                alert_section = response.split("ALERT_LEVEL:")[1].strip()
                alert_level = alert_section.upper()
                if alert_level not in ["RED", "YELLOW", "GREEN"]:
                    alert_level = "YELLOW"
            
            return summary, historical_context, alert_level
            
        except Exception as e:
            print(f"Error parsing synthesis response: {e}")
            return summary, historical_context, alert_level