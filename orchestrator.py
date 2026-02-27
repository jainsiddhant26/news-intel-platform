"""Orchestrator for the News Intelligence Platform using LangGraph state machine."""

from typing import Annotated, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.classification_agent import ClassificationAgent
from agents.collection_agent import CollectionAgent
from agents.sentiment_agent import SentimentAgent
from agents.synthesis_agent import SynthesisAgent
from agents.verification_agent import VerificationAgent
from config import config


class NewsState(TypedDict):
    """State definition for the news processing pipeline."""
    articles: List[Dict[str, str]]
    current_index: int
    processed_articles: List[Dict[str, str]]
    alerts: List[Dict[str, str]]


class NewsOrchestrator:
    """Orchestrator for managing the news intelligence pipeline using LangGraph."""
    
    def __init__(self) -> None:
        """Initialize the orchestrator with all agents and build the workflow."""
        try:
            # Initialize all agents
            self.collection_agent = CollectionAgent()
            self.classification_agent = ClassificationAgent()
            self.sentiment_agent = SentimentAgent()
            self.verification_agent = VerificationAgent()
            self.synthesis_agent = SynthesisAgent()
            
            # Build the workflow
            self.workflow = self._build_workflow()
            
            print("News orchestrator initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize orchestrator: {e}")
            self.workflow = None
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for news processing.
        
        Returns:
            Compiled StateGraph workflow.
        """
        try:
            # Create workflow
            workflow = StateGraph(NewsState)
            
            # Add nodes
            workflow.add_node("collect", self._collect_node)
            workflow.add_node("classify", self._classify_node)
            workflow.add_node("sentiment", self._sentiment_node)
            workflow.add_node("verify", self._verify_node)
            workflow.add_node("synthesize", self._synthesize_node)
            
            # Add edges
            workflow.set_entry_point("collect")
            workflow.add_edge("collect", "classify")
            workflow.add_conditional_edges(
                "classify",
                self._should_continue_processing,
                {
                    "continue": "sentiment",
                    "end": "verify"
                }
            )
            workflow.add_conditional_edges(
                "sentiment",
                self._should_continue_processing,
                {
                    "continue": "synthesize",
                    "end": "verify"
                }
            )
            workflow.add_conditional_edges(
                "synthesize",
                self._should_continue_processing,
                {
                    "continue": "classify",
                    "end": "verify"
                }
            )
            workflow.add_edge("verify", END)
            
            # Compile workflow
            return workflow.compile()
            
        except Exception as e:
            print(f"Error building workflow: {e}")
            return None
    
    def run_pipeline(self, tickers: List[str] = None) -> Dict[str, any]:
        """
        Run the complete news processing pipeline.
        
        Args:
            tickers: List of tickers to monitor (uses config default if None)
            
        Returns:
            Dictionary with processed articles, alerts, and total count.
        """
        if not self.workflow:
            print("Workflow not available")
            return {"processed": [], "alerts": [], "total": 0}
        
        try:
            # Use provided tickers or config defaults
            monitored_tickers = tickers if tickers else config.MONITORED_TICKERS
            
            # Initialize state
            initial_state = {
                "articles": [],
                "current_index": 0,
                "processed_articles": [],
                "alerts": []
            }
            
            print(f"Starting pipeline for tickers: {monitored_tickers}")
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract alerts from processed articles
            alerts = self._extract_alerts(result["processed_articles"])
            
            final_result = {
                "processed": result["processed_articles"],
                "alerts": alerts,
                "total": len(result["processed_articles"])
            }
            
            print(f"Pipeline completed: {final_result['total']} articles processed, {len(alerts)} alerts generated")
            return final_result
            
        except Exception as e:
            print(f"Error running pipeline: {e}")
            return {"processed": [], "alerts": [], "total": 0}
    
    def _collect_node(self, state: NewsState) -> NewsState:
        """Collection node: fetch news articles."""
        try:
            print("📰 Collecting news articles...")
            articles = self.collection_agent.run()
            state["articles"] = articles
            state["current_index"] = 0
            print(f"Collected {len(articles)} articles")
            return state
        except Exception as e:
            print(f"Error in collection node: {e}")
            state["articles"] = []
            return state
    
    def _classify_node(self, state: NewsState) -> NewsState:
        """Classification node: classify current article."""
        try:
            if state["current_index"] >= len(state["articles"]):
                return state
            
            article = state["articles"][state["current_index"]]
            print(f"🏷️  Classifying article {state['current_index'] + 1}/{len(state['articles'])}: {article.get('title', '')[:50]}...")
            
            classified_article = self.classification_agent.run(article)
            state["articles"][state["current_index"]] = classified_article
            return state
        except Exception as e:
            print(f"Error in classification node: {e}")
            return state
    
    def _sentiment_node(self, state: NewsState) -> NewsState:
        """Sentiment node: analyze sentiment of current article."""
        try:
            if state["current_index"] >= len(state["articles"]):
                return state
            
            article = state["articles"][state["current_index"]]
            print(f"😊 Analyzing sentiment for article {state['current_index'] + 1}...")
            
            sentiment_article = self.sentiment_agent.run(article)
            state["articles"][state["current_index"]] = sentiment_article
            return state
        except Exception as e:
            print(f"Error in sentiment node: {e}")
            return state
    
    def _verify_node(self, state: NewsState) -> NewsState:
        """Verification node: verify all articles at once."""
        try:
            print("✅ Verifying articles across sources...")
            verified_articles = self.verification_agent.run(state["articles"])
            state["processed_articles"] = verified_articles
            print(f"Verification completed for {len(verified_articles)} articles")
            return state
        except Exception as e:
            print(f"Error in verification node: {e}")
            state["processed_articles"] = state["articles"]
            return state
    
    def _synthesize_node(self, state: NewsState) -> NewsState:
        """Synthesis node: synthesize current article."""
        try:
            if state["current_index"] >= len(state["articles"]):
                return state
            
            article = state["articles"][state["current_index"]]
            print(f"🔍 Synthesizing article {state['current_index'] + 1}...")
            
            synthesized_article = self.synthesis_agent.run(article)
            state["articles"][state["current_index"]] = synthesized_article
            
            # Add to processed articles
            state["processed_articles"].append(synthesized_article)
            return state
        except Exception as e:
            print(f"Error in synthesis node: {e}")
            return state
    
    def _should_continue_processing(self, state: NewsState) -> str:
        """Determine if we should continue processing articles."""
        if state["current_index"] < len(state["articles"]) - 1:
            state["current_index"] += 1
            return "continue"
        else:
            return "end"
    
    def _extract_alerts(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract alerts from processed articles based on alert levels."""
        alerts = []
        try:
            for article in articles:
                alert_level = article.get("alert_level", "GREEN")
                if alert_level in ["RED", "YELLOW"]:
                    alert = {
                        "title": article.get("title", ""),
                        "alert_level": alert_level,
                        "sentiment": article.get("sentiment", ""),
                        "market_impact": article.get("market_impact", ""),
                        "company_ticker": article.get("company_ticker", ""),
                        "summary": article.get("summary", ""),
                        "source": article.get("source", ""),
                        "verified": article.get("verified", False)
                    }
                    alerts.append(alert)
            
            # Sort by alert level (RED first)
            alerts.sort(key=lambda x: 0 if x["alert_level"] == "RED" else 1)
            
        except Exception as e:
            print(f"Error extracting alerts: {e}")
        
        return alerts