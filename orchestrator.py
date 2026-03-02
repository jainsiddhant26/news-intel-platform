"""Simple sequential orchestrator for the News Intelligence Platform."""
from agents.classification_agent import ClassificationAgent
from agents.collection_agent import CollectionAgent
from agents.sentiment_agent import SentimentAgent
from agents.synthesis_agent import SynthesisAgent
from agents.verification_agent import VerificationAgent
from config import config


class NewsOrchestrator:
    """Orchestrates the full news intelligence pipeline."""

    def __init__(self):
        self.collection_agent = CollectionAgent()
        self.classification_agent = ClassificationAgent()
        self.sentiment_agent = SentimentAgent()
        self.synthesis_agent = SynthesisAgent()
        self.verification_agent = VerificationAgent()
        print("News orchestrator initialized successfully")

    def run_pipeline(self, tickers: list = None) -> dict:
        """Run the full pipeline sequentially."""
        if tickers is None:
            tickers = config.MONITORED_TICKERS

        print(f"Starting pipeline for tickers: {tickers}")

        print("📰 Collecting news articles...")
        articles = self.collection_agent.run(tickers)
        print(f"Collected {len(articles)} articles")

        processed = []
        for i, article in enumerate(articles):
            print(f"🏷️  Classifying article {i+1}/{len(articles)}: {article.get('title','')[:50]}...")
            article = self.classification_agent.run(article)
            print(f"😊 Analyzing sentiment for article {i+1}...")
            article = self.sentiment_agent.run(article)
            print(f"🔍 Synthesizing article {i+1}...")
            article = self.synthesis_agent.run(article)
            processed.append(article)

        print("✅ Verifying articles...")
        processed = self.verification_agent.run(processed)

        alerts = [a for a in processed if a.get('alert_level') == 'RED']
        print(f"Pipeline complete: {len(processed)} articles, {len(alerts)} alerts")

        return {"processed": processed, "alerts": alerts, "total": len(processed)}
