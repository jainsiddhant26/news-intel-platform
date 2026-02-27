"""News collection agent for fetching financial news from various sources."""

from datetime import datetime
from typing import Dict, List

import feedparser
from newsapi import NewsApiClient

from config import config


class CollectionAgent:
    """Agent for collecting news articles from multiple sources."""
    
    def __init__(self) -> None:
        """Initialize the collection agent with API clients."""
        try:
            self.newsapi_client = NewsApiClient(api_key=config.NEWS_API_KEY)
        except Exception as e:
            print(f"Failed to initialize NewsAPI client: {e}")
            self.newsapi_client = None
        
        # RSS feed URLs for financial news
        self.rss_feeds = {
            "Reuters": "https://www.reuters.com/rssFeed/businessNews",
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
            "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        }
    
    def run(self) -> List[Dict[str, str]]:
        """
        Collect news articles from all configured sources.
        
        Returns:
            List of article dictionaries with title, content, url, source, and published_at.
        """
        all_articles = []
        seen_urls = set()
        
        # Collect from NewsAPI
        if self.newsapi_client:
            try:
                newsapi_articles = self._fetch_from_newsapi()
                print(f"NewsAPI: Found {len(newsapi_articles)} articles")
                for article in newsapi_articles:
                    if article["url"] not in seen_urls:
                        all_articles.append(article)
                        seen_urls.add(article["url"])
            except Exception as e:
                print(f"Error fetching from NewsAPI: {e}")
        
        # Collect from RSS feeds
        for source_name, feed_url in self.rss_feeds.items():
            try:
                rss_articles = self._fetch_from_rss(feed_url, source_name)
                print(f"{source_name} RSS: Found {len(rss_articles)} articles")
                for article in rss_articles:
                    if article["url"] not in seen_urls:
                        all_articles.append(article)
                        seen_urls.add(article["url"])
            except Exception as e:
                print(f"Error fetching from {source_name} RSS: {e}")
        
        print(f"Total unique articles collected: {len(all_articles)}")
        return all_articles
    
    def _fetch_from_newsapi(self) -> List[Dict[str, str]]:
        """
        Fetch articles from NewsAPI for monitored tickers.
        
        Returns:
            List of article dictionaries.
        """
        articles = []
        
        for ticker in config.MONITORED_TICKERS:
            try:
                response = self.newsapi_client.get_everything(
                    q=ticker,
                    language="en",
                    sort_by="publishedAt",
                    page_size=5,
                    page=1
                )
                
                for article in response.get("articles", []):
                    articles.append({
                        "title": article.get("title", ""),
                        "content": article.get("description", "") or article.get("content", ""),
                        "url": article.get("url", ""),
                        "source": "NewsAPI",
                        "published_at": article.get("publishedAt", "")
                    })
            except Exception as e:
                print(f"Error fetching NewsAPI articles for {ticker}: {e}")
        
        return articles
    
    def _fetch_from_rss(self, feed_url: str, source_name: str) -> List[Dict[str, str]]:
        """
        Fetch articles from an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            source_name: Name of the source for labeling
            
        Returns:
            List of article dictionaries.
        """
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:5]:  # Limit to top 5 articles
                # Handle different date formats
                published_at = ""
                if hasattr(entry, 'published'):
                    published_at = entry.published
                elif hasattr(entry, 'updated'):
                    published_at = entry.updated
                
                # Clean up content
                content = ""
                if hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description
                
                articles.append({
                    "title": entry.get("title", ""),
                    "content": content,
                    "url": entry.get("link", ""),
                    "source": source_name,
                    "published_at": published_at
                })
                
        except Exception as e:
            print(f"Error parsing RSS feed from {source_name}: {e}")
        
        return articles