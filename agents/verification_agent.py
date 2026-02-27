"""Verification agent for cross-validating news articles across multiple sources."""

import re
from typing import Dict, List, Set

from config import config


class VerificationAgent:
    """Agent for verifying news articles by cross-referencing multiple sources."""
    
    def __init__(self) -> None:
        """Initialize the verification agent."""
        self.similarity_threshold = 0.6  # Minimum similarity for articles to be considered related
    
    def run(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Verify articles by grouping similar stories and marking verification status.
        
        Args:
            articles: List of article dictionaries with title, content, url, source, etc.
            
        Returns:
            Updated articles list with verification fields added.
        """
        try:
            # Group articles by title similarity
            article_groups = self._group_by_similarity(articles)
            
            # Process each group and add verification metadata
            verified_articles = []
            for group in article_groups:
                if len(group) >= 2:
                    # Multiple sources - mark as verified
                    for article in group:
                        article["verified"] = True
                        article["source_count"] = len(group)
                        article["unconfirmed_reason"] = ""
                        verified_articles.append(article)
                    print(f"Verified story with {len(group)} sources: {group[0]['title'][:50]}...")
                else:
                    # Single source - mark as unconfirmed
                    article = group[0]
                    article["verified"] = False
                    article["source_count"] = 1
                    article["unconfirmed_reason"] = "Only one source reporting this story"
                    verified_articles.append(article)
            
            print(f"Verification complete: {len(verified_articles)} articles processed")
            return verified_articles
            
        except Exception as e:
            print(f"Error during verification: {e}")
            # Return original articles with default verification status
            for article in articles:
                article["verified"] = False
                article["source_count"] = 1
                article["unconfirmed_reason"] = "Verification process failed"
            return articles
    
    def _group_by_similarity(self, articles: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """
        Group articles by title keyword similarity.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of article groups where each group contains similar articles.
        """
        groups = []
        used_indices = set()
        
        for i, article1 in enumerate(articles):
            if i in used_indices:
                continue
                
            current_group = [article1]
            used_indices.add(i)
            
            # Find similar articles
            for j, article2 in enumerate(articles):
                if j in used_indices or i == j:
                    continue
                
                similarity = self._calculate_title_similarity(
                    article1.get("title", ""), 
                    article2.get("title", "")
                )
                
                if similarity >= self.similarity_threshold:
                    current_group.append(article2)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using word overlap.
        
        Args:
            title1: First article title
            title2: Second article title
            
        Returns:
            Similarity score between 0 and 1.
        """
        try:
            # Normalize titles: lowercase, remove punctuation, split into words
            words1 = self._normalize_title(title1)
            words2 = self._normalize_title(title2)
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating title similarity: {e}")
            return 0.0
    
    def _normalize_title(self, title: str) -> Set[str]:
        """
        Normalize a title by cleaning and extracting meaningful words.
        
        Args:
            title: Article title to normalize
            
        Returns:
            Set of normalized words.
        """
        try:
            # Convert to lowercase and remove punctuation
            cleaned = re.sub(r'[^\w\s]', ' ', title.lower())
            
            # Split into words and filter out common stop words
            words = set(word.strip() for word in cleaned.split() if len(word.strip()) > 2)
            
            # Remove common financial news stop words that don't add meaning
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
                'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see',
                'two', 'way', 'who', 'boy', 'did', 'get', 'let', 'put', 'say', 'she', 'too', 'use', 'news',
                'report', 'reports', 'says', 'said', 'market', 'markets', 'stock', 'stocks', 'trading', 'trade',
                'financial', 'finance', 'business', 'economy', 'economic', 'company', 'companies', 'inc', 'corp',
                'llc', 'ltd', 'co', 'group', 'shares', 'share', 'price', 'prices', 'up', 'down', 'rise', 'fall'
            }
            
            meaningful_words = words - stop_words
            return meaningful_words
            
        except Exception as e:
            print(f"Error normalizing title: {e}")
            return set()