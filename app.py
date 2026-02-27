"""Streamlit dashboard for the News Intelligence Platform."""

import time
from datetime import datetime
from typing import Dict, List

import streamlit as st

from config import config
from orchestrator import NewsOrchestrator
from rag.retriever import RAGRetriever


def main() -> None:
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(
        page_title="News Intelligence Platform",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Page title
    st.title("📰 News Intelligence Platform")
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    if st.session_state.show_results:
        render_results()
    else:
        render_welcome()
    
    # Historical query section
    render_historical_query()


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False


def render_sidebar() -> None:
    """Render the sidebar with controls."""
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker selection
    available_tickers = config.MONITORED_TICKERS
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers to Monitor",
        options=available_tickers,
        default=available_tickers,
        help="Choose which stock tickers to monitor for news"
    )
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.toggle(
        "🔄 Auto-refresh (5 min)",
        value=st.session_state.auto_refresh,
        help="Automatically refresh news every 5 minutes"
    )
    
    # Run pipeline button
    if st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        run_news_pipeline(selected_tickers)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh and st.session_state.last_updated:
        time_diff = datetime.now() - st.session_state.last_updated
        if time_diff.total_seconds() >= 300:  # 5 minutes
            run_news_pipeline(selected_tickers)
            st.rerun()
    
    # Last updated timestamp
    if st.session_state.last_updated:
        st.sidebar.info(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Refresh status
    if st.session_state.auto_refresh:
        st.sidebar.success("🔄 Auto-refresh enabled")
        if st.session_state.last_updated:
            time_diff = datetime.now() - st.session_state.last_updated
            remaining = 300 - int(time_diff.total_seconds())
            if remaining > 0:
                st.sidebar.write(f"Next refresh in: {remaining//60}:{remaining%60:02d}")


def run_news_pipeline(tickers: List[str]) -> None:
    """Run the news processing pipeline."""
    try:
        with st.spinner("📰 Fetching and analyzing news..."):
            # Initialize orchestrator
            orchestrator = NewsOrchestrator()
            
            # Run pipeline
            result = orchestrator.run_pipeline(tickers)
            
            # Store results
            st.session_state.processed_data = result
            st.session_state.last_updated = datetime.now()
            st.session_state.show_results = True
            
        st.success(f"✅ Pipeline completed! Processed {result['total']} articles")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error running pipeline: {str(e)}")
        st.session_state.show_results = False


def render_welcome() -> None:
    """Render the welcome screen."""
    st.markdown("""
    ## Welcome to the News Intelligence Platform! 📈
    
    This platform provides real-time financial news analysis with:
    
    - 🤖 **AI-powered sentiment analysis** using FinBERT
    - 🏷️ **Automatic classification** by topic and company
    - ✅ **Cross-source verification** for reliability
    - 🚨 **Smart alerts** for important market events
    - 🔍 **Historical context** with RAG capabilities
    
    ### Getting Started:
    1. **Select tickers** in the sidebar to monitor
    2. **Click "Run Pipeline"** to fetch and analyze news
    3. **View results** with alerts and detailed analysis
    4. **Ask questions** about historical context
    
    Configure your API keys in the `.env` file to get started!
    """)


def render_results() -> None:
    """Render the results page with articles and alerts."""
    if not st.session_state.processed_data:
        st.error("No data available. Please run the pipeline first.")
        return
    
    data = st.session_state.processed_data
    articles = data["processed"]
    alerts = data["alerts"]
    
    # Show alerts first
    if alerts:
        st.header("🚨 Alerts")
        
        # RED alerts
        red_alerts = [a for a in alerts if a["alert_level"] == "RED"]
        if red_alerts:
            for alert in red_alerts:
                st.error(f"""
                **🔴 HIGH PRIORITY ALERT**
                **{alert['title']}**
                - Company: {alert['company_ticker']}
                - Sentiment: {alert['sentiment'].upper()}
                - Impact: {alert['market_impact'].upper()}
                - Verified: {'✅' if alert['verified'] else '❌'}
                - Source: {alert['source']}
                """)
        
        # YELLOW alerts
        yellow_alerts = [a for a in alerts if a["alert_level"] == "YELLOW"]
        if yellow_alerts:
            for alert in yellow_alerts:
                st.warning(f"""
                **🟡 MEDIUM PRIORITY ALERT**
                **{alert['title']}**
                - Company: {alert['company_ticker']}
                - Sentiment: {alert['sentiment'].upper()}
                - Impact: {alert['market_impact'].upper()}
                - Verified: {'✅' if alert['verified'] else '❌'}
                - Source: {alert['source']}
                """)
    
    # Show all articles
    st.header(f"📰 All Articles ({len(articles)})")
    
    for i, article in enumerate(articles):
        with st.expander(f"📄 {article.get('title', 'Untitled')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Sentiment badge
                sentiment = article.get('sentiment', 'neutral')
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴',
                    'neutral': '🟡'
                }.get(sentiment, '⚪')
                st.metric("Sentiment", f"{sentiment_color} {sentiment.upper()}")
            
            with col2:
                # Topic
                topic = article.get('topic', 'other')
                st.metric("Topic", topic.replace('_', ' ').title())
            
            with col3:
                # Verification status
                verified = article.get('verified', False)
                verification_status = '✅ Verified' if verified else '❌ Unconfirmed'
                st.metric("Verification", verification_status)
            
            # Additional info
            st.write(f"**Source:** {article.get('source', 'Unknown')}")
            st.write(f"**Company:** {article.get('company_ticker', 'Unknown')}")
            st.write(f"**Region:** {article.get('region', 'Unknown')}")
            st.write(f"**Market Impact:** {article.get('market_impact', 'Unknown').upper()}")
            st.write(f"**Alert Level:** {article.get('alert_level', 'GREEN')}")
            
            # Summary
            summary = article.get('summary', 'No summary available')
            st.write("**Summary:**")
            st.write(summary.replace('•', '\n•'))
            
            # Historical context
            if article.get('historical_context'):
                st.write("**Historical Context:**")
                st.info(article['historical_context'])
            
            # Content snippet
            content = article.get('content', '')[:300]
            if content:
                st.write("**Content Preview:**")
                st.write(f"{content}...")


def render_historical_query() -> None:
    """Render the historical query section."""
    st.header("🔍 Ask About History")
    
    query = st.text_input(
        "Enter your question about financial history or market patterns:",
        placeholder="e.g., What happened to Apple stock during earnings season?",
        key="historical_query"
    )
    
    if st.button("🔍 Search Historical Context", key="search_history"):
        if query.strip():
            with st.spinner("🔍 Searching historical context..."):
                try:
                    retriever = RAGRetriever()
                    if retriever.is_available():
                        results = retriever.retrieve(query, top_k=5)
                        
                        if results:
                            st.info(f"Found {len(results)} relevant historical documents:")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"📚 Result {i} (Similarity: {result['similarity_score']:.2f})"):
                                    st.write(f"**Source:** {result['source']}")
                                    st.write("**Content:**")
                                    st.write(result['content'])
                        else:
                            st.warning("No relevant historical documents found. Try different keywords.")
                    else:
                        st.warning("Historical database not available. Please ingest documents first.")
                        
                except Exception as e:
                    st.error(f"Error searching historical context: {str(e)}")
        else:
            st.warning("Please enter a search query.")


if __name__ == "__main__":
    main()