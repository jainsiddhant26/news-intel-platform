# 📰 News Intelligence Platform
> **Web Search + RAG + Multi-Agent Analysis** — Real-time financial news monitoring, sentiment scoring, and investment insight generation at scale.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-green?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Table of Contents
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Who Is This For?](#-who-is-this-for)
- [Business Value & Impact](#-business-value--impact)
- [System Architecture](#-system-architecture)
- [Multi-Agent Design](#-multi-agent-design)
- [RAG Integration](#-rag-integration)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧩 The Problem

### Context
Financial analysts and portfolio managers at mid-to-large firms are responsible for monitoring news, filings, and market signals across **1,000+ companies simultaneously**. This is operationally impossible to do manually — and missing a signal even 2–4 hours late can result in suboptimal trade timing, compliance risk, or reputational damage.

### Pain Points (PM Perspective)

| # | Pain Point | Impact |
|---|---|---|
| 1 | Analysts manually scan 50–100 news sources daily | 80% of research time spent on low-value aggregation |
| 2 | No unified view of news + filings + social signals | Critical context missed across siloed tools |
| 3 | Sentiment is assessed subjectively | Inconsistent risk evaluation across teams |
| 4 | Competitors act on news 2–4 hrs earlier | Direct loss in trade timing and alpha generation |
| 5 | No historical pattern matching on current events | Missed learnings from analogous past events |

### Problem Statement
> *"How might we give portfolio managers a real-time, AI-powered intelligence layer that monitors, classifies, and contextualizes financial news — so they can make faster, more informed investment decisions without increasing headcount?"*

---

## 💡 Our Solution

The **News Intelligence Platform** is a multi-agent AI system that:

1. **Continuously ingests** news from APIs, RSS feeds, and web sources across 1,000+ companies
2. **Classifies and tags** articles by topic, company ticker, region, and impact level
3. **Scores sentiment** using a finance-tuned NLP model (FinBERT)
4. **Cross-verifies** stories across multiple sources before surfacing alerts
5. **Retrieves historical context** via RAG — matching current events to similar past events
6. **Synthesizes** actionable summaries and priority alerts for portfolio managers

This replaces hours of manual scanning with a **sub-minute, always-on intelligence feed**.

---

## 👥 Who Is This For?

| User | Need | How This Helps |
|---|---|---|
| **Portfolio Manager** | Fast, reliable market signals | Gets priority alerts 2–4 hrs before competitors |
| **Research Analyst** | Deep context on company events | RAG surfaces relevant filings + historical analogs |
| **Risk Officer** | Early warning on regulatory/macro events | Negative sentiment + high-impact alerts flagged instantly |
| **Trading Desk** | Real-time event-driven signals | Verified, classified news fed directly to decision workflow |

---

## 📈 Business Value & Impact

Based on the case study baseline:

- ⚡ **50,000** news items processed daily
- 🕐 Alerts generated **2–4 hours before** competitors notice trends
- 📊 Portfolio managers report **15% improvement** in trade timing
- ⏱️ Research analyst time on news monitoring reduced by **80%**
- 📚 RAG corpus: **10M+ historical articles**, SEC filings, earnings calls, analyst reports

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                  │
│         (Live Feed │ Alerts │ RAG Search │ Charts)      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              LangGraph Orchestrator                     │
│   collect → classify → sentiment → verify → synthesize  │
└───┬──────────┬──────────┬──────────┬──────────┬─────────┘
    │          │          │          │          │
┌───▼──┐  ┌───▼──┐  ┌────▼──┐  ┌───▼──┐  ┌───▼──────┐
│Collec│  │Class │  │Sentim │  │Verif │  │Synthesis │
│tion  │  │ific. │  │ment   │  │icat. │  │Agent     │
│Agent │  │Agent │  │Agent  │  │Agent │  │          │
└───┬──┘  └──────┘  └───────┘  └──────┘  └──────────┘
    │
┌───▼──────────────────────────────────────────────────┐
│              Data Sources                            │
│  NewsAPI │ RSS Feeds │ Web Scraping │ SEC EDGAR      │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   RAG Layer                             │
│     ChromaDB Vector Store │ HuggingFace Embeddings      │
│  Historical Articles │ Filings │ Earnings │ Research    │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Multi-Agent Design

Each agent has a **single responsibility** and passes enriched state to the next:

### 1. 📥 Collection Agent
- Sources: NewsAPI, RSS feeds (Reuters, Bloomberg, FT, CNBC, WSJ), web scraping
- Deduplicates by URL hash
- Normalizes schema: `{title, content, url, source, published_at}`

### 2. 🏷️ Classification Agent
- Uses **Groq LLaMA 3.3** for zero-shot classification
- Tags: `topic` (earnings / macro / regulatory / M&A), `company` (ticker), `region` (US / EU / APAC)

### 3. 🧠 Sentiment Agent
- Uses **FinBERT** (finance-tuned BERT, runs locally — no API cost)
- Returns: `sentiment` (positive / negative / neutral) + `confidence score`
- Secondary LLM pass rates `market_impact`: high / medium / low

### 4. ✅ Verification Agent
- Cross-references: requires ≥ 2 independent sources before surfacing alert
- Flags unverified stories as `status: unconfirmed`

### 5. 📝 Synthesis Agent
- Pulls top-3 RAG results (historical analogs) via vector similarity
- Generates a 3-bullet alert summary using Groq
- Triggers push notification if `sentiment = negative` AND `impact = high`

---

## 📚 RAG Integration

The RAG layer gives agents **long-term memory and institutional knowledge**:

| Knowledge Source | Format | Update Frequency |
|---|---|---|
| Historical news archive (10M+ articles) | Chunked text | Daily |
| Company profiles & financial statements | PDF / structured | Quarterly |
| Analyst reports & market research | PDF | As published |
| SEC filings & earnings call transcripts | Text / EDGAR API | Real-time |

**How it works**: When a new article arrives, the retriever finds the top-5 most semantically similar historical events. These are injected into the synthesis prompt so the LLM can say *"This is similar to the 2022 SVB liquidity signal — which preceded a 40% drawdown in regional bank stocks."*

---

## 🛠️ Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Agent Orchestration | LangGraph | Native state machine, conditional edges, agent handoffs |
| LLM | Groq (LLaMA 3.3 70B) | Free tier, <1s latency, 14,400 req/day |
| Sentiment NLP | HuggingFace FinBERT | Finance-tuned, runs locally, no API cost |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, fast, local |
| Vector DB | ChromaDB | Persistent, local, no infra needed |
| News Data | NewsAPI + feedparser | Free tiers available |
| Frontend | Streamlit | Fastest path to a working dashboard |
| Config Management | python-dotenv | Industry standard for secrets |

---

## 📁 Project Structure

```
news-intel-platform/
├── agents/
│   ├── __init__.py
│   ├── collection_agent.py       # RSS, NewsAPI, scraping
│   ├── classification_agent.py   # Topic, company, region tagging
│   ├── sentiment_agent.py        # FinBERT + impact scoring
│   ├── verification_agent.py     # Multi-source cross-check
│   └── synthesis_agent.py        # Summary generation + alerts
├── rag/
│   ├── __init__.py
│   ├── ingest.py                 # Chunk, embed, store documents
│   ├── retriever.py              # Semantic similarity search
│   └── vectorstore/              # ChromaDB persistent storage (git-ignored)
├── data/
│   ├── raw/                      # Downloaded news/filings (git-ignored)
│   └── sample/                   # Sample docs for testing
├── tests/
│   ├── test_collection.py
│   ├── test_sentiment.py
│   └── test_rag.py
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
├── orchestrator.py               # LangGraph state machine
├── app.py                        # Streamlit dashboard entry point
├── config.py                     # App configuration + constants
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for secrets (commit this)
├── .env                          # Actual secrets (NEVER commit this)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git
- Free accounts at: [NewsAPI](https://newsapi.org) and [Groq Console](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/news-intel-platform.git
cd news-intel-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Now edit .env with your actual API keys
```

### 5. Ingest Sample Documents into RAG
```bash
python rag/ingest.py --source data/sample/
```

### 6. Run the Platform
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in:

```env
# LLM
GROQ_API_KEY=your_groq_key_here

# News Data
NEWS_API_KEY=your_newsapi_key_here

# App Settings
MONITORED_TICKERS=AAPL,MSFT,GOOGL,TSLA,JPM
ALERT_SENTIMENT_THRESHOLD=negative
ALERT_IMPACT_THRESHOLD=high
REFRESH_INTERVAL_SECONDS=300
```

---

## 🗺️ Roadmap

### v0.1 — MVP (Current Sprint)
- [x] Project scaffolding + README
- [ ] Collection Agent (NewsAPI + RSS)
- [ ] Classification Agent (Groq)
- [ ] Sentiment Agent (FinBERT)
- [ ] Basic Streamlit feed

### v0.2 — RAG + Verification
- [ ] ChromaDB ingestion pipeline
- [ ] Historical similarity retrieval
- [ ] Verification Agent (multi-source check)
- [ ] Synthesis Agent with RAG context

### v0.3 — Production Hardening
- [ ] LangGraph full orchestration
- [ ] Alert push notifications (email/Slack)
- [ ] Ticker watchlist management UI
- [ ] CI/CD via GitHub Actions
- [ ] Docker containerization

### v1.0 — Scale
- [ ] Support for 1,000+ company monitoring
- [ ] Real-time WebSocket feed
- [ ] User authentication
- [ ] Export alerts to CSV / integrate with Bloomberg Terminal

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Commit your changes**: `git commit -m "feat: add your feature"`
4. **Push to your fork**: `git push origin feature/your-feature-name`
5. **Open a Pull Request** — describe what you built and why

### Commit Message Convention
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `refactor:` — code restructuring
- `test:` — adding/updating tests

### Code Style
- Follow PEP 8
- Run `black .` before committing
- All new agents must have a corresponding test in `/tests`

---

## 🔒 .gitignore Essentials

Make sure these are in your `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
rag/vectorstore/
data/raw/
.DS_Store
```

---


## 🙋 Author

Built by **Siddhant Jain** as a portfolio project demonstrating production-grade AI system design.  
Connect on [LinkedIn](https://www.linkedin.com/in/jainsiddhant26
)

---

> *"The goal is not to replace analysts — it's to give them superpowers."*
