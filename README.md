# 📰 News Intelligence Platform

Real-time AI news monitoring that tells you what matters — before the market moves.

This project was built almost entirely through **vibe coding**: describing intent in plain
English to Windsurf's Cascade AI, which wrote and iterated on the code. The goal is to
show how a PM can steer an AI-first product from idea to working tool in a single flow.

---

## 🎯 Problem & Outcome

### The Problem

PMs and analysts spend hours every day:
- Checking multiple news sites and terminals
- Manually filtering for portfolio-relevant stories
- Guessing sentiment and market impact
- Losing time before they can act

### The Outcome

This platform does that work automatically in ~60 seconds:
- Ingests live news for selected tickers
- Classifies each article (topic, ticker, region)
- Scores sentiment and market impact
- Generates RED / YELLOW alerts
- Lets you query past events via natural language

You get a **single dashboard** showing "what matters now" plus historical context.

---

## ✨ What the Product Does

| Area | What you get |
|------|--------------|
| 📡 Live feed | 15–20 fresh articles per run from NewsAPI + Yahoo/Reuters/CNBC RSS |
| 🚨 Alerts | High-priority events surfaced at the top (RED/YELLOW) |
| 😊 Sentiment | FinBERT-based positive / negative / neutral with confidence score |
| 📊 Impact | Estimated market impact: HIGH / MEDIUM / LOW |
| ✅ Verification | Basic fact-check flag (verified / unverified) |
| 🔍 History search | Ask questions like "What happened with TSLA last quarter?" |

The UI is a Streamlit app with:
- Ticker selection in the sidebar
- "Run Pipeline" button
- Alerts section
- "All Articles" list with expanders
- "Ask About History" input backed by a RAG system

---

## 🏗️ System Overview

```text
NewsAPI + Yahoo Finance RSS + CNBC RSS + Reuters RSS
                    ↓
            CollectionAgent
                    ↓
         ClassificationAgent (Groq / LLaMA)
                    ↓
          SentimentAgent (FinBERT)
                    ↓
          SynthesisAgent (Groq / LLaMA)
                    ↓
         VerificationAgent (Groq / LLaMA)
                    ↓
       ChromaDB Vector Store (RAG)
                    ↓
            Streamlit Dashboard
```

Main components:
- **Agents** — modular services for collection, classification, sentiment, synthesis, verification
- **Orchestrator** — runs agents sequentially for a batch of articles
- **RAG Layer** — stores historical articles and serves semantic search
- **Streamlit UI** — the user-facing dashboard

---

## 🪜 Build Story — Step by Step

This is how the project was actually created using vibe coding.

---

### Step 1 — Define the Vision

We started with a natural-language brief to the AI:

> "Create a news intelligence platform for financial markets.
> It should monitor stock tickers, analyze news sentiment, detect high-impact events,
> and surface alerts. Use Python with a Streamlit UI."

Cascade responded by scaffolding:
- Project folders (`agents/`, `rag/`, `data/`, `tests/`)
- Core files (`app.py`, `orchestrator.py`, `config.py`)
- A `.env.example` for safe key management
- A `requirements.txt` with initial dependencies

**PM takeaway:** a clear, outcome-focused brief is enough for AI to generate a working skeleton.

---

### Step 2 — Wire Up External Services

We connected three key services:
- **Groq** — LLaMA 3.3 70B for classification, summary generation, and verification
- **NewsAPI** — structured news search by ticker keyword
- **HuggingFace** — FinBERT sentiment model + sentence-transformer embeddings (both run locally)

`config.py` centralizes API keys, default tickers (`AAPL`, `MSFT`, `GOOGL`, `TSLA`, `JPM`), and model names.

**Why this matters to a PM:** swapping models or providers is a config change, not a refactor.

---

### Step 3 — Collection Agent (Data In)

Goal: "Give me a clean list of relevant, deduplicated articles per run."

The **CollectionAgent**:
- Calls NewsAPI for all selected tickers
- Reads RSS feeds from Yahoo Finance, CNBC, Reuters
- Normalizes fields (title, url, source, published_at, content)
- Deduplicates by URL
- Caps total volume to keep latency predictable

From a product lens: this is the data intake valve — everything downstream depends on this shape.

---

### Step 4 — Classification Agent (Understand the Article)

Goal: "Tell me what this news is about and who it impacts."

The **ClassificationAgent** (Groq / LLaMA):
- Predicts topic (earnings, regulation, macro, product, etc.)
- Identifies primary company ticker (AAPL, TSLA, etc.)
- Assigns region (US, EU, Global, etc.)
- On any API error, returns safe defaults immediately — never blocks the pipeline

This converts unstructured news text into structured, filterable data.

---

### Step 5 — Sentiment Agent (Good or Bad?)

Goal: "Is this positive, negative, or neutral — and how strongly?"

The **SentimentAgent**:
- Uses **FinBERT** (a BERT model fine-tuned on financial text) to label each article
- Outputs sentiment label + confidence score
- Calls LLaMA to estimate market impact: HIGH / MEDIUM / LOW
- Falls back to neutral / low impact on any failure

Sentiment + impact together drive the alert severity logic.

---

### Step 6 — Synthesis Agent (Human-Readable Insight)

Goal: "Give a PM a 5-second read on this article."

The **SynthesisAgent**:
- Generates a 2–3 sentence plain-English summary
- Pulls related historical context from the RAG layer
- Assigns alert level:
  - 🔴 **RED** — high impact + negative sentiment
  - 🟡 **YELLOW** — high impact or ambiguous risk
  - 🟢 **GREEN** — everything else

This is what populates the "Alerts" section at the top of the dashboard.

---

### Step 7 — Verification Agent (Trust, But Check)

Goal: "Don't surface claims that don't hold up."

The **VerificationAgent**:
- Uses LLaMA to cross-check headline claims against article body
- Marks each article `verified = True/False`
- Keeps unverified items visible but clearly flagged

This is the foundation for future trust scoring and multi-source confirmation.

---

### Step 8 — RAG Layer (Historical Memory)

Goal: "Connect today's news to related past events automatically."

RAG stack:
- **sentence-transformers/all-MiniLM-L6-v2** for article embeddings
- **ChromaDB** as the persistent vector store

Two scripts:
- `ingest.py` — indexes processed articles
- `retriever.py` — semantic search for the "Ask About History" feature

PM value: the platform becomes more useful over time as it accumulates market event history.

---

### Step 9 — Orchestrator (Glue Everything Together)

Early versions used a LangGraph state machine, which caused an infinite loop on article 1.

We replaced it with a simple sequential Python loop:

```python
for article in articles:
    article = classification_agent.run(article)
    article = sentiment_agent.run(article)
    article = synthesis_agent.run(article)
    processed.append(article)

processed = verification_agent.run(processed)
```

**Lesson:** for linear flows, a plain loop beats a complex graph framework every time.

---

### Step 10 — Dashboard & UX

Goal: "Make this usable by a PM in under a minute."

Design decisions:
- **Always-on UI** — Streamlit loads even if backend imports fail (lazy imports + try/except)
- **Single main CTA** — "Run Pipeline" triggers everything
- **Clear layout:**
  - Top: RED and YELLOW alerts
  - Middle: all 15–20 articles with expandable detail
  - Bottom: free-form historical search

---

## 🧪 Constraints & How We Handled Them

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| App crashed on startup | Top-level imports failing | Moved heavy imports inside the button handler |
| Pipeline looped forever on article 1 | LangGraph state machine bug | Replaced with a simple `for` loop |
| Groq 429 rate limit errors | Too many LLM calls in quick succession | Reduced article volume; immediate fallback on any error |
| Packages installed in wrong environment | System Anaconda vs project venv | Activated venv first before every install |
| `CollectionAgent.run()` signature error | Method didn't accept tickers param | Updated to `run(self, tickers=None)` |

**PM takeaway:** operational constraints (rate limits, latency) directly shape product behavior. Build fallbacks early, not after.

---

## 🚀 How to Run It

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com)
- Free [NewsAPI key](https://newsapi.org)

### Setup

```bash
# 1. Clone
git clone https://github.com/jainsiddhant26/news-intel-platform.git
cd news-intel-platform

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure keys
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=...
# NEWS_API_KEY=...

# 5. Launch
streamlit run app.py
```

Open `http://localhost:8501` → Select tickers → Click **🚀 Run Pipeline**

---

## 📁 Project Structure

```text
news-intel-platform/
├── agents/
│   ├── collection_agent.py      # Multi-source news ingestion
│   ├── classification_agent.py  # Topic / ticker / region tagging
│   ├── sentiment_agent.py       # FinBERT sentiment + impact scoring
│   ├── synthesis_agent.py       # Summaries + alert level
│   └── verification_agent.py    # Claim verification
├── rag/
│   └── vectorstore/             # ChromaDB persistent store
├── data/                        # Cached article data
├── app.py                       # Streamlit dashboard
├── orchestrator.py              # Sequential pipeline runner
├── config.py                    # Centralized config + env loading
├── ingest.py                    # RAG ingestion script
├── retriever.py                 # RAG semantic search
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Orchestration | Python class (`NewsOrchestrator`) |
| LLM | Groq LLaMA 3.3 70B Versatile |
| Sentiment | FinBERT (ProsusAI/finbert) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| News Sources | NewsAPI + RSS (Yahoo Finance, CNBC, Reuters) |
| Language | Python 3.12 |
| AI Editor | Windsurf (Cascade SWE-1.5) |

---

## 💡 Takeaways for PMs

- You can steer a non-trivial AI product without writing a single line of code manually.
- The key PM skills in vibe coding are:
  - Writing clear, outcome-focused briefs
  - Making trade-offs between accuracy, latency, and cost
  - Describing bugs clearly so the AI can diagnose them
  - Knowing when to simplify (replace LangGraph with a `for` loop)

This project is a reusable pattern for any "AI copilot over unstructured text" use case:
customer tickets, incident reports, CRM notes, product reviews, support queues — the pipeline is identical.

---

## 📄 License

MIT — free to use, learn from, and extend.
